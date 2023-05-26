# Copyright 2023 The Wordcab Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ASR Service module that handle all AI interactions."""

import asyncio
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Union

import numpy as np
import torch
from loguru import logger

from wordcab_transcribe.config import settings
from wordcab_transcribe.services.align_service import AlignService
from wordcab_transcribe.services.diarize_service import DiarizeService
from wordcab_transcribe.services.post_processing_service import PostProcessingService
from wordcab_transcribe.services.transcribe_service import TranscribeService
from wordcab_transcribe.services.vad_service import VadService
from wordcab_transcribe.utils import delete_file, enhance_audio, format_segments


class ASRService(ABC):
    """Base ASR Service module that handle all AI interactions and batch processing."""

    def __init__(self) -> None:
        """Initialize the ASR Service.

        This class is not meant to be instantiated. Use the subclasses instead.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Do we have a GPU? If so, use it!
        self.sample_rate = 16000  # The sample rate to use for inference for all audio files (Hz)

        self.queues = None  # the queue to store requests
        self.queue_locks = None  # the lock to access the queue
        self.needs_processing = None  # the flag to indicate if the queue needs processing
        self.needs_processing_timer = None  # the timer to schedule processing

    @abstractmethod
    def schedule_processing_if_needed(self) -> None:
        """Method to schedule processing if needed."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    async def process_input(self) -> None:
        """Process the input request and return the result."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    async def runner(self) -> None:
        """Runner method to process the queue."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def process_batch(self) -> None:
        """Process the batch of requests."""
        raise NotImplementedError("This method should be implemented in a subclass.")


class ASRAsyncService(ASRService):
    """ASR Service module for async endpoints."""

    def __init__(self) -> None:
        """Initialize the ASRAsyncService class."""
        super().__init__()

        self.batch_size = {"transcription": 8, "diarization": 1, "alignment": 1}
        self.thread_executors = {
            "transcription": ThreadPoolExecutor(max_workers=self.batch_size["transcription"]),
            "diarization": ThreadPoolExecutor(max_workers=self.batch_size["diarization"]),
            "alignment": ThreadPoolExecutor(max_workers=self.batch_size["alignment"]),
            "post_processing": ThreadPoolExecutor(max_workers=1),
        }
        self.services = {
            "transcription": TranscribeService(
                model_path=settings.whisper_model,
                compute_type=settings.compute_type,
                device=self.device,
            ),
            "diarization": DiarizeService(
                domain_type=settings.nemo_domain_type,
                storage_path=settings.nemo_storage_path,
                output_path=settings.nemo_output_path,
                device=self.device,
            ),
            "alignment": AlignService(self.device),
            "post_processing": PostProcessingService(),
            "vad": VadService(),
        }
        self.queues = {"transcription": [], "diarization": [], "alignment": []}
        self.queue_locks = {
            "transcription": asyncio.Lock(),
            "diarization": asyncio.Lock(),
            "alignment": asyncio.Lock(),
            "post_processing": asyncio.Lock(),
        }
        self.needs_processing = {
            "transcription": asyncio.Event(),
            "diarization": asyncio.Event(),
            "alignment": asyncio.Event(),
            "post_processing": asyncio.Event(),
        }
        self.needs_processing_timer = {
            "transcription": None,
            "diarization": None,
            "alignment": None,
            "post_processing": None,
        }
        self.dual_channel_transcribe_options = {
            "beam_size": 5,
            "patience": 1,
            "length_penalty": 1,
            "suppress_blank": False,
            "word_timestamps": True,
            "temperature": 0.0,
        }

    def schedule_processing_if_needed(self, task_type: str) -> None:
        """
        Method to schedule processing if needed for a specific task queue.

        Args:
            task_type (str): The task type to schedule processing for.
        """
        if len(self.queues[task_type]) >= settings.max_batch_size:
            self.needs_processing[task_type].set()
        elif self.queues[task_type]:
            self.needs_processing_timer[task_type] = asyncio.get_event_loop().call_at(
                self.queues[task_type][0]["time"] + settings.max_wait,
                self.needs_processing[task_type].set,
            )

    async def process_input(
        self,
        filepath: Union[str, Tuple[str]],
        alignment: bool,
        diarization: bool,
        dual_channel: bool,
        source_lang: str,
        word_timestamps: bool,
    ) -> List[dict]:
        """
        Process the input request and return the result.

        Args:
            filepath (Union[str, Tuple[str]]): Path to the audio file.
            alignment (bool): Whether to do alignment or not.
            diarization (bool): Whether to do diarization or not.
            dual_channel (bool): Whether to do dual channel or not.
            source_lang (str): Source language of the audio file.
            word_timestamps (bool): Whether to return word timestamps or not.

        Returns:
            List[dict]: List of speaker segments.
        """
        task = {
            "input": filepath,
            "dual_channel": dual_channel,
            "source_lang": source_lang,
            "word_timestamps": word_timestamps,
            "post_processed": False,
            "transcription_done": asyncio.Event(),
            "diarization_done": asyncio.Event(),
            "alignment_done": asyncio.Event(),
            "done_event": asyncio.Event(),
            "time": asyncio.get_event_loop().time(),
        }

        async with self.queue_locks["transcription"]:
            self.queues["transcription"].append(task)
            self.schedule_processing_if_needed("transcription")

        if alignment and dual_channel is False:
            async with self.queue_locks["alignment"]:
                self.queues["alignment"].append(task)
                self.schedule_processing_if_needed("alignment")
        else:
            task["alignment_done"].set()

        if diarization and dual_channel is False:
            async with self.queue_locks["diarization"]:
                self.queues["diarization"].append(task)
                self.schedule_processing_if_needed("diarization")
        else:
            task["diarization_done"].set()

        await asyncio.gather(
            task["transcription_done"].wait(),
            task["diarization_done"].wait(),
            task["alignment_done"].wait(),
            return_exceptions=True,
        )

        # Check if there is any exception in the task
        if task["transcription_done"].exception():
            raise Exception(task["transcription_done"].exception())
        elif task["diarization_done"].exception():
            raise Exception(task["diarization_done"].exception())
        elif task["alignment_done"].exception():
            raise Exception(task["alignment_done"].exception())
        else:
            self.queues["post_processing"].append(task)
            self.schedule_processing_if_needed("post_processing")

        await task["done_event"].wait()

        return task["result"]

    async def runner(self, task_type: str) -> None:
        """Runner method to process the queue."""
        while True:
            await self.needs_processing[task_type].wait()
            self.needs_processing[task_type].clear()

            if self.needs_processing_timer[task_type] is not None:
                self.needs_processing_timer[task_type].cancel()
                self.needs_processing_timer[task_type] = None

            async with self.queue_locks[task_type]:
                if self.queues[task_type]:
                    longest_wait = (
                        asyncio.get_event_loop().time() - self.queues[task_type][0]["time"]
                    )
                    logger.debug(f"[{task_type}] longest wait: {longest_wait}")
                else:
                    longest_wait = None
                file_batch = self.queues[task_type][: self.batch_size[task_type]]
                del self.queue[task_type][: len(file_batch)]
                self.schedule_processing_if_needed(task_type)

            asyncio.create_task(self.process_task(file_batch, task_type))

            # try:
            #     results = await asyncio.get_event_loop().run_in_executor(
            #         self.thread_executors[task_type], self.process_task, file_batch, task_type
            #     )

            #     for task, result in zip(file_batch, results):  # noqa B905
            #         task["result"] = result
            #         task["done_event"].set()

            #     del results
            #     del file_batch

            # except Exception as e:
            #     logger.error(f"Error processing batch: {e}\n{traceback.format_exc()}")
            #     for task in file_batch:  # Error handling
            #         task["result"] = e
            #         task["done_event"].set()

    def process_batch(self, file_batch: List[dict], task_type: str) -> List[dict]:
        """
        Process a batch of requests.

        Args:
            file_batch (List[dict]): List of requests to process with their respective parameters.
            task_type (str): The type of task to process.

        Returns:
            List[dict]: List of results.
        """
        results: List[dict] = []
        for task in file_batch:
            filepath: Union[str, Tuple[str]] = task["input"]
            alignment: bool = task["alignment"]  # ignored if dual_channel is True
            diarization: bool = task["diarization"]  # ignored if dual_channel is True
            dual_channel: bool = task["dual_channel"]
            source_lang: str = task["source_lang"]
            word_timestamps: bool = task["word_timestamps"]

            if dual_channel:
                utterances = self._process_dual_channel(
                    filepath, source_lang, word_timestamps
                )
            else:
                utterances = self._process_single_channel(
                    filepath, alignment, diarization, source_lang, word_timestamps
                )

            results.append(utterances)

        return results

    def transcribe_dual_channel(
        self,
        source_lang: str,
        filepath: str,
        speaker_label: int,
    ) -> List[List[dict]]:
        """
        Transcribe multiple segments of audio in the dual channel mode.

        Args:
            source_lang (str): Source language of the audio file.
            filepath (str): Path to the audio file.
            speaker_label (int): Speaker label.

        Returns:
            List[List[dict]]: List of grouped transcribed segments.
        """
        enhanced_filepath = enhance_audio(
            filepath,
            speaker_label=speaker_label,
            apply_agc=True,
            apply_bandpass=False,
        )
        grouped_segments, audio = self.vad_service(enhanced_filepath)
        delete_file(enhanced_filepath)

        final_transcript = []
        silence_padding = torch.from_numpy(np.zeros(int(3 * self.sample_rate))).float()

        for ix, group in enumerate(grouped_segments):
            try:
                audio_segments = []
                for segment in group:
                    segment_start = segment["start"]
                    segment_end = segment["end"]
                    audio_segment = audio[segment_start:segment_end]
                    audio_segments.append(audio_segment)
                    audio_segments.append(silence_padding)

                tensors = torch.cat(audio_segments)
                temp_filepath = f"{filepath}_{speaker_label}_{ix}.wav"
                self.vad_service.save_audio(
                    temp_filepath, tensors, sampling_rate=self.sample_rate
                )

                segments, _ = self.transcribe_model.model.transcribe(
                    audio=temp_filepath,
                    language=source_lang,
                    **self.dual_channel_transcribe_options,
                )
                segments = list(segments)

                group_start = group[0]["start"]

                for segment in segments:
                    segment_dict = {
                        "start": None,
                        "end": None,
                        "text": segment.text,
                        "words": [],
                        "speaker": speaker_label,
                    }

                    for word in segment.words:
                        word_start_adjusted = (
                            group_start / self.sample_rate
                        ) + word.start
                        word_end_adjusted = (group_start / self.sample_rate) + word.end
                        segment_dict["words"].append(
                            {
                                "start": word_start_adjusted,
                                "end": word_end_adjusted,
                                "text": word.word,
                            }
                        )

                        if (
                            segment_dict["start"] is None
                            or word_start_adjusted < segment_dict["start"]
                        ):
                            segment_dict["start"] = word_start_adjusted
                        if (
                            segment_dict["end"] is None
                            or word_end_adjusted > segment_dict["end"]
                        ):
                            segment_dict["end"] = word_end_adjusted

                    final_transcript.append(segment_dict)

                delete_file(temp_filepath)

            except Exception as e:
                logger.error(f"Dual channel trasncription error: {e}")
                pass

        return final_transcript

    def _process_single_channel(
        self,
        filepath: str,
        alignment: bool,
        diarization: bool,
        source_lang: str,
        word_timestamps: bool,
    ) -> List[dict]:
        """
        Process a single channel audio file.

        Args:
            filepath (str): Path to the audio file.
            alignment (bool): Whether to align the segments.
            diarization (bool): Whether to diarize the audio file.
            source_lang (str): Source language of the audio file.
            word_timestamps (bool): Whether to include word timestamps.

        Returns:
            List[dict]: List of speaker segments.
        """
        if alignment:
            _segments = self.transcribe_model(
                filepath, source_lang, word_timestamps=True
            )
            segments = self.align_model(filepath, _segments, source_lang)
        else:
            segments = self.transcribe_model(
                filepath, source_lang, word_timestamps=True
            )

        # Format the segments: the main purpose is to remove extra spaces and
        # to format word_timestamps like the alignment model does if alignment is False
        formatted_segments = format_segments(
            segments, alignment=alignment, word_timestamps=word_timestamps
        )

        if diarization:
            speaker_timestamps = self.diarize_model(filepath)
            utterances = self.post_processing_service.single_channel_postprocessing(
                transcript_segments=formatted_segments,
                speaker_timestamps=speaker_timestamps,
                word_timestamps=word_timestamps,
            )
        else:
            utterances = formatted_segments

        return utterances

    def _process_dual_channel(
        self, filepath: Tuple[str], source_lang: str, word_timestamps: bool
    ) -> List[dict]:
        """
        Process a dual channel audio file.

        Args:
            filepath (Tuple[str]): Tuple of paths to the split audio files.
            source_lang (str): Source language of the audio file.
            word_timestamps (bool): Whether to include word timestamps.

        Returns:
            List[dict]: List of speaker segments.
        """
        left_channel, right_channel = filepath

        left_transcribed_segments = self.transcribe_dual_channel(
            source_lang,
            filepath=left_channel,
            speaker_label=0,
        )
        right_transcribed_segments = self.transcribe_dual_channel(
            source_lang,
            filepath=right_channel,
            speaker_label=1,
        )

        utterances = self.post_processing_service.dual_channel_postprocessing(
            left_segments=left_transcribed_segments,
            right_segments=right_transcribed_segments,
            word_timestamps=word_timestamps,
        )

        return utterances


class ASRLiveService():
    """ASR Service module for live endpoints."""

    def __init__(self) -> None:
        """Initialize the ASRLiveService class."""
        super().__init__()
