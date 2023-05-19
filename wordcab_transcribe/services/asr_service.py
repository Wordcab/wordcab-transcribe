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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Tuple, Union

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


class ASRService:
    """Base ASR Service module that handle all AI interactions and batch processing."""

    def __init__(self) -> None:
        """Initialize the ASR Service.

        This class is not meant to be instantiated.
        Use the subclasses instead: ASRAsyncService or ASRLiveService.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.thread_executor = ThreadPoolExecutor(max_workers=4)
        self.sample_rate = 16000

        # Multi requests support
        self.queue = []
        self.queue_lock = asyncio.Lock()
        self.needs_processing = None
        self.needs_processing_timer = None

        self.max_batch_size = (
            settings.batch_size
        )  # Max number of requests to process at once
        self.max_wait = (
            settings.max_wait
        )  # Max time to wait for more requests before processing

    def schedule_processing_if_needed(self) -> None:
        """Method to schedule processing if needed."""
        if len(self.queue) >= self.max_batch_size:
            self.needs_processing.set()
        elif self.queue:
            self.needs_processing_timer = asyncio.get_event_loop().call_at(
                self.queue[0]["time"] + self.max_wait, self.needs_processing.set
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
            "alignment": alignment,
            "diarization": diarization,
            "dual_channel": dual_channel,
            "source_lang": source_lang,
            "word_timestamps": word_timestamps,
            "done_event": asyncio.Event(),
            "time": asyncio.get_event_loop().time(),
        }

        async with self.queue_lock:
            self.queue.append(task)
            self.schedule_processing_if_needed()

        await task["done_event"].wait()

        return task["result"]

    async def runner(self) -> None:
        """Runner method to process the queue."""
        self.needs_processing = asyncio.Event()
        while True:
            await self.needs_processing.wait()
            self.needs_processing.clear()

            if self.needs_processing_timer is not None:
                self.needs_processing_timer.cancel()
                self.needs_processing_timer = None

            async with self.queue_lock:
                if self.queue:
                    longest_wait = (
                        asyncio.get_event_loop().time() - self.queue[0]["time"]
                    )
                    logger.debug(f"Longest wait: {longest_wait}")
                else:
                    longest_wait = None
                file_batch = self.queue[: self.max_batch_size]
                del self.queue[: len(file_batch)]
                self.schedule_processing_if_needed()

            try:
                results = await asyncio.get_event_loop().run_in_executor(
                    self.thread_executor, self.process_batch, file_batch
                )

                for task, result in zip(file_batch, results):  # noqa B905
                    task["result"] = result
                    task["done_event"].set()

                del results
                del file_batch

            except Exception as e:
                logger.error(f"Error processing batch: {e}\n{traceback.format_exc()}")
                for task in file_batch:  # Error handling
                    task["result"] = e
                    task["done_event"].set()

    def process_batch(self) -> None:
        """Process the batch of requests."""
        raise NotImplementedError("This method should be implemented in a subclass.")


class ASRAsyncService(ASRService):
    """ASR Service module for async endpoints."""

    def __init__(self) -> None:
        """Initialize the ASRAsyncService class."""
        super().__init__()

        self.dual_channel_transcribe_options = {
            "beam_size": 5,
            "patience": 1,
            "length_penalty": 1,
            "suppress_blank": False,
            "word_timestamps": True,
            "temperature": 0.0,
        }

        self.align_model = AlignService(self.device)
        self.transcribe_model = TranscribeService(
            model_path=settings.whisper_model,
            compute_type=settings.compute_type,
            device=self.device,
        )
        self.diarize_model = DiarizeService(
            domain_type=settings.nemo_domain_type,
            storage_path=settings.nemo_storage_path,
            output_path=settings.nemo_output_path,
            device=self.device,
        )
        self.post_processing_service = PostProcessingService()
        self.vad_service = VadService()

    def transcribe(self, filepath: str, source_lang: str, **kwargs: Any) -> List[dict]:
        """
        Transcribe the audio file using the TranscribeService class.

        Args:
            filepath (str): Path to the audio file.
            source_lang (str): Source language of the audio file.
            kwargs (Any): Additional arguments to pass to the transcribe method.

        Returns:
            List[dict]: List of speaker segments.
        """
        segments = self.transcribe_model(filepath, source_lang, **kwargs)

        return segments

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
                print(f"Error: {e}")
                pass

        return final_transcript

    def align(
        self, filepath: str, segments: List[dict], source_lang: str
    ) -> List[dict]:
        """
        Align the segments using the AlignmentService class.

        Args:
            filepath (str): Path to the audio file.
            segments (List[dict]): List of speaker segments.
            source_lang (str): Source language of the audio file.

        Returns:
            List[dict]: List of aligned speaker segments.
        """
        aligned_segments = self.align_model(filepath, segments, source_lang)

        return aligned_segments

    def diarize(self, filepath: str) -> List[dict]:
        """
        Diarize the audio file using the DiarizeService class.

        Args:
            filepath (str): Path to the audio file.

        Returns:
            List[dict]: List of speaker timestamps.
        """
        speaker_timestamps = self.diarize_model(filepath)

        return speaker_timestamps

    def process_batch(self, file_batch: List[dict]) -> List[dict]:
        """
        Process a batch of requests.

        Args:
            file_batch (List[dict]): List of requests to process with their respective parameters.

        Returns:
            List[dict]: List of results.
        """
        results: List[dict] = []
        for task in file_batch:
            filepath: Union[str, Tuple[str]] = task["input"]
            alignment: bool = task["alignment"]  # ignored if diarization is True
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
        segments = self.transcribe(filepath, source_lang, word_timestamps=word_timestamps)

        if alignment:
            formatted_segments = self.align(filepath, segments, source_lang)
        else:
            formatted_segments = format_segments(segments, word_timestamps=word_timestamps)

        if diarization:
            speaker_timestamps = self.diarize(filepath)
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
        )

        return utterances


class ASRLiveService(ASRService):
    """ASR Service module for live endpoints."""

    def __init__(self) -> None:
        """Initialize the ASRLiveService class."""
        super().__init__()
