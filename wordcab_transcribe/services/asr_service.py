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
import functools
import os
import traceback
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
from loguru import logger

from wordcab_transcribe.config import settings
from wordcab_transcribe.logging import time_and_tell
from wordcab_transcribe.services.align_service import AlignService
from wordcab_transcribe.services.diarize_service import DiarizeService
from wordcab_transcribe.services.gpu_service import GPUService
from wordcab_transcribe.services.post_processing_service import PostProcessingService
from wordcab_transcribe.services.transcribe_service import TranscribeService
from wordcab_transcribe.services.vad_service import VadService
from wordcab_transcribe.utils import format_segments, read_audio


class ASRService(ABC):
    """Base ASR Service module that handle all AI interactions and batch processing."""

    def __init__(self) -> None:
        """Initialize the ASR Service.

        This class is not meant to be instantiated. Use the subclasses instead.
        """
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Do we have a GPU? If so, use it!
        self.num_gpus = torch.cuda.device_count() if self.device == "cuda" else 0
        self.num_cpus = os.cpu_count()

        self.sample_rate = (
            16000  # The sample rate to use for inference for all audio files (Hz)
        )

        self.queues = None  # the queue to store requests
        self.queue_locks = None  # the locks to access the queues
        self.needs_processing = (
            None  # the flag to indicate if the queue needs processing
        )
        self.needs_processing_timer = None  # the timer to schedule processing

    @abstractmethod
    async def process_input(self) -> None:
        """Process the input request by creating a task and adding it to the appropriate queues."""
        raise NotImplementedError("This method should be implemented in subclasses.")


class ASRAsyncService(ASRService):
    """ASR Service module for async endpoints."""

    def __init__(self) -> None:
        """Initialize the ASRAsyncService class."""
        super().__init__()

        if self.num_gpus > 1 and self.device == "cuda":
            device_index = list(range(self.num_gpus))
        else:
            device_index = [0]

        self.gpu_handler = GPUService(device=self.device, device_index=device_index)

        self.services: dict = {
            "transcription": TranscribeService(
                model_path=settings.whisper_model,
                compute_type=settings.compute_type,
                device=self.device,
                device_index=device_index,
            ),
            "diarization": DiarizeService(
                domain_type=settings.nemo_domain_type,
                storage_path=settings.nemo_storage_path,
                output_path=settings.nemo_output_path,
                device=self.device,
                device_index=device_index,
            ),
            "alignment": AlignService(self.device),
            "post_processing": PostProcessingService(),
            "vad": VadService(),
        }
        self.dual_channel_transcribe_options: dict = {
            "beam_size": 5,
            "patience": 1,
            "length_penalty": 1,
            "suppress_blank": False,
            "word_timestamps": True,
            "temperature": 0.0,
        }

    @time_and_tell
    async def process_input(
        self,
        filepath: Union[str, Tuple[str, str]],
        alignment: bool,
        diarization: bool,
        dual_channel: bool,
        source_lang: str,
        timestamps_format: str,
        use_batch: bool,
        vocab: List[str],
        word_timestamps: bool,
    ) -> Union[Tuple[List[dict], float], Exception]:
        """Process the input request and return the results.

        This method will create a task and add it to the appropriate queues.
        All tasks are added to the transcription queue, but will be added to the
        alignment and diarization queues only if the user requested it.
        Each step will be processed asynchronously and the results will be returned
        and stored in separated keys in the task dictionary.

        Args:
            filepath (Union[str, Tuple[str, str]]): Path to the audio file or tuple of paths to the audio files.
            alignment (bool): Whether to do alignment or not.
            diarization (bool): Whether to do diarization or not.
            dual_channel (bool): Whether to do dual channel or not.
            source_lang (str): Source language of the audio file.
            timestamps_format (str): Timestamps format to use.
            use_batch (bool): Whether to use batch processing or not.
            vocab (List[str]): List of words to use for the vocabulary.
            word_timestamps (bool): Whether to return word timestamps or not.

        Returns:
            Union[Tuple[List[dict], float], Exception]: The final transcription result associated with the audio
                duration or an exception.
        """
        if isinstance(filepath, tuple):
            audio, duration = [], []
            for path in filepath:
                _audio, _duration = read_audio(path)

                audio.append(_audio)
                duration.append(_duration)

            audio = tuple(audio)
            duration = sum(duration) / len(duration)

        else:
            audio, duration = read_audio(filepath)

        task = {
            "input": audio,
            "alignment": alignment,
            "diarization": diarization,
            "dual_channel": dual_channel,
            "source_lang": source_lang,
            "timestamps_format": timestamps_format,
            "use_batch": use_batch,
            "vocab": vocab,
            "word_timestamps": word_timestamps,
            "transcription_result": None,
            "transcription_done": asyncio.Event(),
            "diarization_result": None,
            "diarization_done": asyncio.Event(),
            "alignment_result": None,
            "alignment_done": asyncio.Event(),
            "post_processing_result": None,
            "post_processing_done": asyncio.Event(),
            "time": asyncio.get_event_loop().time(),
        }

        # Pick the first available GPU for the task
        gpu_index = await self.gpu_handler.get_device() if self.device == "cuda" else 0

        asyncio.get_event_loop().run_in_executor(
            None, functools.partial(self.process_transcription, task, gpu_index)
        )

        if diarization and dual_channel is False:
            asyncio.get_event_loop().run_in_executor(
                None, functools.partial(self.process_diarization, task, gpu_index)
            )
        else:
            task["diarization_done"].set()

        await task["transcription_done"].wait()
        await task["diarization_done"].wait()

        if isinstance(task["diarization_result"], Exception):
            self.gpu_handler.release_device(gpu_index)
            return task["diarization_result"]

        if isinstance(task["transcription_result"], Exception):
            self.gpu_handler.release_device(gpu_index)
            return task["transcription_result"]
        else:
            if alignment and dual_channel is False:
                asyncio.get_event_loop().run_in_executor(
                    None, functools.partial(self.process_alignment, task, gpu_index)
                )
            else:
                task["alignment_done"].set()

        await task["alignment_done"].wait()

        if isinstance(task["alignment_result"], Exception):
            logger.error(f"Alignment failed: {task['alignment_result']}")
            # Failed alignment should not fail the whole request anymore, as not critical
            # So we keep processing the request and return the transcription result
            # return task["alignment_result"]

        self.gpu_handler.release_device(gpu_index)  # Release the GPU

        asyncio.get_event_loop().run_in_executor(
            None, functools.partial(self.process_post_processing, task)
        )

        await task["post_processing_done"].wait()

        result = task.pop("post_processing_result")
        del task  # Delete the task to free up memory

        return result, duration

    @time_and_tell
    def process_transcription(self, task: dict, gpu_index: int) -> None:
        """
        Process a task of transcription and update the task with the result.

        Args:
            task (dict): The task and its parameters.
            gpu_index (int): The GPU index to use for the transcription.

        Returns:
            None: The task is updated with the result.
        """
        try:
            segments = self.services["transcription"](
                task["input"],
                source_lang=task["source_lang"],
                model_index=gpu_index,
                suppress_blank=False,
                vocab=None if task["vocab"] == [] else task["vocab"],
                word_timestamps=True,
                vad_service=self.services["vad"] if task["dual_channel"] else None,
                use_batch=task["use_batch"],
            )
            result = segments

        except Exception as e:
            result = Exception(f"Error in transcription gpu {gpu_index}: {e}\n{traceback.format_exc()}")

        finally:
            task["transcription_result"] = result
            task["transcription_done"].set()

        return None

    @time_and_tell
    def process_diarization(self, task: dict, gpu_index: int) -> None:
        """
        Process a task of diarization.

        Args:
            task (dict): The task and its parameters.
            gpu_index (int): The GPU index to use for the diarization.

        Returns:
            None: The task is updated with the result.
        """
        try:
            result = self.services["diarization"](task["input"], model_index=gpu_index)

        except Exception as e:
            result = Exception(f"Error in diarization: {e}\n{traceback.format_exc()}")

        finally:
            task["diarization_result"] = result
            task["diarization_done"].set()

        return None

    @time_and_tell
    def process_alignment(self, task: dict, gpu_index: int) -> None:
        """
        Process a task of alignment.

        Args:
            task (dict): The task and its parameters.
            gpu_index (int): The GPU index to use for the alignment.

        Returns:
            None: The task is updated with the result.
        """
        try:
            segments = self.services["alignment"](
                task["input"],
                transcript_segments=task["transcription_result"],
                source_lang=task["source_lang"],
                gpu_index=gpu_index,
            )

        except Exception as e:
            segments = Exception(f"Error in alignment: {e}\n{traceback.format_exc()}")

        finally:
            task["alignment_result"] = segments
            task["alignment_done"].set()

        return None

    @time_and_tell
    def process_post_processing(self, task: dict) -> None:
        """
        Process a task of post processing.

        Args:
            task (dict): The task and its parameters.

        Returns:
            None: The task is updated with the result.
        """
        try:
            alignment = task["alignment"]
            diarization = task["diarization"]
            dual_channel = task["dual_channel"]
            word_timestamps = task["word_timestamps"]

            if dual_channel:
                left_segments, right_segments = task["transcription_result"]
                utterances = self.services["post_processing"].dual_channel_speaker_mapping(
                    left_segments=left_segments,
                    right_segments=right_segments,
                    word_timestamps=word_timestamps,
                )
            else:
                segments = (
                    task["alignment_result"]
                    if alignment and not isinstance(task["alignment_result"], Exception)
                    else task["transcription_result"]
                )

                formatted_segments = format_segments(
                    segments=segments,
                    alignment=alignment,
                    use_batch=task["use_batch"],
                    word_timestamps=word_timestamps,
                )

                if diarization:
                    utterances = self.services[
                        "post_processing"
                    ].single_channel_speaker_mapping(
                        transcript_segments=formatted_segments,
                        speaker_timestamps=task["diarization_result"],
                        word_timestamps=word_timestamps,
                    )
                else:
                    utterances = formatted_segments

            final_utterances = self.services[
                "post_processing"
            ].final_processing_before_returning(
                utterances=utterances,
                diarization=diarization,
                dual_channel=task["dual_channel"],
                timestamps_format=task["timestamps_format"],
                word_timestamps=word_timestamps,
            )

        except Exception as e:
            final_utterances = Exception(f"Error in post-processing: {e}\n{traceback.format_exc()}")

        finally:
            task["post_processing_result"] = final_utterances
            task["post_processing_done"].set()

        return None


class ASRLiveService:
    """ASR Service module for live endpoints."""

    def __init__(self) -> None:
        """Initialize the ASRLiveService class."""
        super().__init__()
