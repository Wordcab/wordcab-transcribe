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

import torch

from wordcab_transcribe.config import settings
from wordcab_transcribe.logging import time_and_tell
from wordcab_transcribe.services.align_service import AlignService
from wordcab_transcribe.services.diarize_service import DiarizeService
from wordcab_transcribe.services.post_processing_service import PostProcessingService
from wordcab_transcribe.services.transcribe_service import TranscribeService
from wordcab_transcribe.services.vad_service import VadService
from wordcab_transcribe.utils import format_segments


class ASRService(ABC):
    """Base ASR Service module that handle all AI interactions and batch processing."""

    def __init__(self) -> None:
        """Initialize the ASR Service.

        This class is not meant to be instantiated. Use the subclasses instead.
        """
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Do we have a GPU? If so, use it!
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
    def schedule_processing_if_needed(self) -> None:
        """Method to schedule processing if needed."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    async def process_input(self) -> None:
        """Process the input request by creating a task and adding it to the appropriate queues."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    async def runner(self) -> None:
        """Runner method to process the queue."""
        raise NotImplementedError("This method should be implemented in subclasses.")


class ASRAsyncService(ASRService):
    """ASR Service module for async endpoints."""

    def __init__(self) -> None:
        """Initialize the ASRAsyncService class."""
        super().__init__()

        self.task_threads: dict = {
            "transcription": 1,
            "diarization": 1,
            "alignment": 1,
            "post_processing": 8,
        }
        self.thread_executors: dict = {
            "transcription": ThreadPoolExecutor(
                max_workers=self.task_threads["transcription"]
            ),
            "diarization": ThreadPoolExecutor(
                max_workers=self.task_threads["diarization"]
            ),
            "alignment": ThreadPoolExecutor(max_workers=self.task_threads["alignment"]),
            "post_processing": ThreadPoolExecutor(
                max_workers=self.task_threads["post_processing"]
            ),
        }
        self.services: dict = {
            "transcription": TranscribeService(
                model_path=settings.whisper_model,
                compute_type=settings.compute_type,
                device=self.device,
                num_workers=self.task_threads["transcription"],
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
        self.queues: dict = {
            "transcription": [],
            "diarization": [],
            "alignment": [],
            "post_processing": [],
        }
        self.queue_locks: dict = {
            "transcription": asyncio.Lock(),
            "diarization": asyncio.Lock(),
            "alignment": asyncio.Lock(),
            "post_processing": asyncio.Lock(),
        }
        self.needs_processing: dict = {
            "transcription": asyncio.Event(),
            "diarization": asyncio.Event(),
            "alignment": asyncio.Event(),
            "post_processing": asyncio.Event(),
        }
        self.needs_processing_timer: dict = {
            "transcription": None,
            "diarization": None,
            "alignment": None,
            "post_processing": None,
        }
        self.dual_channel_transcribe_options: dict = {
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
        if (
            len(self.queues[task_type]) >= 1
        ):  # We process the queue as soon as we have one request
            self.needs_processing[task_type].set()
        elif self.queues[task_type]:
            self.needs_processing_timer[task_type] = asyncio.get_event_loop().call_at(
                self.queues[task_type][0]["time"] + settings.max_wait,
                self.needs_processing[task_type].set,
            )

    @time_and_tell
    async def process_input(
        self,
        filepath: Union[str, Tuple[str, str]],
        alignment: bool,
        diarization: bool,
        dual_channel: bool,
        source_lang: str,
        timestamps_format: str,
        word_timestamps: bool,
    ) -> Union[List[dict], Exception]:
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
            word_timestamps (bool): Whether to return word timestamps or not.

        Returns:
            Union[List[dict], Exception]: The final transcription result or an exception.
        """
        task = {
            "input": filepath,  # TODO: Should be the file tensors to be optimized (loaded via torchaudio)
            "alignment": alignment,
            "diarization": diarization,
            "dual_channel": dual_channel,
            "source_lang": source_lang,
            "timestamps_format": timestamps_format,
            "word_timestamps": word_timestamps,
            "post_processed": False,
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

        async with self.queue_locks["transcription"]:
            self.queues["transcription"].append(task)
            self.schedule_processing_if_needed("transcription")

        if diarization and dual_channel is False:
            async with self.queue_locks["diarization"]:
                self.queues["diarization"].append(task)
                self.schedule_processing_if_needed("diarization")
        else:
            task["diarization_done"].set()

        await asyncio.gather(
            task["transcription_done"].wait(),
            task["diarization_done"].wait(),
            return_exceptions=True,
        )

        if isinstance(task["diarization_result"], Exception):
            return task["diarization_result"]

        if isinstance(task["transcription_result"], Exception):
            return task["transcription_result"]
        else:
            if alignment and dual_channel is False:
                async with self.queue_locks["alignment"]:
                    self.queues["alignment"].append(task)
                    self.schedule_processing_if_needed("alignment")
            else:
                task["alignment_done"].set()

        await asyncio.gather(
            task["alignment_done"].wait(),
            return_exceptions=True,
        )

        if isinstance(task["alignment_result"], Exception):
            return task["alignment_result"]
        else:
            self.queues["post_processing"].append(task)
            self.schedule_processing_if_needed("post_processing")

        await task["post_processing_done"].wait()

        result = task.pop("post_processing_result")
        del task  # Delete the task to free up memory

        return result

    async def runner(self, task_type: str) -> None:
        """
        Runner generic method to process the queue for a specific task type.

        Args:
            task_type (str): The task type used by the runner.
        """
        while True:
            await self.needs_processing[task_type].wait()
            self.needs_processing[task_type].clear()

            if self.needs_processing_timer[task_type] is not None:
                self.needs_processing_timer[task_type].cancel()
                self.needs_processing_timer[task_type] = None

            async with self.queue_locks[task_type]:
                task_to_run = self.queues[task_type][0]
                del self.queues[task_type][0]

            self.schedule_processing_if_needed(task_type)

            func = getattr(self, f"process_{task_type}")

            try:
                results = await asyncio.get_event_loop().run_in_executor(
                    self.thread_executors[task_type], func, task_to_run
                )

                task_to_run[f"{task_type}_result"] = results

                del results

            except Exception as e:
                task_to_run[f"{task_type}_result"] = Exception(
                    f"Error in {task_type}: {e}\n{traceback.format_exc()}"
                )

            finally:
                task_to_run[f"{task_type}_done"].set()

    @time_and_tell
    def process_transcription(
        self, task: dict
    ) -> Union[List[dict], Tuple[List[dict], List[dict]]]:
        """
        Process a task of transcription.

        Args:
            task (dict): The task and its parameters.

        Returns:
            List[dict]: List of transcribed segments.

        Raises:
            ValueError: If the task is not a dual channel task and the input is not a string.
        """
        segments = self.services["transcription"](
            task["input"],
            source_lang=task["source_lang"],
            word_timestamps=True,
        )

        return segments

    @time_and_tell
    def process_diarization(self, task: dict) -> List[dict]:
        """
        Process a task of diarization.

        Args:
            task (dict): The task and its parameters.

        Returns:
            List[dict]: List of speaker turns.
        """
        utterances = self.services["diarization"](task["input"])

        return utterances

    @time_and_tell
    def process_alignment(self, task: dict) -> List[dict]:
        """
        Process a task of alignment.

        Args:
            task (dict): The task and its parameters.

        Returns:
            List[dict]: List of aligned segments.
        """
        segments = self.services["alignment"](
            task["input"],
            transcript_segments=task["transcription_result"],
            source_lang=task["source_lang"],
        )

        return segments

    @time_and_tell
    def process_post_processing(self, task: dict) -> List[dict]:
        """
        Process a task of post processing.

        Args:
            task (dict): The task and its parameters.

        Returns:
            List[dict]: List of post processed segments.
        """
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
                task["alignment_result"] if alignment else task["transcription_result"]
            )

            formatted_segments = format_segments(
                segments=segments,
                alignment=alignment,
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

        return final_utterances


class ASRLiveService:
    """ASR Service module for live endpoints."""

    def __init__(self) -> None:
        """Initialize the ASRLiveService class."""
        super().__init__()
