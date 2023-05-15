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
from typing import List

import torch
from loguru import logger

from wordcab_transcribe.config import settings
from wordcab_transcribe.services.align_service import AlignService
from wordcab_transcribe.services.diarize_service import DiarizeService
from wordcab_transcribe.services.post_processing_service import PostProcessingService
from wordcab_transcribe.services.transcribe_service import TranscribeService
from wordcab_transcribe.utils import format_segments


class ASRService:
    """Base ASR Service module that handle all AI interactions and batch processing."""

    def __init__(self) -> None:
        """Initialize the ASR Service.

        This class is not meant to be instantiated.
        Use the subclasses instead: ASRAsyncService or ASRLiveService.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.thread_executor = ThreadPoolExecutor(max_workers=4)

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
        filepath: str,
        source_lang: str,
        alignment: bool,
    ) -> List[dict]:
        """
        Process the input request and return the result.

        Args:
            filepath (str): Path to the audio file.
            source_lang (str): Source language of the audio file.
            alignment (bool): Whether to do alignment or not.

        Returns:
            List[dict]: List of speaker segments.
        """
        task = {
            "input": filepath,
            "source_lang": source_lang,
            "alignment": alignment,
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

        self.transcribe_model = TranscribeService(
            model_path=settings.whisper_model,
            compute_type=settings.compute_type,
            device=self.device,
        )
        self.align_model = AlignService(self.device)
        self.diarize_model = DiarizeService(
            domain_type=settings.nemo_domain_type,
            storage_path=settings.nemo_storage_path,
            output_path=settings.nemo_output_path,
            device=self.device,
        )
        self.post_processing_model = PostProcessingService()

    def transcribe(self, filepath: str, source_lang: str) -> List[dict]:
        """
        Transcribe the audio file using the TranscribeService class.

        Args:
            filepath (str): Path to the audio file.
            source_lang (str): Source language of the audio file.

        Returns:
            List[dict]: List of speaker segments.
        """
        segments = self.transcribe_model(filepath, source_lang)

        return segments

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

    def post_process(
        self, segments: List[dict], speaker_timestamps: List[dict]
    ) -> List[dict]:
        """
        Post process the segments using the PostProcessingService class.

        Args:
            segments (List[dict]): List of speaker segments.
            speaker_timestamps (List[dict]): List of speaker timestamps.

        Returns:
            List[dict]: List of speaker segments.
        """
        utterances = self.post_processing_model(segments, speaker_timestamps)

        return utterances

    def process_batch(self, file_batch: List[dict]) -> List[dict]:
        """
        Process a batch of requests.

        Args:
            file_batch (List[dict]): List of requests to process with their respective parameters.

        Returns:
            List[dict]: List of results.
        """
        results = []
        for task in file_batch:
            filepath = task["input"]
            source_lang = task["source_lang"]
            alignment = task["alignment"]

            segments = self.transcribe(filepath, source_lang)

            if alignment:
                formatted_segments = self.align(filepath, segments, source_lang)
            else:
                formatted_segments = format_segments(segments)

            speaker_timestamps = self.diarize(filepath)

            utterances = self.post_process(formatted_segments, speaker_timestamps)

            results.append(utterances)

        return results


class ASRLiveService(ASRService):
    """ASR Service module for live endpoints."""

    def __init__(self) -> None:
        """Initialize the ASRLiveService class."""
        super().__init__()
