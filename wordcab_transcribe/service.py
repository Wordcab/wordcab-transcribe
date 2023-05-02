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
"""Service module to handle AI model interactions."""

import asyncio
import functools
from typing import List

import numpy as np
import torch
from faster_whisper import WhisperModel
from loguru import logger
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from wordcab_transcribe.config import settings
from wordcab_transcribe.utils import (
    convert_seconds_to_hms,
    format_segments,
    load_nemo_config,
)


class ASRService:
    """ASR Service class to handle AI model interactions."""

    def __init__(self) -> None:
        """Initialize the ASRService class."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = settings.whisper_model
        self.compute_type = settings.compute_type
        self.embeddings_model = settings.embeddings_model

        self.model = WhisperModel(
            self.whisper_model, device=self.device, compute_type=self.compute_type
        )
        self.embedding_model = PretrainedSpeakerEmbedding(
            self.embeddings_model, device=self.device
        )
        self.msdd_model = NeuralDiarizer(
            cfg=load_nemo_config(settings.nemo_domain_type)
        ).to(self.device)

        # Multi requests support
        self.queue = []
        self.queue_lock = None
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
        self, filepath: str, num_speakers: int, source_lang: str, timestamps: str
    ) -> List[dict]:
        """
        Process the input request and return the result.

        Args:
            filepath (str): Path to the audio file.
            num_speakers (int): Number of speakers to detect.
            source_lang (str): Source language of the audio file.
            timestamps (str): Timestamps unit to use.

        Returns:
            List[dict]: List of speaker segments.
        """
        one_task = {
            "done_event": asyncio.Event(),
            "input": filepath,
            "num_speakers": num_speakers,
            "source_lang": source_lang,
            "timestamps": timestamps,
            "time": asyncio.get_event_loop().time(),
        }
        async with self.queue_lock:
            self.queue.append(one_task)
            self.schedule_processing_if_needed()

        await one_task["done_event"].wait()

        return one_task["result"]

    async def runner(self) -> None:
        """Process the input requests in the queue."""
        self.queue_lock = asyncio.Lock()
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
                results = []
                for task in file_batch:
                    res = await asyncio.get_event_loop().run_in_executor(
                        None,
                        functools.partial(
                            self.inference,
                            task["input"],
                            task["num_speakers"],
                            task["source_lang"],
                            task["timestamps"],
                        ),
                    )
                    results.append(res)
                for task, result in zip(file_batch, results):  # noqa: B905
                    task["result"] = result
                    task["done_event"].set()
                del file_batch
                del results

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                for task in file_batch:  # Error handling
                    task["result"] = e
                    task["done_event"].set()

    def inference(
        self,
        filepath: str,
        num_speakers: int,
        source_lang: str,
        timestamps: str,
    ) -> List[dict]:
        """
        Inference method to process the audio file.

        Args:
            filepath (str): Path to the audio file.
            num_speakers (int): Number of speakers to detect.
            source_lang (str): Source language of the audio file.
            timestamps (str): Timestamps unit to use.

        Returns:
            List[dict]: List of diarized segments.
        """
        segments, _ = self.model.transcribe(
            filepath, language=source_lang, beam_size=5, word_timestamps=True
        )
        segments = format_segments(list(segments))

        duration = segments[-1]["end"]
        diarized_segments = self.diarize(
            filepath, segments, duration, num_speakers, timestamps
        )

        return diarized_segments

    def diarize_nemo(self, filepath: str, segments: List[dict], duration: float):
        """Diarize the segments using nemo."""
        self.msdd_model.diarize()

    def diarize(
        self,
        filepath: str,
        segments: List[dict],
        duration: float,
        num_speakers: int,
        timestamps: str,
    ) -> List[dict]:
        """
        Diarize the segments using pyannote.

        Args:
            filepath (str): Path to the audio file.
            segments (List[dict]): List of segments to diarize.
            duration (float): Duration of the audio file.
            num_speakers (int): Number of speakers; defaults to 0.
            timestamps (str): Format of timestamps; defaults to "seconds".

        Returns:
            List[dict]: List of diarized segments with speaker labels.
        """
        embeddings = np.zeros(shape=(len(segments), 192))

        for i, segment in enumerate(segments):
            embeddings[i] = self.segment_embedding(filepath, segment, duration)

        embeddings = np.nan_to_num(embeddings)

        num_speakers = num_speakers or 0
        best_num_speakers = self._get_num_speakers(embeddings, num_speakers)

        identified_segments = self._assign_speaker_label(
            segments, embeddings, best_num_speakers
        )
        joined_segments = self.join_utterances(identified_segments, timestamps)

        return joined_segments

    def segment_embedding(
        self, filepath: str, segment: dict, duration: float
    ) -> np.ndarray:
        """
        Get the embedding of a segment.

        Args:
            filepath (str): Path to the audio file.
            segment (dict): Segment to get the embedding.
            duration (float): Duration of the audio file.

        Returns:
            np.ndarray: Embedding of the segment.
        """
        start = segment["start"]
        end = min(duration, segment["end"])

        clip = Segment(start=start, end=end)

        audio = Audio()
        waveform, _ = audio.crop(filepath, clip)

        return self.embedding_model(waveform[None])

    def join_utterances(self, segments: List[dict], timestamps: str) -> List[dict]:
        """
        Join the segments of the same speaker.

        Args:
            segments (List[dict]): List of segments.
            timestamps (str): Format of timestamps to use.

        Returns:
            List[dict]: List of joined segments with speaker labels.
        """
        utterance_list = []
        current_utterance = None
        text = ""

        for idx, segment in enumerate(segments):
            if idx == 0 or segments[idx - 1]["speaker"] != segment["speaker"]:
                if current_utterance is not None:
                    current_utterance["end"] = segments[idx - 1]["end"]
                    current_utterance["text"] = text.strip()
                    utterance_list.append(current_utterance)
                    text = ""

                current_utterance = {
                    "start": segment["start"],
                    "speaker": segment["speaker"],
                }

            text += segment["text"] + " "

        if current_utterance:
            current_utterance["end"] = segments[idx]["end"]
            current_utterance["text"] = text.strip()
            utterance_list.append(current_utterance)

        for utterance in utterance_list:
            if timestamps == "hms":
                utterance["start"] = convert_seconds_to_hms(utterance["start"])
                utterance["end"] = convert_seconds_to_hms(utterance["end"])
            elif timestamps == "seconds":
                utterance["start"] = float(utterance["start"])
                utterance["end"] = float(utterance["end"])
            elif timestamps == "milliseconds":
                utterance["start"] = float(utterance["start"] * 1000)
                utterance["end"] = float(utterance["end"] * 1000)

        return utterance_list

    def _get_num_speakers(self, embeddings: np.ndarray, num_speakers: int) -> int:
        """
        Get the number of speakers in the audio file.

        Args:
            embeddings (np.ndarray): Embeddings of the segments.
            num_speakers (int): Number of speakers.

        Returns:
            int: Number of speakers.
        """
        if num_speakers == 0:
            score_num_speakers = {}
            try:
                for i in range(2, 11):
                    clustering = AgglomerativeClustering(i).fit(embeddings)
                    score = silhouette_score(
                        embeddings, clustering.labels_, metric="euclidean"
                    )
                    score_num_speakers[i] = score

                best_num_speakers = max(
                    score_num_speakers, key=lambda x: score_num_speakers[x]
                )
            except Exception as e:
                logger.warning(
                    f"Error while getting number of speakers: {e}, defaulting to 1"
                )
                best_num_speakers = 1
        else:
            best_num_speakers = num_speakers

        return best_num_speakers

    def _assign_speaker_label(
        self, segments: List[dict], embeddings: np.ndarray, best_num_speakers: int
    ) -> List[int]:
        """
        Assign a speaker label to each segment.

        Args:
            segments (List[dict]): List of segments.
            embeddings (np.ndarray): Embeddings of the segments.
            best_num_speakers (int): Number of speakers.

        Returns:
            List[int]: List of segments with speaker labels.
        """
        if best_num_speakers == 1:
            for i in range(len(segments)):
                segments[i]["speaker"] = 1
        else:
            clustering = AgglomerativeClustering(best_num_speakers).fit(embeddings)
            labels = clustering.labels_
            for i in range(len(segments)):
                segments[i]["speaker"] = labels[i] + 1

        return segments
