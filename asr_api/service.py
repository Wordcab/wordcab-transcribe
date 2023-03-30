# Copyright (c) 2023, The Wordcab team. All rights reserved.
"""Service module to handle AI model interactions."""

import asyncio
import functools
import io
import numpy as np
from loguru import logger
from typing import List

import torch

from faster_whisper import WhisperModel

from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

from asr_api.utils import format_segments


class ASRService():
    def __init__(
        self,
        model_size: str = "large-v2",
        embds_model: str = "speechbrain/spkrec-ecapa-voxceleb",
    ) -> None:
        """
        ASR Service class to handle AI model interactions.

        Args:
            model_size (str, optional): Model size to use. Defaults to "large-v2".
            embds_model (str, optional): Speaker embeddings model to use. 
            Defaults to "speechbrain/spkrec-ecapa-voxceleb".
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_size = model_size
        self.embds_model = embds_model

        self.model = WhisperModel(
            self.model_size, 
            device=self.device, 
            compute_type="int8_float16"
        )
        self.embedding_model = PretrainedSpeakerEmbedding( 
            self.embds_model,
            device=self.device
        )

        # Multi requests support
        self.queue = []
        self.queue_lock = None
        self.needs_processing = None
        self.needs_processing_timer = None

        self.max_batch_size = 1  # Max number of requests to process at once
        self.max_wait = 0.1


    def schedule_processing_if_needed(self):
        """Method to schedule processing if needed."""
        if len(self.queue) >= self.max_batch_size:
            self.needs_processing.set()
        elif self.queue:
            self.needs_processing_timer = asyncio.get_event_loop().call_at(
                self.queue[0]["time"] + self.max_wait, self.needs_processing.set
            )


    async def process_input(self, filepath: str, num_speakers: int) -> None:
        """
        Process the input request and return the result.

        Args:
            filepath (str): Path to the audio file.
            num_speakers (int): Number of speakers to detect.
        """
        our_task = {
            "done_event": asyncio.Event(),
            "input": filepath,
            "num_speakers": num_speakers,
            "time": asyncio.get_event_loop().time(),
        }
        async with self.queue_lock:
            self.queue.append(our_task)
            self.schedule_processing_if_needed()

        await our_task["done_event"].wait()

        return our_task["result"]


    async def runner(self):
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
                    longest_wait = asyncio.get_event_loop().time() - self.queue[0]["time"]
                else:
                    longest_wait = None
                file_batch = self.queue[:self.max_batch_size]
                del self.queue[:len(file_batch)]
                self.schedule_processing_if_needed()

            try:
                batch = [(task["input"], task["num_speakers"]) for task in file_batch]
                results = []
                for input_file, num_speakers in batch:
                    res = await asyncio.get_event_loop().run_in_executor(
                        None, functools.partial(self.inference, input_file, num_speakers)
                    )
                    results.append(res)
                for task, result in zip(file_batch, results):
                    task["result"] = result
                    task["done_event"].set()
                del file_batch
                del results

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                for task in file_batch:  # Error handling
                    task["result"] = e
                    task["done_event"].set()


    def inference(self, filepath: str, num_speakers: int) -> List[dict]:
        """
        Inference method to process the audio file.

        Args:
            filepath (str): Path to the audio file.
            num_speakers (int): Number of speakers to detect.

        Returns:
            List[dict]: List of diarized segments.
        """
        segments, _ = self.model.transcribe(filepath, language="en", beam_size=5, word_timestamps=True)
        segments = format_segments(list(segments))

        duration = segments[-1]["end"]

        diarized_segments = self.diarize(filepath, segments, duration, num_speakers)

        return diarized_segments


    def diarize(
        self, audio_obj: io.BytesIO, segments: List[dict], duration: float, num_speakers: int = None
    ) -> List[dict]:
        """
        Diarize the segments using pyannote.

        Args:
            audio_obj (io.BytesIO): Audio file object.
            segments (List[dict]): List of segments to diarize.
            duration (float): Duration of the audio file.
            num_speakers (int, optional): Number of speakers. Defaults to None.

        Returns:
            List[dict]: List of diarized segments with speaker labels.
        """
        embeddings = np.zeros(shape=(len(segments), 192))

        for i, segment in enumerate(segments):
            embeddings[i] = self.segment_embedding(audio_obj, segment, duration)

        embeddings = np.nan_to_num(embeddings)

        num_speakers = num_speakers or 0
        best_num_speakers = self._get_num_speakers(embeddings, num_speakers)

        identified_segments = self._assign_speaker_label(segments, embeddings, best_num_speakers)

        joined_segments = self.join_utterances(identified_segments)

        return joined_segments


    def segment_embedding(self, audio_obj: io.BytesIO, segment: dict, duration: float) -> np.ndarray:
        """
        Get the embedding of a segment.

        Args:
            audio_obj (io.BytesIO): Audio file object.
            segment (dict): Segment to get the embedding.
            duration (float): Duration of the audio file.

        Returns:
            np.ndarray: Embedding of the segment.
        """
        start = segment["start"]
        end = min(duration, segment["end"])

        clip = Segment(start=start, end=end)

        audio = Audio()
        waveform, _ = audio.crop(audio_obj, clip)
        
        return self.embedding_model(waveform[None])


    def join_utterances(self, segments: List[dict]) -> List[dict]:
        """
        Join the segments of the same speaker.

        Args:
            segments (List[dict]): List of segments.

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
                    "speaker": segment["speaker"]
                }
            
            text += segment["text"] + " "

        if current_utterance:
            current_utterance["end"] = segments[idx]["end"]
            current_utterance["text"] = text.strip()
            utterance_list.append(current_utterance)

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

            for i in range(2, 11):
                clustering = AgglomerativeClustering(i).fit(embeddings)
                score = silhouette_score(embeddings, clustering.labels_, metric="euclidean")
                score_num_speakers[i] = score

            best_num_speakers = max(score_num_speakers, key=lambda x: score_num_speakers[x])

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
        clustering = AgglomerativeClustering(best_num_speakers).fit(embeddings)
        labels = clustering.labels_

        for i in range(len(segments)):
            segments[i]["speaker"] = labels[i] + 1
 
        return segments
