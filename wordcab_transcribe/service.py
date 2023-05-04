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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import librosa
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from loguru import logger
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from wordcab_transcribe.config import settings
from wordcab_transcribe.utils import (
    format_segments,
    get_segment_timestamp_anchor,
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

        self.nemo_tmp = Path.cwd() / "temp_outputs"
        if not self.nemo_tmp.exists():
            self.nemo_tmp.mkdir(parents=True, exist_ok=True)

        self.model_whisper = WhisperModel(
            self.whisper_model, device=self.device, compute_type=self.compute_type
        )
        self.model_msdd = NeuralDiarizer(
            cfg=load_nemo_config(
                domain_type=settings.nemo_domain_type,
                storage_path=settings.nemo_storage_path,
                output_path=settings.nemo_output_path,
            )
        ).to(self.device)

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
        task = {
            "input": filepath,
            "num_speakers": num_speakers,
            "source_lang": source_lang,
            "timestamps": timestamps,
            "done_event": asyncio.Event(),
            "time": asyncio.get_event_loop().time(),
        }

        async with self.queue_lock:
            self.queue.append(task)
            self.schedule_processing_if_needed()

        await task["done_event"].wait()

        return task["result"]

    def inference_with_whisper(self, filepath: str, source_lang: str) -> List[dict]:
        """Run inference with whisper model."""
        segments, _ = self.model_whisper.transcribe(
            filepath, language=source_lang, beam_size=5, word_timestamps=True
        )
        segments = format_segments(list(segments))

        return segments

    def inference_with_msdd(self, filepath: str) -> List[dict]:
        """Run inference with msdd model."""
        signal, sample_rate = librosa.load(filepath, sr=None)

        tmp_save_path = self.nemo_tmp / "mono_file.wav"

        sf.write(str(tmp_save_path), signal, sample_rate, "PCM_24")

        self.model_msdd.diarize()

        speaker_ts = []
        with open(f"{settings.nemo_output_path}/pred_rttms/mono_file.rttm", "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        return speaker_ts

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

                for task, result in zip(file_batch, results):
                    task["result"] = result
                    task["done_event"].set()

                del results
                del file_batch

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                for task in file_batch:  # Error handling
                    task["result"] = e
                    task["done_event"].set()

    def process_batch(self, file_batch: List[dict]) -> List[dict]:
        """
        Process a batch of requests.

        Args:
            file_batch (List[dict]): List of requests.

        Returns:
            List[dict]: List of results.
        """
        results = []
        for task in file_batch:
            filepath = task["input"]
            source_lang = task["source_lang"]
            # num_speakers = task["num_speakers"]
            # timestamps = task["timestamps"]

            formatted_segments = self.inference_with_whisper(filepath, source_lang)
            speaker_timestamps = self.inference_with_msdd(filepath)

            segments_with_speaker_mapping = self.segments_speaker_mapping(
                formatted_segments, speaker_timestamps
            )
            logger.debug(f"\n\nsegments_with_speaker_mapping: {segments_with_speaker_mapping}")
            utterances = self.utterances_speaker_mapping(
                segments_with_speaker_mapping, speaker_timestamps
            )
            logger.debug(f"\n\nutterances: {utterances}\n\n")
            
            results.append(utterances)

        return results

    def segments_speaker_mapping(
        self, transcript_segments: List[dict], speaker_timestamps: List[str], anchor_option: str = "start"
    ) -> List[dict]:
        """
        Map each transcript segment to its corresponding speaker.

        Args:
            transcript_segments (List[dict]): List of transcript segments.
            speaker_timestamps (List[str]): List of speaker timestamps.
            anchor_option (str): Anchor option to use.

        Returns:
            List[dict]: List of transcript segments with speaker mapping.
        """
        _, end, speaker = speaker_timestamps[0]
        segment_position, turn_idx = 0, 0
        segment_speaker_mapping = []

        for segment in transcript_segments:
            segment_start, segment_end, segment_text = (
                int(segment["start"] * 1000),
                int(segment["end"] * 1000),
                segment["text"],
            )

            segment_position = get_segment_timestamp_anchor(
                segment_start, segment_end, anchor_option
            )

            while segment_position > float(end):
                turn_idx += 1
                turn_idx = min(turn_idx, len(speaker_timestamps) - 1)
                _, end, speaker = speaker_timestamps[turn_idx]
                if turn_idx == len(speaker_timestamps) - 1:
                    end = get_segment_timestamp_anchor(
                        segment_start, segment_end, option="end"
                    )
                    break

            segment_speaker_mapping.append(
                {
                    "start": segment_start,
                    "end": segment_end,
                    "text": segment_text,
                    "speaker": speaker,
                }
            )

        return segment_speaker_mapping

    def utterances_speaker_mapping(
        self, transcript_segments: List[dict], speaker_timestamps: List[dict]
    ) -> List[dict]:
        """
        Map utterances of the same speaker together.

        Args:
            transcript_segments (List[dict]): List of transcript segments.
            speaker_timestamps (List[dict]): List of speaker timestamps.

        Returns:
            List[dict]: List of sentences with speaker mapping.
        """
        start_t0, end_t0, speaker_t0 = speaker_timestamps[0]
        previous_speaker = speaker_t0

        sentences = []
        current_sentence = {
            "speaker": speaker_t0, "start": start_t0, "end": end_t0, "text": ""
        }

        for segment in transcript_segments:
            text_segment, speaker = segment["text"], segment["speaker"]
            start_t, end_t = segment["start"], segment["end"]

            if speaker != previous_speaker:
                sentences.append(current_sentence)
                current_sentence = {
                    "speaker": speaker, "start": start_t, "end": end_t, "text": "",
                }
            else:
                current_sentence["end"] = end_t

            current_sentence["text"] += text_segment + " "
            previous_speaker = speaker

        sentences.append(current_sentence)

        return sentences
