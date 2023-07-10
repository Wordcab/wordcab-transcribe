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
"""Post-Processing Service for audio files."""

from typing import Any, Dict, List
from loguru import logger

from wordcab_transcribe.utils import (
    _convert_s_to_ms,
    _convert_ms_to_s,
    convert_timestamp,
    format_punct,
    get_segment_timestamp_anchor,
    is_empty_string,
)


class PostProcessingService:
    """Post-Processing Service for audio files."""

    def __init__(self) -> None:
        """Initialize the PostProcessingService."""
        self.sample_rate = 16000

    def single_channel_speaker_mapping(
        self,
        transcript_segments: List[dict],
        speaker_timestamps: List[dict],
        word_timestamps: bool,
    ) -> List[dict]:
        """Run the post-processing functions on the inputs.

        The postprocessing pipeline is as follows:
        1. Map each transcript segment to its corresponding speaker.
        2. Group utterances of the same speaker together.

        Args:
            transcript_segments (List[dict]): List of transcript segments.
            speaker_timestamps (List[dict]): List of speaker timestamps.
            word_timestamps (bool): Whether to include word timestamps.

        Returns:
            List[dict]: List of sentences with speaker mapping.
        """
        words_with_speaker_mapping = self.words_speaker_mapping(
            transcript_segments, speaker_timestamps,
        )

        utterances = self.reconstruct_utterances(
            words_with_speaker_mapping, word_timestamps
        )

        return utterances

    def dual_channel_speaker_mapping(
        self,
        left_segments: List[dict],
        right_segments: List[dict],
        word_timestamps: bool,
    ) -> List[dict]:
        """Run the dual channel post-processing functions on the inputs.

        The postprocessing pipeline is as follows:
        1. Reconstruct proper timestamps for each segment.
        2. Merge the left and right segments together and sort them by timestamp.

        Args:
            left_segments (List[dict]): List of left channel segments.
            right_segments (List[dict]): List of right channel segments.
            word_timestamps (bool): Whether to include word timestamps.

        Returns:
            List[dict]: List of sentences with speaker mapping.
        """
        utterances = self.merge_segments(left_segments, right_segments)
        utterances = self.reconstruct_utterances(utterances, word_timestamps)

        return utterances

    def words_speaker_mapping(
        self,
        transcript_segments: List[dict],
        speaker_timestamps: List[str],
    ) -> List[dict]:
        """
        Map each word to its corresponding speaker based on the speaker timestamps.

        Args:
            transcript_segments (List[dict]): List of transcript segments.
            speaker_timestamps (List[str]): List of speaker timestamps.

        Returns:
            List[dict]: List of words with speaker mapping.
        """
        _, end, speaker = speaker_timestamps[0]
        turn_idx = 0

        all_words = []
        for segment in transcript_segments:
            all_words.extend(segment["words"])

        mapped_words = []
        for word in all_words:
            word_start, word_end = _convert_s_to_ms(word["start"]), _convert_s_to_ms(word["end"])
            while word_start > float(end):
                turn_idx += 1
                turn_idx = min(turn_idx, len(speaker_timestamps) - 1)
                _, end, speaker = speaker_timestamps[turn_idx]
                if turn_idx == len(speaker_timestamps) - 1:
                    end = get_segment_timestamp_anchor(
                        word_start, word_end, option="end"
                    )
                    break

            mapped_words.append(
                {
                    "start": _convert_ms_to_s(word_start),
                    "end": _convert_ms_to_s(word_end),
                    "text": word["word"],
                    "speaker": speaker,
                }
            )

        return mapped_words

    def reconstruct_utterances(
        self,
        transcript_words: List[dict],
        word_timestamps: bool,
    ) -> List[dict]:
        """
        Reconstruct the utterances based on the speaker mapping.

        Args:
            transcript_words (List[dict]): List of transcript words.
            word_timestamps (bool): Whether to include word timestamps.

        Returns:
            List[dict]: List of sentences with speaker mapping.
        """
        start_t0, end_t0, speaker_t0 = (
            transcript_words[0]["start"],
            transcript_words[0]["end"],
            transcript_words[0]["speaker"],
        )

        previous_speaker = speaker_t0
        current_sentence = {
            "speaker": speaker_t0,
            "start": start_t0,
            "end": end_t0,
            "text": "",
        }
        if word_timestamps:
            current_sentence["words"] = []

        sentences = []
        for word in transcript_words:
            text, speaker = word["text"], word["speaker"]
            start_t, end_t = word["start"], word["end"]

            if speaker != previous_speaker:
                sentences.append(current_sentence)
                current_sentence = {
                    "speaker": speaker,
                    "start": start_t,
                    "end": end_t,
                    "text": "",
                }
                if word_timestamps:
                    current_sentence["words"] = []
            else:
                current_sentence["end"] = end_t

            current_sentence["text"] += text + " "
            previous_speaker = speaker
            if word_timestamps:
                current_sentence["words"].append(
                    dict(
                        word=text,
                        start=start_t,
                        end=end_t,
                    )
                )

        # Catch the last sentence
        sentences.append(current_sentence)

        return sentences

    def merge_segments(
        self,
        speaker_0_segments: List[Dict[str, Any]],
        speaker_1_segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge two lists of segments, keeping the chronological order.

        Args:
            speaker_0_segments (List[Dict[str, Any]]): List of segments from speaker 0.
            speaker_1_segments (List[Dict[str, Any]]): List of segments from speaker 1.

        Returns:
            List[Dict[str, Any]]: Merged list of segments.
        """
        merged_segments = speaker_0_segments + speaker_1_segments
        merged_segments.sort(key=lambda seg: seg["start"])

        return merged_segments

    def final_processing_before_returning(
        self,
        utterances: List[dict],
        diarization: bool,
        dual_channel: bool,
        timestamps_format: str,
        word_timestamps: bool,
    ) -> List[dict]:
        """
        Do final processing before returning the utterances to the API.

        Args:
            utterances (List[dict]): List of utterances.
            diarization (bool): Whether diarization is enabled.
            dual_channel (bool): Whether dual channel is enabled.
            timestamps_format (str): Timestamps format used for conversion.
            word_timestamps (bool): Whether to include word timestamps.

        Returns:
            List[dict]: List of utterances with final processing.
        """
        include_speaker = diarization or dual_channel
        _utterances = [
            {
                "text": format_punct(utterance["text"]),
                "start": convert_timestamp(utterance["start"], timestamps_format),
                "end": convert_timestamp(utterance["end"], timestamps_format),
                "speaker": int(utterance["speaker"]) if include_speaker else None,
                "words": utterance["words"] if word_timestamps else [],
            }
            for utterance in utterances
            if not is_empty_string(utterance["text"])
        ]

        return _utterances
