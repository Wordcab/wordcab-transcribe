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

from typing import Any, Dict, List, Optional

from wordcab_transcribe.utils import (
    _convert_s_to_ms,
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

    def single_channel_postprocessing(
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
        segments_with_speaker_mapping = self.segments_speaker_mapping(
            transcript_segments, speaker_timestamps, word_timestamps
        )

        utterances = self.utterances_speaker_mapping(
            segments_with_speaker_mapping,
            word_timestamps,
            speaker_timestamps,
        )

        return utterances

    def dual_channel_postprocessing(
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
        utterances = self.utterances_speaker_mapping(utterances, word_timestamps)

        return utterances

    def segments_speaker_mapping(
        self,
        transcript_segments: List[dict],
        speaker_timestamps: List[str],
        word_timestamps: bool,
        anchor_option: str = "start",
    ) -> List[dict]:
        """
        Map each transcript segment to its corresponding speaker.

        Args:
            transcript_segments (List[dict]): List of transcript segments.
            speaker_timestamps (List[str]): List of speaker timestamps.
            anchor_option (str): Anchor option to use.
            word_timestamps (bool): Whether to include word timestamps.

        Returns:
            List[dict]: List of transcript segments with speaker mapping.
        """
        _, end, speaker = speaker_timestamps[0]
        segment_position, turn_idx = 0, 0
        segment_speaker_mapping = []

        for idx, segment in enumerate(transcript_segments):
            if "start" not in segment:
                if idx == 0:
                    segment["start"] = 0
                else:
                    segment["start"] = transcript_segments[idx - 1]["end"]
            if "end" not in segment:
                segment["end"] = segment["start"] + 1

            # Convert segment timestamps to milliseconds to match speaker timestamps.
            segment_start, segment_end, segment_text = (
                int(_convert_s_to_ms(segment["start"])),
                int(_convert_s_to_ms(segment["end"])),
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

            _segment = {
                "start": segment_start / 1000,
                "end": segment_end / 1000,
                "text": segment_text,
                "speaker": speaker,
            }
            if word_timestamps:
                _segment["words"] = segment["words"]

            segment_speaker_mapping.append(_segment)

        return segment_speaker_mapping

    def utterances_speaker_mapping(
        self,
        transcript_segments: List[dict],
        word_timestamps: bool,
        speaker_timestamps: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Map utterances of the same speaker together for dual channel use case.

        Args:
            transcript_segments (List[dict]): List of transcript segments.
            word_timestamps (bool): Whether to include word timestamps.
            speaker_timestamps (Optional[List[str]]): List of speaker timestamps.

        Returns:
            List[dict]: List of sentences with speaker mapping.
        """
        if speaker_timestamps:
            start_t0, end_t0, speaker_t0 = speaker_timestamps[0]
        else:
            start_t0, end_t0, speaker_t0 = (
                transcript_segments[0]["start"],
                transcript_segments[0]["end"],
                transcript_segments[0]["speaker"],
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
        for segment in transcript_segments:
            text_segment, speaker = segment["text"], segment["speaker"]
            start_t, end_t = segment["start"], segment["end"]

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

            current_sentence["text"] += text_segment + " "
            previous_speaker = speaker
            if word_timestamps:
                current_sentence["words"].extend(segment["words"])

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
        word_timestamps: bool
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
