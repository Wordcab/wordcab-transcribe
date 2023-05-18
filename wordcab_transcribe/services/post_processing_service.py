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

from wordcab_transcribe.utils import get_segment_timestamp_anchor


class PostProcessingService:
    """Post-Processing Service for audio files."""

    def __init__(self) -> None:
        """Initialize the PostProcessingService."""
        pass

    def single_channel_postprocessing(
        self, transcript_segments: List[dict], speaker_timestamps: List[dict]
    ) -> List[dict]:
        """Run the post-processing functions on the inputs.

        The postprocessing pipeline is as follows:
        1. Map each transcript segment to its corresponding speaker.
        2. Group utterances of the same speaker together.

        Args:
            transcript_segments (List[dict]): List of transcript segments.
            speaker_timestamps (List[dict]): List of speaker timestamps.

        Returns:
            List[dict]: List of sentences with speaker mapping.
        """
        segments_with_speaker_mapping = self.segments_speaker_mapping(
            transcript_segments, speaker_timestamps
        )

        utterances = self.utterances_speaker_mapping(
            segments_with_speaker_mapping, speaker_timestamps
        )

        return utterances

    def dual_channel_postprocessing(
        self, left_segments: List[dict], right_segments: List[dict]
    ) -> List[dict]:
        """Run the dual channel post-processing functions on the inputs.

        The postprocessing pipeline is as follows:
        1. Reconstruct proper timestamps for each segment.
        2. Merge the left and right segments together and sort them by timestamp.

        Args:
            left_segments (List[dict]): List of left channel segments.
            right_segments (List[dict]): List of right channel segments.

        Returns:
            List[dict]: List of sentences with speaker mapping.
        """
        _left_segments = self.reconstruct_segments(left_segments, speaker_label=0)
        _right_segments = self.reconstruct_segments(right_segments, speaker_label=1)

        utterances = self.merge_segments(_left_segments, _right_segments)

        utterances = self.utterances_speaker_mapping_dual_channel(utterances)

        return utterances

    def segments_speaker_mapping(
        self,
        transcript_segments: List[dict],
        speaker_timestamps: List[str],
        anchor_option: str = "start",
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

        for idx, segment in enumerate(transcript_segments):
            if "start" not in segment:
                if idx == 0:
                    segment["start"] = 0
                else:
                    segment["start"] = transcript_segments[idx - 1]["end"]
            if "end" not in segment:
                segment["end"] = segment["start"] + 1

            segment_start, segment_end, segment_text = (
                int(segment["start"] * 1000),
                int(segment["end"] * 1000),
                segment["word"],
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
            "speaker": speaker_t0,
            "start": start_t0,
            "end": end_t0,
            "text": "",
        }

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
            else:
                current_sentence["end"] = end_t

            current_sentence["text"] += text_segment + " "
            previous_speaker = speaker

        sentences.append(current_sentence)

        return sentences

    def utterances_speaker_mapping_dual_channel(
        self, transcript_segments: List[dict]
    ) -> List[dict]:
        """
        Map utterances of the same speaker together for dual channel use case.

        Args:
            transcript_segments (List[dict]): List of transcript segments.

        Returns:
            List[dict]: List of sentences with speaker mapping.
        """
        current_sentence = {
            "speaker": transcript_segments[0]["speaker"],
            "start": transcript_segments[0]["start"],
            "end": transcript_segments[0]["end"],
            "text": "",
        }
        previous_speaker = transcript_segments[0]["speaker"]

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
            else:
                current_sentence["end"] = end_t

            current_sentence["text"] += text_segment + " "
            previous_speaker = speaker

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

    def reconstruct_segments(
        self,
        segment_list: list,
        speaker_label: int,
        time_threshold: Optional[float] = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct segments based on the words timestamps.

        Args:
            segment_list (list): List of the transcript segments.
            speaker_label (int): The speaker label.
            time_threshold (float, optional): The time threshold in seconds. Defaults to 1.0.

        Returns:
            List[Dict[str, Any]]: List of reconstructed segments.
        """
        reconstructed_segments = []

        for segment in segment_list:
            words = segment["words"]
            if not words:
                continue

            new_segment = {
                "start": words[0].start,
                "end": words[0].end,
                "text": words[0].word.strip(),
                "speaker": speaker_label,
            }

            for i in range(1, len(words)):
                if words[i].start - new_segment["end"] > time_threshold:
                    reconstructed_segments.append(new_segment)
                    new_segment = {
                        "start": words[i].start,
                        "end": words[i].end,
                        "text": words[i].word.strip(),
                        "speaker": speaker_label,
                    }
                else:
                    new_segment["end"] = words[i].end
                    new_segment["text"] += " " + words[i].word.strip()

            reconstructed_segments.append(new_segment)

        for segment in reconstructed_segments:
            segment["start"] = round(segment["start"] * 1000, 3)
            segment["end"] = round(segment["end"] * 1000, 3)

        return reconstructed_segments
