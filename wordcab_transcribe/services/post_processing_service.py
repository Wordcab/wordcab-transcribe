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

from wordcab_transcribe.utils import (
    _convert_ms_to_s,
    _convert_s_to_ms,
    convert_timestamp,
    format_punct,
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
        segments_with_speaker_mapping = self.segments_speaker_mapping(
            transcript_segments,
            speaker_timestamps,
        )

        utterances = self.reconstruct_utterances(
            segments_with_speaker_mapping, word_timestamps
        )

        return utterances

    def dual_channel_speaker_mapping(
        self,
        left_segments: List[dict],
        right_segments: List[dict],
    ) -> List[dict]:
        """
        Run the dual channel post-processing functions on the inputs by merging the segments based on the timestamps.

        Args:
            left_segments (List[dict]): List of left channel segments.
            right_segments (List[dict]): List of right channel segments.

        Returns:
            List[dict]: List of sentences with speaker mapping.
        """
        utterances = self.merge_segments(left_segments, right_segments)

        return utterances

    def segments_speaker_mapping(
        self,
        transcript_segments: List[dict],
        speaker_timestamps: List[dict],
    ) -> List[dict]:
        """Function to map transcription and diarization results.

        Map each segment to its corresponding speaker based on the speaker timestamps and reconstruct the utterances
        when the speaker changes in the middle of a segment.

        Args:
            transcript_segments (List[dict]): List of transcript segments.
            speaker_timestamps (List[dict]): List of speaker timestamps.

        Returns:
            List[dict]: List of sentences with speaker mapping.
        """
        turn_idx = 0
        _, end, speaker = speaker_timestamps[turn_idx]

        segment_index = 0
        segment_speaker_mapping = []
        while segment_index < len(transcript_segments):
            segment = transcript_segments[segment_index]
            segment_start, segment_end, segment_text = (
                _convert_s_to_ms(segment["start"]),
                _convert_s_to_ms(segment["end"]),
                segment["text"],
            )

            while segment_start > float(end) or abs(segment_start - float(end)) < 300:
                turn_idx += 1
                turn_idx = min(turn_idx, len(speaker_timestamps) - 1)
                _, end, speaker = speaker_timestamps[turn_idx]
                if turn_idx == len(speaker_timestamps) - 1:
                    end = segment_end
                    break

            if segment_end > float(end):
                words = segment["words"]

                word_index = next(
                    (
                        i
                        for i, word in enumerate(words)
                        if _convert_s_to_ms(word["start"]) > float(end)
                        or abs(_convert_s_to_ms(word["start"]) - float(end)) < 300
                    ),
                    None,
                )

                if word_index is not None:
                    _splitted_segment = segment_text.split()

                    if word_index > 0:
                        _segment_to_add = dict(
                            start=words[0]["start"],
                            end=words[word_index - 1]["end"],
                            text=" ".join(_splitted_segment[:word_index]),
                            speaker=speaker,
                            words=words[:word_index],
                        )

                    else:
                        _segment_to_add = dict(
                            start=words[0]["start"],
                            end=words[0]["end"],
                            text=_splitted_segment[0],
                            speaker=speaker,
                            words=words[:1],
                        )

                    segment_speaker_mapping.append(_segment_to_add)
                    transcript_segments.insert(
                        segment_index + 1,
                        dict(
                            start=words[word_index]["start"],
                            end=_convert_ms_to_s(segment_end),
                            text=" ".join(_splitted_segment[word_index:]),
                            words=words[word_index:],
                        ),
                    )
                else:
                    segment_speaker_mapping.append(
                        dict(
                            start=_convert_ms_to_s(segment_start),
                            end=_convert_ms_to_s(segment_end),
                            text=segment_text,
                            speaker=speaker,
                            words=words,
                        )
                    )
            else:
                segment_speaker_mapping.append(
                    dict(
                        start=_convert_ms_to_s(segment_start),
                        end=_convert_ms_to_s(segment_end),
                        text=segment_text,
                        speaker=speaker,
                        words=segment["words"],
                    )
                )

            segment_index += 1

        return segment_speaker_mapping

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
        for segment in transcript_words:
            text, speaker = segment["text"], segment["speaker"]
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

            current_sentence["text"] += text + " "
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
