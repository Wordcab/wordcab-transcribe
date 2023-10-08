# Copyright 2023 The Wordcab Team. All rights reserved.
#
# Licensed under the Wordcab Transcribe License 0.1 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Wordcab/wordcab-transcribe/blob/main/LICENSE
#
# Except as expressly provided otherwise herein, and to the fullest
# extent permitted by law, Licensor provides the Software (and each
# Contributor provides its Contributions) AS IS, and Licensor
# disclaims all warranties or guarantees of any kind, express or
# implied, whether arising under any law or from any usage in trade,
# or otherwise including but not limited to the implied warranties
# of merchantability, non-infringement, quiet enjoyment, fitness
# for a particular purpose, or otherwise.
#
# See the License for the specific language governing permissions
# and limitations under the License.
"""Post-Processing Service for audio files."""

from typing import List, Tuple, Union

from wordcab_transcribe.models import (
    DiarizationOutput,
    DiarizationSegment,
    MultiChannelTranscriptionOutput,
    Timestamps,
    TranscriptionOutput,
    Utterance,
    Word,
)
from wordcab_transcribe.utils import convert_timestamp, format_punct


class PostProcessingService:
    """Post-Processing Service for audio files."""

    def __init__(self) -> None:
        """Initialize the PostProcessingService."""
        self.sample_rate = 16000

    def single_channel_speaker_mapping(
        self,
        transcript_segments: List[Utterance],
        speaker_timestamps: DiarizationOutput,
        word_timestamps: bool,
    ) -> List[Utterance]:
        """Run the post-processing functions on the inputs.

        The postprocessing pipeline is as follows:
        1. Map each transcript segment to its corresponding speaker.
        2. Group utterances of the same speaker together.

        Args:
            transcript_segments (List[Utterance]):
                List of transcript utterances.
            speaker_timestamps (DiarizationOutput):
                List of speaker timestamps.
            word_timestamps (bool):
                Whether to include word timestamps.

        Returns:
            List[Utterance]:
                List of utterances with speaker mapping.
        """
        segments_with_speaker_mapping = self.segments_speaker_mapping(
            transcript_segments,
            speaker_timestamps.segments,
        )

        utterances = self.reconstruct_utterances(
            segments_with_speaker_mapping, word_timestamps
        )

        return utterances

    def multi_channel_speaker_mapping(
        self, multi_channel_segments: List[MultiChannelTranscriptionOutput]
    ) -> TranscriptionOutput:
        """
        Run the multi-channel post-processing functions on the inputs by merging the segments based on the timestamps.

        Args:
            multi_channel_segments (List[MultiChannelTranscriptionOutput]):
                List of segments from multi speakers.

        Returns:
            TranscriptionOutput: List of sentences with speaker mapping.
        """
        words_with_speaker_mapping = [
            (segment.speaker, word)
            for output in multi_channel_segments
            for segment in output.segments
            for word in segment.words
        ]
        words_with_speaker_mapping.sort(key=lambda x: x[1].start)

        utterances: List[Utterance] = self.reconstruct_multi_channel_utterances(
            words_with_speaker_mapping
        )

        return utterances

    def segments_speaker_mapping(
        self,
        transcript_segments: List[Utterance],
        speaker_timestamps: List[DiarizationSegment],
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

        def _assign_speaker(
            mapping: list,
            seg_index: int,
            split: bool,
            current_speaker: str,
            current_split_len: int,
        ):
            """Assign speaker to the segment."""
            if split and len(mapping) > 1:
                last_split_len = len(mapping[seg_index - 1].text)
                if last_split_len > current_split_len:
                    current_speaker = mapping[seg_index - 1].speaker
                elif last_split_len < current_split_len:
                    mapping[seg_index - 1].speaker = current_speaker
            return current_speaker

        threshold = 0.3
        turn_idx = 0
        was_split = False
        _, end, speaker = speaker_timestamps[turn_idx]

        segment_index = 0
        segment_speaker_mapping = []
        while segment_index < len(transcript_segments):
            segment: Utterance = transcript_segments[segment_index]
            segment_start, segment_end, segment_text = (
                segment.start,
                segment.end,
                segment.text,
            )
            while (
                segment_start > float(end)
                or abs(segment_start - float(end)) < threshold
            ):
                turn_idx += 1
                turn_idx = min(turn_idx, len(speaker_timestamps) - 1)
                _, end, speaker = speaker_timestamps[turn_idx]
                if turn_idx == len(speaker_timestamps) - 1:
                    end = segment_end
                    break

            if segment_end > float(end) and abs(segment_end - float(end)) > threshold:
                words = segment.words
                word_index = next(
                    (
                        i
                        for i, word in enumerate(words)
                        if word.start > float(end)
                        or abs(word.start - float(end)) < threshold
                    ),
                    None,
                )

                if word_index is not None:
                    _split_segment = segment_text.split()

                    if word_index > 0:
                        text = " ".join(_split_segment[:word_index])
                        speaker = _assign_speaker(
                            segment_speaker_mapping,
                            segment_index,
                            was_split,
                            speaker,
                            len(text),
                        )

                        _segment_to_add = Utterance(
                            start=words[0].start,
                            end=words[word_index - 1].end,
                            text=text,
                            speaker=speaker,
                            words=words[:word_index],
                        )
                    else:
                        text = _split_segment[0]
                        speaker = _assign_speaker(
                            segment_speaker_mapping,
                            segment_index,
                            was_split,
                            speaker,
                            len(text),
                        )

                        _segment_to_add = Utterance(
                            start=words[0].start,
                            end=words[0].end,
                            text=_split_segment[0],
                            speaker=speaker,
                            words=words[:1],
                        )
                    segment_speaker_mapping.append(_segment_to_add)
                    transcript_segments.insert(
                        segment_index + 1,
                        Utterance(
                            start=words[word_index].start,
                            end=segment_end,
                            text=" ".join(_split_segment[word_index:]),
                            words=words[word_index:],
                        ),
                    )
                    was_split = True
                else:
                    speaker = _assign_speaker(
                        segment_speaker_mapping,
                        segment_index,
                        was_split,
                        speaker,
                        len(segment_text),
                    )
                    was_split = False

                    segment_speaker_mapping.append(
                        Utterance(
                            start=segment_start,
                            end=segment_end,
                            text=segment_text,
                            speaker=speaker,
                            words=words,
                        )
                    )
            else:
                speaker = _assign_speaker(
                    segment_speaker_mapping,
                    segment_index,
                    was_split,
                    speaker,
                    len(segment_text),
                )
                was_split = False

                segment_speaker_mapping.append(
                    Utterance(
                        start=segment_start,
                        end=segment_end,
                        text=segment_text,
                        speaker=speaker,
                        words=segment.words,
                    )
                )
            segment_index += 1

        return segment_speaker_mapping

    def reconstruct_utterances(
        self,
        transcript_segments: List[Utterance],
        word_timestamps: bool,
    ) -> List[Utterance]:
        """
        Reconstruct the utterances based on the speaker mapping.

        Args:
            transcript_words (List[Utterance]):
                List of transcript segments.
            word_timestamps (bool):
                Whether to include word timestamps.

        Returns:
            List[Utterance]:
                List of sentences with speaker mapping.
        """
        start_t0, end_t0, speaker_t0 = (
            transcript_segments[0].start,
            transcript_segments[0].end,
            transcript_segments[0].speaker,
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
            text, speaker = segment.text, segment.speaker
            start_t, end_t = segment.start, segment.end

            if speaker != previous_speaker:
                sentences.append(Utterance(**current_sentence))
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
                current_sentence["words"].extend(segment.words)

        # Catch the last sentence
        sentences.append(Utterance(**current_sentence))

        return sentences

    def reconstruct_multi_channel_utterances(
        self,
        transcript_words: List[Tuple[int, Word]],
    ) -> List[Utterance]:
        """
        Reconstruct multi-channel utterances based on the speaker mapping.

        Args:
            transcript_words (List[Tuple[int, Word]]):
                List of tuples containing the speaker and the word.

        Returns:
            List[Utterance]: List of sentences with speaker mapping.
        """
        speaker_t0, word = transcript_words[0]
        start_t0, end_t0 = word.start, word.end

        previous_speaker = speaker_t0
        current_sentence = {
            "speaker": speaker_t0,
            "start": start_t0,
            "end": end_t0,
            "text": "",
            "words": [],
        }

        sentences = []
        for speaker, word in transcript_words:
            start_t, end_t, text = word.start, word.end, word.word

            if speaker != previous_speaker:
                sentences.append(current_sentence)
                current_sentence = {
                    "speaker": speaker,
                    "start": start_t,
                    "end": end_t,
                    "text": "",
                }
                current_sentence["words"] = []
            else:
                current_sentence["end"] = end_t

            current_sentence["text"] += text
            previous_speaker = speaker
            current_sentence["words"].append(word)

        # Catch the last sentence
        sentences.append(current_sentence)

        for sentence in sentences:
            sentence["text"] = sentence["text"].strip()

        return [Utterance(**sentence) for sentence in sentences]

    def final_processing_before_returning(
        self,
        utterances: List[Utterance],
        offset_start: Union[float, None],
        timestamps_format: Timestamps,
        word_timestamps: bool,
    ) -> List[Utterance]:
        """
        Do final processing before returning the utterances to the API.

        Args:
            utterances (List[Utterance]):
                List of utterances.
            offset_start (Union[float, None]):
                Offset start.
            timestamps_format (Timestamps):
                Timestamps format. Can be `s`, `ms`, or `hms`.
            word_timestamps (bool):
                Whether to include word timestamps.

        Returns:
            List[Utterance]:
                List of utterances with final processing.
        """
        if offset_start is not None:
            offset_start = float(offset_start)
        else:
            offset_start = 0.0

        for utterance in utterances:
            utterance.text = format_punct(utterance.text)
            utterance.start = convert_timestamp(
                (utterance.start + offset_start), timestamps_format
            )
            utterance.end = convert_timestamp(
                (utterance.end + offset_start), timestamps_format
            )
            utterance.words = utterance.words if word_timestamps else None

        return utterances
