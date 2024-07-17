# Copyright 2024 The Wordcab Team. All rights reserved.
#
# Licensed under the MIT License (the "License");
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

from wordcab_transcribe.config import settings
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

        if settings.enable_punctuation_based_alignment:
            from deepmultilingualpunctuation import PunctuationModel

            self.punct_model = PunctuationModel(model="kredor/punctuate-all")
        else:
            self.punct_model = None

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
        """Map each word to its corresponding speaker based on speaker timestamps.

        Args:
            transcript_words (List[Word]): List of transcribed words with timing information.
            speaker_timestamps (List[DiarizationSegment]): List of speaker timestamps.

        Returns:
            List[Utterance]: List of utterances with speaker mapping.
        """

        def _create_utterance(start_word: Word, end_word: Word, words: List[Word], speaker: str) -> Utterance:
            return Utterance(
                start=start_word.start,
                end=end_word.end,
                text=" ".join(word.word for word in words),
                speaker=speaker,
                words=words
            )

        utterances = []
        current_speaker = None
        current_words = []
        speaker_index = 0
        _, speaker_end, speaker = speaker_timestamps[speaker_index]

        for segment in transcript_segments:
            for word in segment.words:
                while word.start >= speaker_end and speaker_index < len(speaker_timestamps) - 1:
                    speaker_index += 1
                    _, speaker_end, speaker = speaker_timestamps[speaker_index]

                if speaker != current_speaker:
                    if current_words:
                        utterances.append(_create_utterance(current_words[0], current_words[-1], current_words, current_speaker))
                        current_words = []
                    current_speaker = speaker

                current_words.append(word)

        if current_words:
            utterances.append(_create_utterance(current_words[0], current_words[-1], current_words, current_speaker))

        return utterances

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

    def punctuation_based_alignment(
        self,
        utterances: List[Utterance],
        speaker_timestamps: DiarizationOutput,
    ):
        pass
        # word_list = []
        # for utterance in utterances:
        #     for word in utterance.words:
        #         word_list.append(word.word)
        #
        # labled_words = self.punct_model.predict(word_list)
        #
        # ending_puncts = ".?!"
        # model_puncts = ".,;:!?"
        #
        # def is_acronym(w):
        #     return re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", w)
        #
        # for ix, (word, labeled_tuple) in enumerate(zip(word_list, labled_words)):
        #     if (
        #             word
        #             and labeled_tuple[1] in ending_puncts
        #             and (word[-1] not in model_puncts or is_acronym(word))
        #     ):
        #         word += labeled_tuple[1]
        #         if word.endswith(".."):
        #             word = word.rstrip(".")
        #         word_dict["word"] = word

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
                List of utterances after final processing.
        """
        if offset_start is not None:
            offset_start = float(offset_start)
        else:
            offset_start = 0.0

        final_utterances = []
        for utterance in utterances:
            # Check if the utterance is not empty
            if utterance.text.strip():
                utterance.text = format_punct(utterance.text)
                utterance.start = convert_timestamp(
                    (utterance.start + offset_start), timestamps_format
                )
                utterance.end = convert_timestamp(
                    (utterance.end + offset_start), timestamps_format
                )
                utterance.words = utterance.words if word_timestamps else None

                final_utterances.append(utterance)

        return final_utterances
