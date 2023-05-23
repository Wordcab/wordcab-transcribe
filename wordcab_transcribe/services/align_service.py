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
"""Alignment Service for transcribed audio files."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, TypedDict, Union

import nltk
import numpy as np
import pandas as pd
import torch
import torchaudio
from loguru import logger
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from wordcab_transcribe.utils import interpolate_nans


MODEL_MAPPING = OrderedDict(
    # (language, model_name, source)
    [
        ("ar", ("jonatasgrosman/wav2vec2-large-xlsr-53-arabic", "huggingface")),
        ("da", ("saattrupdan/wav2vec2-xls-r-300m-ftspeech", "huggingface")),
        ("de", ("VOXPOPULI_ASR_BASE_10K_DE", "torchaudio")),
        ("el", ("jonatasgrosman/wav2vec2-large-xlsr-53-greek", "huggingface")),
        ("en", ("WAV2VEC2_ASR_BASE_960H", "torchaudio")),
        ("es", ("VOXPOPULI_ASR_BASE_10K_ES", "torchaudio")),
        ("fa", ("jonatasgrosman/wav2vec2-large-xlsr-53-persian", "huggingface")),
        ("fi", ("jonatasgrosman/wav2vec2-large-xlsr-53-finnish", "huggingface")),
        ("fr", ("VOXPOPULI_ASR_BASE_10K_FR", "torchaudio")),
        ("he", ("imvladikon/wav2vec2-xls-r-300m-hebrew", "huggingface")),
        ("hu", ("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian", "huggingface")),
        ("it", ("VOXPOPULI_ASR_BASE_10K_IT", "torchaudio")),
        ("ja", ("jonatasgrosman/wav2vec2-large-xlsr-53-japanese", "huggingface")),
        ("nl", ("jonatasgrosman/wav2vec2-large-xlsr-53-dutch", "huggingface")),
        ("pl", ("jonatasgrosman/wav2vec2-large-xlsr-53-polish", "huggingface")),
        ("pt", ("jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", "huggingface")),
        ("ru", ("jonatasgrosman/wav2vec2-large-xlsr-53-russian", "huggingface")),
        ("tr", ("mpoyraz/wav2vec2-xls-r-300m-cv7-turkish", "huggingface")),
        ("uk", ("Yehor/wav2vec2-xls-r-300m-uk-with-small-lm", "huggingface")),
        ("zh", ("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", "huggingface")),
    ]
)


@dataclass
class Point:
    """Point class for alignment."""

    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    """Segment of a speech."""

    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        """Return a string representation of the segment."""
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        """Return the length of the segment."""
        return self.end - self.start


class SingleWordSegment(TypedDict):
    """A single word of a speech."""

    word: str
    start: float
    end: float
    score: float


class SingleCharSegment(TypedDict):
    """A single char of a speech."""

    char: str
    start: float
    end: float
    score: float


class SingleSegment(TypedDict):
    """A single segment (up to multiple sentences) of a speech."""

    start: float
    end: float
    text: str


class SingleAlignedSegment(TypedDict):
    """A single segment (up to multiple sentences) of a speech with word alignment."""

    start: float
    end: float
    text: str
    words: List[SingleWordSegment]
    chars: Optional[List[SingleCharSegment]]


class TranscriptionResult(TypedDict):
    """A list of segments and word segments of a speech."""

    segments: List[SingleSegment]
    language: str


class AlignService:
    """Alignment Service for transcribed audio files."""

    def __init__(self, device: str) -> None:
        """Initialize the Alignment Service."""
        self.device = device
        self.sample_rate = 16000

        self.model_map = MODEL_MAPPING
        self.available_lang = self.model_map.keys()

    def __call__(
        self, filepath: str, transcript_segments: List[dict], source_lang: str
    ) -> List[SingleAlignedSegment]:
        """
        Run the alignment service on the given transcript segments and source language.

        Args:
            filepath (str): The path to the audio file.
            transcript_segments (List[dict]): The transcript segments to align.
            source_lang (str): The source language of the transcript segments.

        Returns:
            List[SingleAlignedSegment]: The aligned transcript segments.
        """
        if source_lang not in self.available_lang:
            return transcript_segments

        model, metadata = self.load_model(source_lang)
        model = model.to(self.device)

        result_aligned = self.align(
            transcript_segments, model, metadata, filepath, self.device
        )

        del model
        torch.cuda.empty_cache()

        return result_aligned

    def load_model(
        self, language: str
    ) -> Tuple[Union[Wav2Vec2ForCTC, torchaudio.models.Wav2Vec2Model], Dict[str, int]]:
        """
        Load the model for the given language from torch or huggingface hub.

        Args:
            language (str): The language to load the model for.

        Returns:
            Tuple[Union[Wav2Vec2ForCTC, torchaudio.models.Wav2Vec2Model], Dict[str, int]]: The model its metadata.

        Raises:
            ValueError: If the model could not be loaded.
            NotImplementedError: If the model type is not implemented.
        """
        model_path, model_type = self.model_map[language]

        try:
            if model_type == "huggingface":
                model, align_dictionary = self._load_hf_model(model_path)
            elif model_type == "torchaudio":
                model, align_dictionary = self._load_torch_model(model_path)
            else:
                raise NotImplementedError(f"Model type {model_type} not implemented.")

        except Exception as e:
            raise ValueError(f"Could not load model for language {language}.") from e

        metadata = {
            "language": language,
            "dictionary": align_dictionary,
            "type": model_type,
        }

        return model, metadata

    def _load_torch_model(
        self, model_path: str
    ) -> Tuple[torchaudio.models.Wav2Vec2Model, Dict[str, int]]:
        """
        Load the torch model from the torchaudio pipelines.

        Args:
            model_path (str): The path to the model.

        Returns:
            Tuple[torchaudio.models.Wav2Vec2Model, Dict[str, int]]: The model and the align dictionary.
        """
        bundle = torchaudio.pipelines.__dict__[model_path]

        model = bundle.get_model()
        labels = bundle.get_labels()

        align_dictionary = {
            character.lower(): code for code, character in enumerate(labels)
        }

        return model, align_dictionary

    def _load_hf_model(self, model_path: str) -> Tuple[Wav2Vec2ForCTC, Dict[str, int]]:
        """
        Load the huggingface model from the huggingface hub.

        Args:
            model_path (str): The path to the model.

        Returns:
            Tuple[Wav2Vec2ForCTC, Dict[str, int]]: The model and the align dictionary.
        """
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
        processor = Wav2Vec2Processor.from_pretrained(model_path)

        align_dictionary = {
            character.lower(): code
            for character, code in processor.tokenizer.get_vocab().items()
        }

        return model, align_dictionary

    # flake8: noqa: C901
    def align(
        self,
        transcript: Iterator[SingleSegment],
        model: torch.nn.Module,
        align_model_metadata: dict,
        audio_path: str,
        device: str,
        interpolate_method: Optional[str] = "nearest",
        return_char_alignments: Optional[bool] = False,
    ) -> List[SingleAlignedSegment]:
        """Align the given transcript to the given audio.

        Taken from https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py.

        Args:
            transcript (Iterator[SingleSegment]): The transcript to align.
            model (torch.nn.Module): The model to use for alignment.
            align_model_metadata (dict): The metadata of the model.
            audio_path (str): The audio to align the transcript to.
            device (str): The device to run the model on.
            interpolate_method (str, optional): The method to use for interpolation. Defaults to "nearest".
            return_char_alignments (bool, optional): Whether to return the character alignments. Defaults to False.

        Returns:
            AlignedTranscriptionResult: The aligned transcription result.

        Raises:
            NotImplementedError: If the model type is not implemented.
        """
        audio, sample_rate = torchaudio.load(audio_path, normalize=True)

        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )
            audio = resampler(audio)

        max_duration = float(audio.shape[1]) / sample_rate

        model_dictionary = align_model_metadata["dictionary"]
        model_lang = align_model_metadata["language"]
        model_type = align_model_metadata["type"]

        # 1. Preprocess to keep only characters in dictionary
        for segment in transcript:
            # strip spaces at beginning / end, but keep track of the amount.
            num_leading = len(segment["text"]) - len(segment["text"].lstrip())
            num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
            text = segment["text"]

            per_word = text.split(" ") if model_lang not in ["ja", "zh"] else text

            clean_char, clean_cdx = [], []
            for cdx, char in enumerate(text):
                char_ = (
                    char.lower().replace(" ", "|")
                    if model_lang not in ["ja", "zh"]
                    else char.lower()
                )

                # ignore whitespace at beginning and end of transcript
                if cdx < num_leading:
                    pass
                elif cdx > len(text) - num_trailing - 1:
                    pass
                elif char_ in model_dictionary.keys():
                    clean_char.append(char_)
                    clean_cdx.append(cdx)

            clean_wdx = []
            for wdx, wrd in enumerate(per_word):
                if any([c in model_dictionary.keys() for c in wrd]):
                    clean_wdx.append(wdx)

            sentence_spans = list(
                nltk.tokenize.punkt.PunktSentenceTokenizer().span_tokenize(text)
            )

            segment["clean_char"] = clean_char
            segment["clean_cdx"] = clean_cdx
            segment["clean_wdx"] = clean_wdx
            segment["sentence_spans"] = sentence_spans

        aligned_segments: List[SingleAlignedSegment] = []

        # 2. Get prediction matrix from alignment model & align
        for segment in transcript:
            t1 = segment["start"]
            t2 = segment["end"]
            text = segment["text"]

            aligned_seg: SingleAlignedSegment = {
                "start": t1,
                "end": t2,
                "text": text,
                "words": [],
            }

            if return_char_alignments:
                aligned_seg["chars"] = []

            # check we can align
            if len(segment["clean_char"]) == 0:
                logger.debug(
                    f"Failed to align segment ({segment['text']}): no characters in this segment "
                    "found in model dictionary, resorting to original..."
                )
                aligned_segments.append(aligned_seg)
                continue

            if t1 >= max_duration or t2 - t1 < 0.02:
                logger.debug(
                    "Failed to align segment: original start time longer than audio duration, skipping..."
                )
                aligned_segments.append(aligned_seg)
                continue

            text_clean = "".join(segment["clean_char"])
            tokens = [model_dictionary[c] for c in text_clean]

            f1 = int(t1 * self.sample_rate)
            f2 = int(t2 * self.sample_rate)

            # TODO: Probably can get some speedup gain with batched inference here
            waveform_segment = audio[:, f1:f2]

            with torch.inference_mode():
                if model_type == "torchaudio":
                    emissions, _ = model(waveform_segment.to(device))
                elif model_type == "huggingface":
                    emissions = model(waveform_segment.to(device)).logits
                else:
                    raise NotImplementedError(
                        f"Align model of type {model_type} not supported."
                    )
                emissions = torch.log_softmax(emissions, dim=-1)

            emission = emissions[0].cpu().detach()

            blank_id = 0
            for char, code in model_dictionary.items():
                if char == "[pad]" or char == "<pad>":
                    blank_id = code

            trellis = self.get_trellis(emission, tokens, blank_id)
            path = self.backtrack(trellis, emission, tokens, blank_id)

            if path is None:
                logger.debug(
                    f'Failed to align segment ({segment["text"]}): backtrack failed, resorting to original...'
                )
                aligned_segments.append(aligned_seg)
                continue

            char_segments = self.merge_repeats(path, text_clean)

            duration = t2 - t1
            ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

            # assign timestamps to aligned characters
            char_segments_arr = []
            word_idx = 0
            for cdx, char in enumerate(text):
                start, end, score = None, None, None
                if cdx in segment["clean_cdx"]:
                    char_seg = char_segments[segment["clean_cdx"].index(cdx)]
                    start = round(char_seg.start * ratio + t1, 3)
                    end = round(char_seg.end * ratio + t1, 3)
                    score = round(char_seg.score, 3)

                char_segments_arr.append(
                    {
                        "char": char,
                        "start": start,
                        "end": end,
                        "score": score,
                        "word-idx": word_idx,
                    }
                )

                # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
                if model_lang in ["ja", "zh"]:
                    word_idx += 1
                elif cdx == len(text) - 1 or text[cdx + 1] == " ":
                    word_idx += 1

            char_segments_arr = pd.DataFrame(char_segments_arr)

            aligned_subsegments = []
            # assign sentence_idx to each character index
            char_segments_arr["sentence-idx"] = None
            for sdx, (sstart, send) in enumerate(segment["sentence_spans"]):
                curr_chars = char_segments_arr.loc[
                    (char_segments_arr.index >= sstart)
                    & (char_segments_arr.index <= send)
                ]
                char_segments_arr.loc[
                    (char_segments_arr.index >= sstart)
                    & (char_segments_arr.index <= send),
                    "sentence-idx",
                ] = sdx

                sentence_text = text[sstart:send]
                sentence_start = curr_chars["start"].min()
                sentence_end = curr_chars["end"].max()
                sentence_words = []

                for word_idx in curr_chars["word-idx"].unique():
                    word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
                    word_text = "".join(word_chars["char"].tolist()).strip()
                    if len(word_text) == 0:
                        continue
                    word_chars = word_chars[word_chars["char"] != " "]
                    word_start = word_chars["start"].min()
                    word_end = word_chars["end"].max()
                    word_score = round(word_chars["score"].mean(), 3)

                    # -1 indicates unalignable
                    word_segment = {"word": word_text}

                    if not np.isnan(word_start):
                        word_segment["start"] = word_start
                    if not np.isnan(word_end):
                        word_segment["end"] = word_end
                    if not np.isnan(word_score):
                        word_segment["score"] = word_score

                    sentence_words.append(word_segment)

                aligned_subsegments.append(
                    {
                        "text": sentence_text,
                        "start": sentence_start,
                        "end": sentence_end,
                        "words": sentence_words,
                    }
                )

                if return_char_alignments:
                    curr_chars = curr_chars[["char", "start", "end", "score"]]
                    curr_chars.fillna(-1, inplace=True)
                    curr_chars = curr_chars.to_dict("records")
                    curr_chars = [
                        {key: val for key, val in char.items() if val != -1}
                        for char in curr_chars
                    ]
                    aligned_subsegments[-1]["chars"] = curr_chars

            aligned_subsegments = pd.DataFrame(aligned_subsegments)
            aligned_subsegments["start"] = interpolate_nans(
                aligned_subsegments["start"], method=interpolate_method
            )
            aligned_subsegments["end"] = interpolate_nans(
                aligned_subsegments["end"], method=interpolate_method
            )
            # concatenate sentences with same timestamps
            agg_dict = {"text": " ".join, "words": "sum"}
            if return_char_alignments:
                agg_dict["chars"] = "sum"
            aligned_subsegments = aligned_subsegments.groupby(
                ["start", "end"], as_index=False
            ).agg(agg_dict)
            aligned_subsegments = aligned_subsegments.to_dict("records")
            aligned_segments += aligned_subsegments

        return aligned_segments

    def get_trellis(self, emission, tokens, blank_id=0):
        """Get trellis for Viterbi decoding."""
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        # Trellis has extra diemsions for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code.
        trellis = torch.empty((num_frame + 1, num_tokens + 1))
        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
        trellis[0, -num_tokens:] = -float("inf")
        trellis[-num_tokens:, 0] = float("inf")

        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis

    def backtrack(self, trellis, emission, tokens, blank_id=0):
        """Backtrack to find the most likely token sequence."""
        # Note:
        # j and t are indices for trellis, which has extra dimensions
        # for time and tokens at the beginning.
        # When referring to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when referring to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            # 2. Store the path with frame-wise probability.
            prob = (
                emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
            )
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(j - 1, t - 1, prob))

            # 3. Update the token
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            # failed
            return None
        return path[::-1]

    def merge_repeats(self, path, transcript):
        """Merge repeated tokens in the path."""
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments
