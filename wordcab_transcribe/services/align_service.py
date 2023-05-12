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
from typing import Dict, List, Tuple, Union

import torch
import torchaudio
import whisperx
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


MODEL_MAPPING = OrderedDict(
    # (language, model_name, source)
    [
        ("ar", ("jonatasgrosman/wav2vec2-large-xlsr-53-arabic", "huggingface")),
        ("da", ("saattrupdan/wav2vec2-xls-r-300m-ftspeech", "huggingface")),
        ("de", ("WAV2VEC2_ASR_BASE_10K_DE", "torchaudio")),
        ("el", ("jonatasgrosman/wav2vec2-large-xlsr-53-greek", "huggingface")),
        ("en", ("WAV2VEC2_ASR_BASE_960H", "torchaudio")),
        ("es", ("WAV2VEC2_ASR_BASE_10K_ES", "torchaudio")),
        ("fa", ("jonatasgrosman/wav2vec2-large-xlsr-53-persian", "huggingface")),
        ("fi", ("jonatasgrosman/wav2vec2-large-xlsr-53-finnish", "huggingface")),
        ("fr", ("WAV2VEC2_ASR_BASE_10K_FR", "torchaudio")),
        ("he", ("imvladikon/wav2vec2-xls-r-300m-hebrew", "huggingface")),
        ("hu", ("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian", "huggingface")),
        ("it", ("WAV2VEC2_ASR_BASE_10K_IT", "torchaudio")),
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


class AlignService:
    """Alignment Service for transcribed audio files."""

    def __init__(self, device: str) -> None:
        """Initialize the Alignment Service."""
        self.device = device

        self.model_map = MODEL_MAPPING
        self.available_lang = self.model_map.keys()

    def __call__(
        self, filepath: str, transcript_segments: List[dict], source_lang: str
    ) -> None:
        """Run the alignment service on the given transcript segments and source language."""
        if source_lang not in self.available_lang:
            return transcript_segments

        model, metadata = self.load_model(source_lang)
        model = model.to(self.device)

        result_aligned = whisperx.align(
            transcript_segments, model, metadata, filepath, self.device
        )
        word_timestamps = result_aligned["word_segments"]

        del model
        torch.cuda.empty_cache()

        return word_timestamps

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
