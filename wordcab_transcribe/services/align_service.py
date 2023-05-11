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
from typing import Dict, List, Tuple

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


MODEL_MAPPING = OrderedDict(
    [
        ("ar", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic", "hfhub"),
        ("da", "saattrupdan/wav2vec2-xls-r-300m-ftspeech", "hfhub"),
        ("de", "WAV2VEC2_ASR_BASE_10K_DE", "torchaudio"),
        ("el", "jonatasgrosman/wav2vec2-large-xlsr-53-greek", "hfhub"),
        ("en", "WAV2VEC2_ASR_BASE_960H", "torchaudio"),
        ("es", "WAV2VEC2_ASR_BASE_10K_ES", "torchaudio"),
        ("fa", "jonatasgrosman/wav2vec2-large-xlsr-53-persian", "hfhub"),
        ("fi", "jonatasgrosman/wav2vec2-large-xlsr-53-finnish", "hfhub"),
        ("fr", "WAV2VEC2_ASR_BASE_10K_FR", "torchaudio"),
        ("he", "imvladikon/wav2vec2-xls-r-300m-hebrew", "hfhub"),
        ("hu", "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian", "hfhub"),
        ("it", "WAV2VEC2_ASR_BASE_10K_IT", "torchaudio"),
        ("ja", "jonatasgrosman/wav2vec2-large-xlsr-53-japanese", "hfhub"),
        ("nl", "jonatasgrosman/wav2vec2-large-xlsr-53-dutch", "hfhub"),
        ("pl", "jonatasgrosman/wav2vec2-large-xlsr-53-polish", "hfhub"),
        ("pt", "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", "hfhub"),
        ("ru", "jonatasgrosman/wav2vec2-large-xlsr-53-russian", "hfhub"),
        ("tr", "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish", "hfhub"),
        ("uk", "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm", "hfhub"),
        ("zh", "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", "hfhub"),
    ]
)


class AlignService:
    """Alignment Service for transcribed audio files."""

    def __init__(self, device: str) -> None:
        """Initialize the Alignment Service."""
        self.device = device

        self.model_map = self.MODEL_MAPPING
        self.available_lang = self.model_map.keys()

    def __call__(self, transcript_segments: List[dict], source_lang: str) -> None:
        pass

    def load_model(self, language: str) -> None:
        """Load the model for the given language."""
        model_path, model_type = self.model_map[language]

        if model_type == "hfhub":
            model, align_dictionary = self._load_hf_model(model_path)
        elif model_type == "torchaudio":
            model, align_dictionary = self._load_torch_model(model_path)
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented.")

    def _load_torch_model(self, model_path: str) -> Tuple[torchaudio.models.Wav2Vec2Model, Dict[str, int]]:
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

        align_dictionary = {character.lower(): code for code, character in enumerate(labels)}

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
            character.lower(): code for character, code in processor.tokenizer.get_vocab().items()
        }

        return model, align_dictionary
