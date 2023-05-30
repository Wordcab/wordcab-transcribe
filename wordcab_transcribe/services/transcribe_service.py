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
"""Transcribe Service for audio files."""

from typing import List, Optional, Union

import numpy as np
from faster_whisper import WhisperModel


class TranscribeService:
    """Transcribe Service for audio files."""

    def __init__(
        self,
        model_path: str,
        compute_type: str,
        device: str,
        num_workers: int,
    ) -> None:
        """Initialize the Transcribe Service.

        This service uses the WhisperModel from faster-whisper to transcribe audio files.

        Args:
            model_path (str): Path to the model checkpoint. This can be a local path or a URL.
            compute_type (str): Compute type to use for inference. Can be "int8", "int8_float16", "int16" or "float_16".
            device (str): Device to use for inference. Can be "cpu" or "cuda".
            num_workers (int): Number of workers to use for inference.
        """
        self.model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
            num_workers=num_workers,
        )

    def __call__(
        self, audio: Union[str, np.ndarray], source_lang: str, **kwargs: Optional[dict]
    ) -> List[dict]:
        """
        Run inference with the transcribe model.

        Args:
            audio (Union[str, np.ndarray]): Path to the audio file or audio data.
            source_lang (str): Language of the audio file.
            kwargs (Any): Additional arguments to pass to TranscribeService.

        Returns:
            List[dict]: List of segments with the following keys: "start", "end", "text", "confidence".
        """
        segments, _ = self.model.transcribe(audio, language=source_lang, **kwargs)

        results = [segment._asdict() for segment in segments]

        return results
