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

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer

from wordcab_transcribe.services.vad_service import VADService


class TranscribeService:
    """Transcribe Service for audio files."""

    def __init__(
        self,
        model_path: str,
        compute_type: str,
        device: str,
        batch_size: Optional[int] = 1
    ) -> None:
        """Initialize the Transcribe Service.

        This service uses the WhisperModel from faster-whisper to transcribe audio files.

        Args:
            model_path (str): Path to the model checkpoint. This can be a local path or a URL.
            compute_type (str): Compute type to use for inference. Can be "int8", "int8_float16", "int16" or "float_16".
            device (str): Device to use for inference. Can be "cpu" or "cuda".
            batch_size (Optional[int], optional): Batch size to use for inference. Defaults to 1.
        """
        self.model = WhisperModel(model_path, device=device, compute_type=compute_type)
        self.tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language="en",  # Default language, to gain some speed
        )

        self._batch_size = batch_size
        self.sample_rate = 16000
        self._chunk_size = 30
        self._n_samples = self.sample_rate * self._chunk_size
        self._n_mels = 80
        
        assets_dir = Path(__file__).parent.parent / "assets" / "mel_filters.npz"
        with np.load(str(assets_dir)) as f:
            self.mel_filters = torch.from_numpy(f[f"mel_{self._n_mels}"]).to(device)


    def __call__(
        self,
        audio: Union[str, np.ndarray],
        source_lang: str,
        vad_service: VADService,
        **kwargs: Optional[dict]
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
        # Old one batch
        # segments, _ = self.model.transcribe(audio, language=source_lang, **kwargs)

        # results = [segment._asdict() for segment in segments]

        # return results

        if self.sample_rate != vad_service.sample_rate:
            self.sample_rate = vad_service.sample_rate

        vad_timestamps, audio = vad_service(audio, group_timestamps=False)
        # vad_timestamps = merge_chunks(vad_timestamps, self._chunk_size)

        if self.tokenizer.language_code != source_lang:
            self.tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task="transcribe",
                language=source_lang,
            )

        segments: List[dict] = []
        for idx, output in enumerate(
            self.pipeline(self._chunk(audio, vad_timestamps), batch_size=self._batch_size)
        ):
            text = output["text"]

            if self._batch_size in [0, 1, None]:
                text = text[0]

            segments.append(
                {
                    "text": text,
                    "start": vad_timestamps[idx]["start"],
                    "end": vad_timestamps[idx]["end"]
                }
            )

        return segments

    def pipeline(self, audio_chunk: torch.tensor, batch_size: int) -> None:
        """"""
        _padding = self._n_samples - audio_chunk.shape[0]
        features = self._log_mel_spectrogram(audio_chunk, padding=_padding)

        outputs = self._generate_segment_batched(features, self.tokenizer, self.options)

        return outputs

    def _generate_segment_batched(self) -> None:
        """"""

    def _chunk(self, audio: torch.tensor, vad_timestamps: List[dict]) -> torch.tensor:
        """
        Create chunks from the audio file based on the VAD timestamps.

        Args:
            audio (torch.tensor): Audio data.
            vad_timestamps (List[dict]): List of VAD timestamps.

        Returns:
            torch.tensor: One chunk of audio data as a tensor.
        """
        for chunk in vad_timestamps:
            f1 = int(chunk["start"] * self.sample_rate)
            f2 = int(chunk["end"] * self.sample_rate)

            yield audio[f1:f2]

    def _log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        n_fft: int = 400,
        hop_length: int = 160,
        padding: int = 0,
    ) -> torch.Tensor:
        """
        Compute the log-Mel spectrogram of a given audio tensor.

        Args:
            audio (torch.Tensor): Audio tensor of shape (n_samples,).
            n_fft (int, optional): Length of the FFT window. Defaults to 400.
            hop_length (int, optional): Hop length for the STFT. Defaults to 160.
            padding (int, optional): Padding to apply to the audio. Defaults to 0.

        Returns:
            torch.Tensor: Log-Mel spectrogram of shape (n_mels, T).
        """
        if padding > 0:
            audio = F.pad(audio, (0, padding))

        window = torch.hann_window(n_fft).to(audio.device)
        stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
        
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec
