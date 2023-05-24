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
import torch.nn.functional as F  # noqa N812
from ctranslate2 import StorageView
from ctranslate2.models import WhisperGenerationResult
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import (
    TranscriptionOptions,
    get_ctranslate2_storage,
    get_suppressed_tokens,
)
from torch.utils.data import DataLoader, IterableDataset

from wordcab_transcribe.services.vad_service import VadService


class AudioDataset(IterableDataset):
    """Audio Dataset for transcribing audio files in batches."""

    def __init__(
        self,
        audio_chunks: List[torch.tensor],
        n_samples: int,
        mel_filters: torch.Tensor,
    ) -> None:
        """
        Initialize the Audio Dataset for transcribing audio files in batches.

        Args:
            audio_chunks (List[torch.tensor]): List of audio chunks.
            n_samples (int): Number of samples.
            mel_filters (torch.Tensor): Mel filters tensor.
        """
        self.audio_chunks = audio_chunks
        self.n_samples = n_samples
        self.mel_filters = mel_filters

    def __iter__(self) -> iter:
        """Iterate over the audio chunks and yield the features."""
        for audio_chunk in self.audio_chunks:
            _padding = self.n_samples - audio_chunk.shape[0]
            features = self._log_mel_spectrogram(audio_chunk, padding=_padding)

            yield features

    def _log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        n_fft: Optional[int] = 400,
        hop_length: Optional[int] = 160,
        padding: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        Compute the log-Mel spectrogram of a given audio tensor.

        Args:
            audio (torch.Tensor): Audio tensor of shape (n_samples,).
            n_fft (int, optional): Number of FFT points. Defaults to 400.
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


class TranscribeService:
    """Transcribe Service for audio files."""

    def __init__(
        self,
        model_path: str,
        compute_type: str,
        device: str,
        batch_size: Optional[int] = 32,
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
            self.mel_filters = torch.from_numpy(f[f"mel_{self._n_mels}"])

        self.options = TranscriptionOptions(
            beam_size=5,
            best_of=5,
            patience=1,
            length_penalty=1,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=True,
            temperatures=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            initial_prompt=None,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=get_suppressed_tokens(self.tokenizer, [-1]),
            without_timestamps=False,
            max_initial_timestamp=1.0,
            word_timestamps=False,
            prepend_punctuations="\"'“¿([{-",
            append_punctuations="\"'.。,，!！?？:：”)]}、",
        )

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        source_lang: str,
        vad_service: VadService,
        **kwargs: Optional[dict],
    ) -> List[dict]:
        """
        Run inference with the transcribe model.

        Args:
            audio (Union[str, np.ndarray]): Path to the audio file or audio data.
            source_lang (str): Language of the audio file.
            vad_service (VadService): VadService to use for splitting the audio file into segments.
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

        audio_chunks = []
        for chunk in vad_timestamps:
            audio_chunks.append(audio[chunk["start"] : chunk["end"]])

        segments: List[dict] = []
        for idx, output in enumerate(
            self.pipeline(audio_chunks, batch_size=self._batch_size)
        ):
            segments.append(
                {
                    "text": output,
                    "start": vad_timestamps[idx]["start"] / self.sample_rate,
                    "end": vad_timestamps[idx]["end"] / self.sample_rate,
                }
            )

        return segments

    def pipeline(self, audio_chunks: torch.tensor, batch_size: int) -> List[dict]:
        """
        Transcription pipeline for audio chunks in batches.

        Args:
            audio_chunks (torch.tensor): Audio chunks to transcribe.
            batch_size (int): Batch size to use for inference.

        Returns:
            List[dict]: List of segments with the following keys: "start", "end", "text", "confidence".
        """
        dataset = AudioDataset(audio_chunks, self._n_samples, self.mel_filters)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        outputs = []

        for batch in dataloader:
            batch_outputs = self._generate_segment_batched(
                batch, self.tokenizer, self.options
            )
            outputs.extend(batch_outputs)

        return outputs

    def _generate_segment_batched(
        self, batch: torch.tensor, tokenizer: Tokenizer, options: TranscriptionOptions
    ) -> List[dict]:
        """
        Use the ctranslate2 Whisper model to generate text from audio chunks.

        Args:
            batch (torch.tensor): Audio chunks to transcribe.
            tokenizer (Tokenizer): Tokenizer to use for encoding the text.
            options (TranscriptionOptions): Options to use for transcription.

        Returns:
            List[dict]: List of segments with the following keys: "start", "end", "text", "confidence".
        """
        batch_size = batch.size(0)

        all_tokens = []
        prompt_reset_since = 0

        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)

        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.model.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        encoder_output = self._encode(batch)

        # TODO: Could be better to get the results as np.ndarray/torch.tensor and not as a class for speed
        # Atm, we need to extract the results as a Python list which is slow because we get this results:
        # https://opennmt.net/CTranslate2/python/ctranslate2.models.WhisperGenerationResult.html
        # TODO: We access the inherited ctranslate2 model for generation here. This is not ideal.
        result: WhisperGenerationResult = self.model.model.generate(
            encoder_output,
            [prompt] * batch_size,
            beam_size=5,
            patience=1,
            length_penalty=1,
            return_scores=True,
            return_no_speech_prob=True,
            suppress_blank=False,
        )

        decoded_outputs = self._decode_batch(result)

        return decoded_outputs

    def _encode(self, features: np.ndarray) -> StorageView:
        """
        Encode a batch of features using the ctranslate2 Whisper model.

        Args:
            features (np.ndarray): Batch of features

        Returns:
            StorageView: Encoded features.
        """
        # TODO: We call the inherited model here, because faster_whisper model does not allow to
        # access the device and the device_index. We should fix this in the future.
        to_cpu = (
            self.model.model.device == "cuda" and len(self.model.model.device_index) > 1
        )

        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)

        features = get_ctranslate2_storage(features)

        return self.model.model.encode(features, to_cpu=to_cpu)

    def _decode_batch(self, whisper_results: WhisperGenerationResult) -> List[str]:
        """
        Extract the token ids from the sequences ids and decode them using the tokenizer.

        Args:
            whisper_results (WhisperGenerationResult): Whisper generation results.

        Returns:
            List[str]: List of decoded texts.
        """
        tokens_to_decode = [
            [token for token in result.sequences_ids[0] if token < self.tokenizer.eot]
            for result in whisper_results
        ]
        # TODO: We call the inherited tokenizer here, because faster_whisper tokenizer
        # doesn't have the decode_batch method. We should fix this in the future.
        return self.tokenizer.tokenizer.decode_batch(tokens_to_decode)
