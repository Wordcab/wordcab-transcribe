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

import math
import os
import zlib
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F  # noqa N812
import torchaudio
from ctranslate2.models import WhisperGenerationResult
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_ctranslate2_storage
from torch.utils.data import DataLoader, IterableDataset

from wordcab_transcribe.logging import time_and_tell


# Word implementation from faster-whisper:
# https://github.com/guillaumekln/faster-whisper/blob/master/faster_whisper/transcribe.py#L24
class Word(NamedTuple):
    """Word unit for word_timestamps option."""

    start: float
    end: float
    word: str
    probability: float


class AudioDataset(IterableDataset):
    """Audio Dataset for transcribing audio files in batches."""

    def __init__(
        self,
        audio: Union[str, torch.Tensor],
        chunk_size: int,
        hop_length: int,
        mel_filters: torch.Tensor,
        n_fft: int,
        n_samples: int,
        sample_rate: int,
    ) -> None:
        """
        Initialize the Audio Dataset for transcribing audio files in batches.

        Args:
            audio (Union[str, torch.Tensor]): Audio file path or audio tensor.
            chunk_size (int): Size of audio chunks.
            hop_length (int): Hop length for the STFT.
            mel_filters (torch.Tensor): Mel filters to apply to the STFT.
            n_fft (int): Size of the FFT.
            n_samples (int): Number of samples to pad the audio.
            sample_rate (int): Sample rate of the audio.
        """
        self.chunk_size = chunk_size
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_samples = n_samples
        self.mel_filters = mel_filters
        self.sample_rate = sample_rate

        if isinstance(audio, str):
            waveform = self.read_audio(audio)
        elif isinstance(audio, torch.Tensor):
            waveform = audio
        else:
            raise TypeError("Audio must be a string or a tensor.")

        (
            self.indexes,
            _audio_chunks,
            self.time_offsets,
            self.segment_durations,
        ) = self.create_chunks(waveform)

        self.features = [
            self._log_mel_spectrogram(chunk, padding=self.n_samples - chunk.shape[-1])
            for chunk in _audio_chunks
        ]

    def __len__(self) -> int:
        """Get the number of audio chunks."""
        return len(self.indexes)

    def __iter__(self) -> Dict[str, Union[torch.Tensor, float]]:
        """Iterate over the audio chunks and yield the features."""
        for index, feature, time_offset, segment_duration in zip(
            self.indexes, self.features, self.time_offsets, self.segment_durations
        ):
            yield {
                "index": index,
                "feature": feature,
                "time_offset": time_offset,
                "segment_duration": segment_duration,
            }

    def read_audio(self, filepath: str) -> torch.Tensor:
        """Read an audio file and return the audio tensor."""
        wav, sr = torchaudio.load(filepath)

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            transform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )
            wav = transform(wav)
            sr = self.sample_rate

        return wav.squeeze(0)

    @time_and_tell
    def create_chunks(
        self, waveform: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        """
        Create 30-second chunks from the audio file loaded as a tensor.

        Args:
            waveform (torch.Tensor): Audio tensor of shape (n_samples,).

        Returns:
            Tuple[List[torch.Tensor], List[int], List[float]]: Tuple of audio chunks,
        """
        num_segments = math.ceil(waveform.size(0) / self.n_samples)
        indexes = [i for i in range(num_segments)]

        segments = [
            waveform[i * self.n_samples : (i + 1) * self.n_samples]
            for i in range(num_segments)
        ]

        time_offsets = [(i * self.chunk_size) for i in range(num_segments)]
        segment_durations = [
            self.chunk_size
            if len(segment) == self.n_samples
            else len(segment) / self.sample_rate
            for segment in segments
        ]

        return indexes, segments, time_offsets, segment_durations

    def _log_mel_spectrogram(
        self, audio: torch.Tensor, padding: int = 0
    ) -> torch.Tensor:
        """
        Compute the log-Mel spectrogram of a given audio tensor.

        Args:
            audio (torch.Tensor): Audio tensor of shape (n_samples,).
            padding (int): Number of samples to pad the audio.

        Returns:
            torch.Tensor: Log-Mel spectrogram of shape (n_mels, T).
        """
        if padding > 0:
            audio = F.pad(audio, (0, padding))

        window = torch.hann_window(self.n_fft).to(audio.device)
        stft = torch.stft(
            audio, self.n_fft, self.hop_length, window=window, return_complex=True
        )

        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec


class FallBackDataset(IterableDataset):
    """Custom Dataset for transcribing fallback segments in batches."""

    def __init__(self, failed_segments: List[Dict[str, Any]]) -> None:
        """
        Initialize the Dataset.

        Args:
            failed_segments (List[Dict[str, Any]]): List of failed segments.
        """
        self.segments = failed_segments

    def __iter__(self) -> Dict[str, Any]:
        """
        Iterate over the failed segments and yield the features.

        Yields:
            Dict[str, Any]: Dictionary containing the features.
            A segment looks like this:
            {
                "index": 0,  # Index of the segment in the original list of segments.
                "feature": torch.Tensor,  # Tensor of shape (n_mels, T).
                "time_offset": 0,
                "segment_duration": 30.0,
            }
        """
        yield from self.segments


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
        self.tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language="en",  # Default language, to gain some speed
        )

        self._batch_size = 8  # TODO: Make this configurable
        self.sample_rate = 16000

        self.n_fft = 400
        self.n_mels = 80
        self.chunk_size = 30
        self.hop_length = 160

        self.n_samples = self.sample_rate * self.chunk_size

        assets_dir = Path(__file__).parent.parent / "assets" / "mel_filters.npz"
        with np.load(str(assets_dir)) as f:
            self.mel_filters = torch.from_numpy(f[f"mel_{self.n_mels}"])

        self.compression_ratio_threshold = 2.4
        self.log_probability_threshold = -1.0

    def __call__(
        self,
        audio: Union[str, torch.Tensor],
        source_lang: str,
        **kwargs: Optional[dict],
    ) -> List[dict]:
        """
        Run inference with the transcribe model.

        Args:
            audio (Union[str, torch.Tensor]): Audio file to transcribe.
            source_lang (str): Language of the audio file.
            kwargs (Any): Additional arguments to pass to TranscribeService.

        Returns:
            List[dict]: List of segments with the following keys: "start", "end", "text", "confidence".
        """
        if self.tokenizer.language_code != source_lang:
            self.tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task="transcribe",
                language=source_lang,
            )

        outputs = self.pipeline(audio, batch_size=self._batch_size)

        return outputs

    @time_and_tell
    def pipeline(self, audio: Union[str, torch.Tensor], batch_size: int) -> List[dict]:
        """
        Transcription pipeline for audio chunks in batches.

        Args:
            audio (Union[str, torch.Tensor]): Audio file to transcribe.
            batch_size (int): Batch size to use for inference.

        Returns:
            List[dict]: List of segments with the following keys: "start", "end", "text".
        """
        dataset = AudioDataset(
            audio=audio,
            chunk_size=self.chunk_size,
            hop_length=self.hop_length,
            mel_filters=self.mel_filters,
            n_fft=self.n_fft,
            n_samples=self.n_samples,
            sample_rate=self.sample_rate,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self._collate_fn,
        )

        _outputs = [None for _ in range(len(dataset))]

        # The first pass of inference is done with non-greedy settings to achieve better results.
        beam_size = 5
        num_hypotheses = 1
        patience = 1.0
        sampling_top_k = 1
        temperature = 1.0
        stop_temperature = None

        while True:
            outputs_that_need_reprocessing = []

            for batch in dataloader:
                batch_outputs = self._generate_segment_batched(
                    features=batch["features"],
                    time_offsets=batch["time_offsets"],
                    segment_durations=batch["segment_durations"],
                    tokenizer=self.tokenizer,
                    beam_size=beam_size,
                    num_hypotheses=num_hypotheses,
                    patience=patience,
                    sampling_top_k=sampling_top_k,
                    temperature=temperature,
                )

                for output_index, output in enumerate(batch_outputs):
                    if output["need_fallback"]:
                        outputs_that_need_reprocessing.append(
                            {
                                "index": batch["indexes"][output_index],
                                "feature": batch["features"][output_index],
                                "time_offset": batch["time_offsets"][output_index],
                                "segment_duration": batch["segment_durations"][
                                    output_index
                                ],
                            }
                        )
                    else:
                        _outputs[batch["indexes"][output_index]] = output["segments"]

            if len(outputs_that_need_reprocessing) > 0 and stop_temperature != 1.0:
                dataloader = DataLoader(
                    FallBackDataset(outputs_that_need_reprocessing),
                    batch_size=batch_size,
                    collate_fn=self._collate_fn,
                )
                # The second pass of inference is done with greedy settings to speed up the process.
                beam_size = 1
                num_hypotheses = 5
                sampling_top_k = 0
                temperature = (temperature + 0.2) if temperature != 1.0 else 0.2
                stop_temperature = temperature  # Used to stop the loop if the temperature reaches 1.0 again.
            else:
                break  # All segments have been processed successfully.

        outputs = [item for sublist in _outputs for item in sublist]

        return outputs

    # This is an adapted version of the faster-whisper transcription pipeline:
    # https://github.com/guillaumekln/faster-whisper/blob/master/faster_whisper/transcribe.py
    @time_and_tell
    def _generate_segment_batched(
        self,
        features: torch.Tensor,
        time_offsets: List[float],
        segment_durations: List[float],
        tokenizer: Tokenizer,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
        length_penalty: float = 1.0,
        patience: int = 1.0,
        prefix: Optional[str] = None,
        num_hypotheses: int = 1,
        sampling_top_k: int = 1,
        temperature: float = 1.0,
        without_timestamps: bool = False,
    ) -> List[dict]:
        """
        Use the ctranslate2 Whisper model to generate text from audio chunks.

        Args:
            features (torch.Tensor): List of audio chunks.
            time_offsets (List[float]): Time offsets for the audio chunks.
            segment_durations (List[float]): Durations of the audio chunks.
            tokenizer (Tokenizer): Tokenizer to use for encoding the text.
            beam_size (int): Beam size to use for beam search.
            length_penalty (float): Length penalty to use for beam search.
            initial_prompt (Optional[str]): Initial prompt to use for the generation.
            num_hypotheses (int): Number of hypotheses used by generate.
            patience (int): Patience to use for beam search.
            prefix (Optional[str]): Prefix to use for the generation.
            sampling_top_k (int): Sampling top k to use for sampling.
            temperature (float): Temperature to use for sampling.
            without_timestamps (bool): Whether to remove timestamps from the generated text.

        Returns:
            List[dict]: List of segments with the following keys: "start", "end", "text", "confidence".
        """
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        batch_size = features.size(0)

        all_tokens = []
        prompt_reset_since = 0

        if initial_prompt is not None:
            initial_prompt = " " + initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)

        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.model.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=without_timestamps,
            prefix=prefix,
        )

        features = get_ctranslate2_storage(features)

        # TODO: Could be better to get the results as np.ndarray/torch.tensor and not as a class for speed
        # Atm, we need to extract the results as a Python list which is slow because we get this results:
        # https://opennmt.net/CTranslate2/python/ctranslate2.models.WhisperGenerationResult.html
        # TODO: We access the inherited ctranslate2 model for generation here. This is not ideal.
        result: WhisperGenerationResult = self.model.model.generate(
            features,
            [prompt] * batch_size,
            beam_size=beam_size,
            patience=patience,
            num_hypotheses=num_hypotheses,
            length_penalty=length_penalty,
            return_scores=True,
            return_no_speech_prob=True,
            suppress_blank=False,
            sampling_temperature=temperature,
            sampling_topk=sampling_top_k,
        )

        outputs = []
        for res, time_offset, segment_duration in zip(
            result, time_offsets, segment_durations
        ):
            current_segments = []
            tokens = res.sequences_ids[0]
            segment_score = res.scores[0]
            _text = tokenizer.decode(tokens).strip()

            compression_ratio, average_log_probability = self._get_quality_metrics(
                tokens,
                _text,
                segment_score,
                length_penalty,
            )

            # We check if the segment is valid based on the metrics thresholds.
            if (
                average_log_probability > self.log_probability_threshold
                and compression_ratio < self.compression_ratio_threshold
            ):
                single_timestamp_ending = (
                    len(tokens) >= 2
                    and tokens[-2] < tokenizer.timestamp_begin
                    and tokens[-1] >= tokenizer.timestamp_begin
                )

                consecutive_timestamps = [
                    i
                    for i in range(len(tokens))
                    if i > 0
                    and tokens[i] >= tokenizer.timestamp_begin
                    and tokens[i - 1] >= tokenizer.timestamp_begin
                ]

                if len(consecutive_timestamps) > 0:
                    slices = list(consecutive_timestamps)
                    if single_timestamp_ending:
                        slices.append(len(tokens))

                    last_slice = 0
                    for current_slice in slices:
                        sliced_tokens = tokens[last_slice:current_slice]
                        start_timestamp_position = (
                            sliced_tokens[0] - tokenizer.timestamp_begin
                        )
                        end_timestamp_position = (
                            sliced_tokens[-1] - tokenizer.timestamp_begin
                        )
                        start_time = time_offset + start_timestamp_position * 0.02
                        end_time = time_offset + end_timestamp_position * 0.02

                        current_segments.append(
                            dict(
                                start=start_time,
                                end=end_time,
                                tokens=sliced_tokens,
                            )
                        )
                        last_slice = current_slice
                else:
                    duration = segment_duration
                    timestamps = [
                        token for token in tokens if token >= tokenizer.timestamp_begin
                    ]
                    if (
                        len(timestamps) > 0
                        and timestamps[-1] != tokenizer.timestamp_begin
                    ):
                        last_timestamp_position = (
                            timestamps[-1] - tokenizer.timestamp_begin
                        )
                        duration = last_timestamp_position * 0.02

                    current_segments.append(
                        dict(
                            start=time_offset,
                            end=time_offset + duration,
                            tokens=tokens,
                        )
                    )

            # TODO: Implement word timestamps

            outputs.append(
                {
                    "segments": self._decode_batch(current_segments),
                    "need_fallback": len(current_segments) == 0,
                }
            )

        return outputs

    def _decode_batch(self, outputs: List[dict]) -> List[dict]:
        """
        Extract the token ids from the sequences ids and decode them using the tokenizer.

        Args:
            outputs (List[dict]): List of outputs from the model.

        Returns:
            List[str]: List of decoded texts.
        """
        if len(outputs) == 0:
            return outputs

        tokens_to_decode = [
            [token for token in out["tokens"] if token < self.tokenizer.eot]
            for out in outputs
        ]
        # TODO: We call the inherited tokenizer here, because faster_whisper tokenizer
        # doesn't have the decode_batch method. We should fix this in the future.
        decoded_tokens = self.tokenizer.tokenizer.decode_batch(tokens_to_decode)

        for out, text in zip(outputs, decoded_tokens):
            out["text"] = text

        return outputs

    def _get_quality_metrics(
        self, tokens: List[int], text: str, score: float, length_penalty: float
    ) -> Tuple[float, float]:
        """
        Get the compression ratio and the average log probability of the outputs to score them.

        Args:
            tokens (List[int]): List of token ids.
            text (str): Decoded text.
            score (float): Score of the sequence.
            length_penalty (float): Length penalty to apply to the average log probability.

        Returns:
            Tuple[float, float]: Compression ratio and average log probability.
        """
        text_bytes = text.encode("utf-8")
        compression_ratio = len(text_bytes) / len(zlib.compress(text_bytes))

        seq_len = len(tokens)
        cumulative_log_probability = score * (seq_len**length_penalty)
        average_log_probability = cumulative_log_probability / (seq_len + 1)

        return compression_ratio, average_log_probability

    def _collate_fn(
        self, items: List[Dict[str, Union[int, torch.Tensor, List[float]]]]
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Collator function for the dataloader.

        Args:
            items (List[Dict[str, Union[int, torch.Tensor, List[float]]]]): List of items to collate.

        Returns:
            Dict[str, Union[torch.Tensor, List]]: Collated items.
        """
        collated_items = {
            "indexes": [item["index"] for item in items],
            "features": torch.stack([item["feature"] for item in items]),
            "time_offsets": [item["time_offset"] for item in items],
            "segment_durations": [item["segment_duration"] for item in items],
        }

        return collated_items
