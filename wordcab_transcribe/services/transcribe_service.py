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

import itertools
import math
import os
import zlib
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F  # noqa N812
import torchaudio
from ctranslate2 import StorageView
from ctranslate2.models import WhisperGenerationResult
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_ctranslate2_storage
from torch.utils.data import DataLoader, IterableDataset

from wordcab_transcribe.logging import time_and_tell
from wordcab_transcribe.services.vad_service import VadService
from wordcab_transcribe.utils import check_number_of_segments, enhance_audio


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
        audio: Union[str, torch.Tensor, List[torch.Tensor]],
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
            audio (Union[str, torch.Tensor, List[torch.Tensor]]): Audio file, tensor, or list of tensors.
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

        elif isinstance(audio, list):
            if not all(isinstance(a, torch.Tensor) for a in audio):
                raise TypeError("Audio must be a list of tensors.")
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
        self, waveform: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        """
        Create 30-second chunks from the audio file loaded as a tensor.

        Args:
            waveform (Union[torch.Tensor, List[torch.Tensor]]): Audio file loaded as a tensor.

        Returns:
            Tuple[List[torch.Tensor], List[int], List[float]]: Tuple of audio chunks,
        """
        if isinstance(waveform, torch.Tensor):
            num_segments = math.ceil(waveform.size(0) / self.n_samples)
            segments = [
                waveform[i * self.n_samples : (i + 1) * self.n_samples]
                for i in range(num_segments)
            ]
        else:
            segments = waveform
            num_segments = len(segments)

        indexes = [i for i in range(num_segments)]
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
        self.tokens_per_second = self.sample_rate // self.hop_length

        assets_dir = Path(__file__).parent.parent / "assets" / "mel_filters.npz"
        with np.load(str(assets_dir)) as f:
            self.mel_filters = torch.from_numpy(f[f"mel_{self.n_mels}"])

        self.compression_ratio_threshold = 2.4
        self.log_probability_threshold = -0.8

        self.prepend_punctuation = "\"'“¿([{-"
        self.append_punctuation = "\"'.。,，!！?？:：”)]}、"

    def __call__(
        self,
        audio: Union[str, torch.Tensor, Tuple[str, str]],
        source_lang: str,
        suppress_blank: bool = False,
        word_timestamps: bool = True,
        vad_service: Optional[VadService] = None,
    ) -> Union[List[dict], List[List[dict]]]:
        """
        Run inference with the transcribe model.

        Args:
            audio (Union[str, torch.Tensor, Tuple[str, str]]): Audio file path or audio tensor. If a tuple is passed,
                the task is assumed to be a dual_channel task and the tuple should contain the paths to the two
                audio files.
            source_lang (str): Language of the audio file.
            suppress_blank (bool): Whether to suppress blank at the beginning of the sampling.
            word_timestamps (bool): Whether to return word timestamps.
            vad_service (Optional[VADService]): VADService to use for voice activity detection in the dual_channel case.

        Returns:
            Union[List[dict], List[List[dict]]]: List of transcriptions. If the task is a dual_channel task,
                a list of lists is returned.
        """
        if self.tokenizer.language_code != source_lang:
            self.tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task="transcribe",
                language=source_lang,
            )

        if isinstance(audio, tuple):
            outputs = []
            for audio_index, audio_file in enumerate(audio):
                outputs.append(
                    self._transcribe_dual_channel(
                        audio_file, audio_index, vad_service,
                    )
                )

        else:
            outputs = self.pipeline(audio, self._batch_size, suppress_blank, word_timestamps)

        return outputs

    @time_and_tell
    def pipeline(
        self,
        audio: Union[str, torch.Tensor],
        batch_size: int,
        suppress_blank: bool = True,
        word_timestamps: bool = False,
    ) -> List[dict]:
        """
        Transcription pipeline for audio chunks in batches.

        Args:
            audio (Union[str, torch.Tensor]): Audio file to transcribe.
            batch_size (int): Batch size to use for inference.
            suppress_blank (bool): Whether to suppress blank at the beginning of the sampling.
            word_timestamps (bool): Whether to return word timestamps.

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
                    suppress_blank=suppress_blank,
                    temperature=temperature,
                    last_chance_inference=False if stop_temperature != 1.0 else True,
                    word_timestamps=word_timestamps,
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

        outputs = list(itertools.chain.from_iterable(_outputs))

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
        last_chance_inference: bool = False,
        length_penalty: float = 1.0,
        patience: int = 1.0,
        prefix: Optional[str] = None,
        num_hypotheses: int = 1,
        sampling_top_k: int = 1,
        suppress_blank: bool = True,
        temperature: float = 1.0,
        without_timestamps: bool = False,
        word_timestamps: bool = False,
    ) -> List[dict]:
        """
        Use the ctranslate2 Whisper model to generate text from audio chunks.

        Args:
            features (torch.Tensor): List of audio chunks.
            time_offsets (List[float]): Time offsets for the audio chunks.
            segment_durations (List[float]): Durations of the audio chunks.
            tokenizer (Tokenizer): Tokenizer to use for encoding the text.
            beam_size (int): Beam size to use for beam search.
            last_chance_inference (bool): Whether to accept the result of the inference even if not perfect.
            length_penalty (float): Length penalty to use for beam search.
            initial_prompt (Optional[str]): Initial prompt to use for the generation.
            num_hypotheses (int): Number of hypotheses used by generate.
            patience (int): Patience to use for beam search.
            prefix (Optional[str]): Prefix to use for the generation.
            sampling_top_k (int): Sampling top k to use for sampling.
            suppress_blank (bool): Whether to suppress blank output of the sampling.
            temperature (float): Temperature to use for sampling.
            without_timestamps (bool): Whether to remove timestamps from the generated text.
            word_timestamps (bool): Whether to use word timestamps instead of character timestamps.

        Returns:
            List[dict]: List of segments with the following keys: "segments", "need_fallback".
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

        features = self._encode_batch(features, word_timestamps=word_timestamps)

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
            suppress_blank=suppress_blank,
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
            # Or if it is the last chance inference, we will accept the result even if not perfect.
            if (
                average_log_probability > self.log_probability_threshold
                and compression_ratio < self.compression_ratio_threshold
            ) or last_chance_inference:
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

            outputs.append(
                dict(
                    segments=self._decode_batch(current_segments, tokenizer),
                    need_fallback=len(current_segments) == 0,
                )
            )

        if word_timestamps:
            segment_sizes = [
                int(segment_duration / (self.hop_length / self.sample_rate))
                for segment_duration in segment_durations
            ]
            self._add_word_timestamps(
                outputs,
                tokenizer,
                features,
                segment_sizes,
                time_offsets,
                self.prepend_punctuation,
                self.append_punctuation,
            )

        return outputs

    def _encode_batch(
        self, features: torch.Tensor, word_timestamps: bool
    ) -> StorageView:
        """Encode the features using the model encoder.

        We encode the features only if word timestamps are enabled.
        Otherwise, we just return the features formatted as a StorageView.

        Args:
            features (torch.Tensor): Features to encode.
            word_timestamps (bool): Whether to encode the features or not.

        Returns:
            StorageView: Encoded features.
        """
        features = get_ctranslate2_storage(features)

        if (
            word_timestamps
        ):  # We encode the features to re-use the encoder output later.
            features = self.model.model.encode(features, to_cpu=False)

        return features

    def _decode_batch(self, outputs: List[dict], tokenizer: Tokenizer) -> List[dict]:
        """
        Extract the token ids from the sequences ids and decode them using the tokenizer.

        Args:
            outputs (List[dict]): List of outputs from the model.
            tokenizer (Tokenizer): Tokenizer to use to decode the token ids.

        Returns:
            List[str]: List of decoded texts.
        """
        if len(outputs) == 0:
            return outputs

        tokens_to_decode = [
            [token for token in out["tokens"] if token < tokenizer.eot]
            for out in outputs
        ]
        # TODO: We call the inherited tokenizer here, because faster_whisper tokenizer
        # doesn't have the decode_batch method. We should fix this in the future.
        decoded_tokens = tokenizer.tokenizer.decode_batch(tokens_to_decode)

        for out, text in zip(outputs, decoded_tokens):
            out["text"] = text

        return outputs

    def _transcribe_dual_channel(
        self,
        audio: Union[str, torch.Tensor],
        speaker_id: int,
        vad_service: VadService,
    ) -> List[dict]:
        """
        Transcribe an audio file with two channels.

        Args:
            audio (Union[str, torch.Tensor]): Audio file path or loaded audio.
            speaker_id (int): Speaker ID used in the diarization.
            vad_service (VadService): VAD service.

        Returns:
            List[dict]: List of transcribed segments.
        """
        enhanced_audio = enhance_audio(audio, apply_agc=True, apply_bandpass=False)
        grouped_segments, audio = vad_service(enhanced_audio)

        final_transcript = []
        silence_padding = torch.from_numpy(np.zeros(int(3 * self.sample_rate))).float()

        for group in grouped_segments:
            audio_segments = []
            for segment in group:
                audio_segments.extend([
                    audio[segment["start"]: segment["end"]], silence_padding
                ])

            segments = self.pipeline(torch.cat(audio_segments), self._batch_size, False, True)

            group_start = group[0]["start"]
            for segment in segments:
                segment_dict = {
                    "start": None,
                    "end": None,
                    "text": segment["text"],
                    "words": [],
                    "speaker": speaker_id,
                }

                for word in segment["words"]:
                    word_start_adjusted = (
                        group_start / self.sample_rate
                    ) + word["start"]
                    word_end_adjusted = (group_start / self.sample_rate) + word["end"]
                    segment_dict["words"].append(
                        {
                            "start": word_start_adjusted,
                            "end": word_end_adjusted,
                            "word": word["word"],
                        }
                    )

                    if (
                        segment_dict["start"] is None
                        or word_start_adjusted < segment_dict["start"]
                    ):
                        segment_dict["start"] = word_start_adjusted
            
                    if (
                        segment_dict["end"] is None
                        or word_end_adjusted > segment_dict["end"]
                    ):
                        segment_dict["end"] = word_end_adjusted

                final_transcript.append(segment_dict)

        return final_transcript

    @time_and_tell
    def _add_word_timestamps(
        self,
        outputs: List[dict],
        tokenizer: Tokenizer,
        encoder_output: StorageView,
        segment_sizes: List[int],
        time_offsets: List[float],
        prepend_punctuation: str,
        append_punctuation: str,
    ) -> None:
        """
        Add word timestamps to the segments.

        Args:
            outputs (List[dict]): List of outputs from the model.
            tokenizer (Tokenizer): Tokenizer to use to decode the token ids.
            encoder_output (StorageView): Encoder output.
            segment_sizes (List[int]): List of segment sizes.
            time_offsets (List[float]): List of time offsets.
            prepend_punctuation (str): Punctuation to prepend to the text.
            append_punctuation (str): Punctuation to append to the text.
        """
        text_tokens_per_output = []
        for out in outputs:
            text_tokens_per_segment = [
                [token for token in segment["tokens"] if token < tokenizer.eot]
                for segment in out["segments"]
            ]
            text_tokens_per_output.append(text_tokens_per_segment)

        alignments = self._find_alignment(
            encoder_output, text_tokens_per_output, tokenizer, segment_sizes
        )
        self._merge_punctuation(alignments, prepend_punctuation, append_punctuation)

        for out, alignment, text_tokens_per_segment, time_offset in zip(
            outputs, alignments, text_tokens_per_output, time_offsets
        ):
            if out["need_fallback"]:
                continue

            word_index = 0

            for segment_idx, segment in enumerate(out["segments"]):
                saved_tokens = 0
                words = []

                if isinstance(alignment, int):
                    alignment = [alignment]

                while word_index < len(alignment) and saved_tokens < len(
                    text_tokens_per_segment[segment_idx]
                ):
                    timing = alignment[word_index]

                    if timing["word"]:
                        words.append(
                            dict(
                                word=timing["word"],
                                start=round(time_offset + timing["start"], 2),
                                end=round(time_offset + timing["end"], 2),
                                probability=timing["probability"],
                            )
                        )

                    saved_tokens += len(timing["tokens"])
                    word_index += 1

                if len(words) > 0:
                    segment["start"] = words[0]["start"]
                    segment["end"] = words[-1]["end"]

                segment["words"] = words

    def _find_alignment(
        self,
        encoder_output: StorageView,
        text_tokens_per_output: List[List[int]],
        tokenizer: Tokenizer,
        segment_sizes: List[int],
        median_filter_width: int = 7,
    ) -> List[List[dict]]:
        """
        Find the alignment between the encoder output and the text tokens in a batch.

        Args:
            encoder_output (StorageView): Encoder output.
            text_tokens_per_output (List[List[int]]): List of text tokens per output.
            tokenizer (Tokenizer): Tokenizer to use to decode the token ids.
            segment_sizes (List[int]): List of segment sizes.
            median_filter_width (int): Width of the median filter to apply on the alignment.

        Returns:
            List[List[dict]]: List of alignments per output.
        """
        text_tokens_per_output = [
            list(itertools.chain.from_iterable(list_of_tokens))
            for list_of_tokens in text_tokens_per_output
        ]

        results = self.model.model.align(
            encoder_output,
            tokenizer.sot_sequence,
            text_tokens_per_output,
            segment_sizes,
            median_filter_width=median_filter_width,
        )

        final_alignments = []
        for res, text_tokens in zip(results, text_tokens_per_output):
            words, word_tokens = tokenizer.split_to_word_tokens(
                text_tokens + [tokenizer.eot]
            )
            word_boundaries = np.pad(
                np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0)
            )
            if len(word_boundaries) <= 1:
                final_alignments.append([])
                continue

            alignments = res.alignments
            text_indices = np.array([pair[0] for pair in alignments])
            time_indices = np.array([pair[1] for pair in alignments])

            jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(
                bool
            )
            jump_times = time_indices[jumps] / self.tokens_per_second
            start_times = jump_times[word_boundaries[:-1]]
            end_times = jump_times[word_boundaries[1:]]

            text_token_probs = res.text_token_probs
            word_probabilities = [
                np.mean(text_token_probs[i:j])
                for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
            ]

            word_durations = end_times - start_times
            word_durations = word_durations[word_durations.nonzero()]

            if len(word_durations) > 0:
                median_duration = np.median(word_durations)
                max_duration = median_duration * 2

                if len(word_durations) >= 2 and word_durations[1] > max_duration:
                    boundary = max(end_times[2] / 2, end_times[2] - max_duration)
                    end_times[0] = start_times[1] = boundary

                if (
                    len(word_durations) >= 1
                    and end_times[0] - start_times[0] > max_duration
                ):
                    start_times[0] = max(0, end_times[0] - max_duration)

            final_alignments.append(
                [
                    dict(
                        word=word,
                        tokens=tokens,
                        start=start,
                        end=end,
                        probability=probability,
                    )
                    for word, tokens, start, end, probability in zip(
                        words, word_tokens, start_times, end_times, word_probabilities
                    )
                ]
            )

        return final_alignments

    def _merge_punctuation(
        self, alignments: List[List[dict]], prepended: str, appended: str
    ) -> None:
        """
        Fix punctuation boundaries for the alignments.

        Args:
            alignments (List[List[dict]]): List of alignments.
            prepended (str): Prepended punctuation.
            appended (str): Appended punctuation.
        """
        for alignment in alignments:
            # merge prepended punctuations
            i = len(alignment) - 2
            j = len(alignment) - 1
            while i >= 0:
                previous = alignment[i]
                following = alignment[j]
                if (
                    previous["word"].startswith(" ")
                    and previous["word"].strip() in prepended
                ):
                    # prepend it to the following word
                    following["word"] = previous["word"] + following["word"]
                    following["tokens"] = previous["tokens"] + following["tokens"]
                    previous["word"] = ""
                    previous["tokens"] = []
                else:
                    j = i
                i -= 1

            # merge appended punctuations
            i = 0
            j = 1
            while j < len(alignment):
                previous = alignment[i]
                following = alignment[j]
                if not previous["word"].endswith(" ") and following["word"] in appended:
                    # append it to the previous word
                    previous["word"] = previous["word"] + following["word"]
                    previous["tokens"] = previous["tokens"] + following["tokens"]
                    following["word"] = ""
                    following["tokens"] = []
                else:
                    i = j
                j += 1

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
