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
"""Transcribe Service for audio files."""
from typing import Iterable, List, NamedTuple, Optional, Union

import torch
from faster_whisper import WhisperModel
from loguru import logger
from tensorshare import Backend, TensorShare

from wordcab_transcribe.config import settings
from wordcab_transcribe.engines.tensorrt_llm.model import WhisperModelTRT
from wordcab_transcribe.models import (
    MultiChannelSegment,
    MultiChannelTranscriptionOutput,
    Segment,
    TranscriptionOutput,
    Word,
)


class FasterWhisperModel(NamedTuple):
    """Faster Whisper Model."""

    model: WhisperModel
    lang: str


class TranscribeService:
    """Transcribe Service for audio files."""

    def __init__(
        self,
        model_path: str,
        model_engine: str,
        compute_type: str,
        device: str,
        device_index: Union[int, List[int]],
        extra_languages: Union[List[str], None] = None,
        extra_languages_model_paths: Union[List[str], None] = None,
    ) -> None:
        """Initialize the Transcribe Service.

        This service uses the WhisperModel from faster-whisper to transcribe audio files.

        Args:
            model_path (str):
                Path to the model checkpoint. This can be a local path or a URL.
            compute_type (str):
                Compute type to use for inference. Can be "int8", "int8_float16", "int16" or "float_16".
            device (str):
                Device to use for inference. Can be "cpu" or "cuda".
            device_index (Union[int, List[int]]):
                Index of the device to use for inference.
            extra_languages (Union[List[str], None]):
                List of extra languages to transcribe. Defaults to None.
            extra_languages_model_paths (Union[List[str], None]):
                List of paths to the extra language models. Defaults to None.
        """
        self.device = device
        self.compute_type = compute_type
        self.model_path = model_path
        self.model_engine = model_engine

        if self.model_engine == "faster-whisper":
            logger.info("Using faster-whisper model engine.")
            self.model = WhisperModel(
                self.model_path,
                device=self.device,
                device_index=device_index,
                compute_type=self.compute_type,
            )
        elif self.model_engine == "tensorrt-llm":
            logger.info("Using tensorrt-llm model engine.")
            if "distil" in self.model_path:
                n_mels = 128
            else:
                n_mels = 80
            self.model = WhisperModelTRT(
                self.model_path,
                device=self.device,
                device_index=device_index,
                compute_type=self.compute_type,
                asr_options={
                    "word_align_model": settings.align_model,
                },
                n_mels=n_mels,
            )
        else:
            self.model = WhisperModel(
                self.model_path,
                device=self.device,
                device_index=device_index,
                compute_type=self.compute_type,
            )

        self.extra_lang = extra_languages
        self.extra_lang_models = extra_languages_model_paths

    def __call__(
        self,
        audio: Union[
            str,
            torch.Tensor,
            TensorShare,
            List[str],
            List[torch.Tensor],
            List[TensorShare],
        ],
        source_lang: str,
        model_index: int,
        batch_size: int = 1,
        num_beams: int = 1,
        suppress_blank: bool = False,
        vocab: Union[List[str], None] = None,
        word_timestamps: bool = True,
        internal_vad: bool = False,
        repetition_penalty: float = 1.0,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = True,
    ) -> Union[TranscriptionOutput, List[TranscriptionOutput]]:
        """
        Run inference with the transcribe model.

        Args:
            audio (Union[str, torch.Tensor, TensorShare, List[str], List[torch.Tensor], List[TensorShare]]):
                Audio file path or audio tensor. If a tuple is passed, the task is assumed
                to be a multi_channel task and the list of audio files or tensors is passed.
            source_lang (str):
                Language of the audio file.
            model_index (int):
                Index of the model to use.
            batch_size (int):
                Batch size to use during generation.
            num_beams (int):
                Number of beams to use during generation.
            suppress_blank (bool):
                Whether to suppress blank at the beginning of the sampling.
            vocab (Union[List[str], None]):
                Vocabulary to use during generation if not None. Defaults to None.
            word_timestamps (bool):
                Whether to return word timestamps.
            internal_vad (bool):
                Whether to use faster-whisper's VAD or not.
            repetition_penalty (float):
                Repetition penalty to use during generation beamed search.
            compression_ratio_threshold (float):
                If the gzip compression ratio is above this value, treat as failed.
            log_prob_threshold (float):
                If the average log probability over sampled tokens is below this value, treat as failed.
            no_speech_threshold (float):
                If the no_speech probability is higher than this value AND the average log probability
                over sampled tokens is below `log_prob_threshold`, consider the segment as silent.
            condition_on_previous_text (bool):
                If True, the previous output of the model is provided as a prompt for the next window;
                disabling may make the text inconsistent across windows, but the model becomes less prone
                to getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

        Returns:
            Union[TranscriptionOutput, List[TranscriptionOutput]]:
                Transcription output. If the task is a multi_channel task, a list of TranscriptionOutput is returned.
        """

        if (
            vocab is not None
            and isinstance(vocab, list)
            and len(vocab) > 0
            and vocab[0].strip()
        ):
            words = ", ".join(vocab)
            prompt = f"Vocab: {words.strip()}."
        else:
            prompt = None

        if not isinstance(audio, list):
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            elif isinstance(audio, TensorShare):
                ts = audio.to_tensors(backend=Backend.NUMPY)
                audio = ts["audio"]

            if self.model_engine == "faster-whisper":
                segments, _ = self.model.transcribe(
                    audio,
                    language=source_lang,
                    initial_prompt=prompt,
                    repetition_penalty=repetition_penalty,
                    compression_ratio_threshold=compression_ratio_threshold,
                    log_prob_threshold=log_prob_threshold,
                    no_speech_threshold=no_speech_threshold,
                    condition_on_previous_text=condition_on_previous_text,
                    suppress_blank=suppress_blank,
                    word_timestamps=word_timestamps,
                    vad_filter=internal_vad,
                    vad_parameters={
                        "threshold": 0.5,
                        "min_speech_duration_ms": 250,
                        "min_silence_duration_ms": 100,
                        "speech_pad_ms": 30,
                        "window_size_samples": 512,
                    },
                )
            elif self.model_engine == "tensorrt-llm":
                segments = self.model.transcribe(
                    audio_data=[audio],
                    lang_codes=[source_lang],
                    tasks=["transcribe"],
                    initial_prompts=[prompt],
                    batch_size=batch_size,
                    use_vad=internal_vad,
                    generate_kwargs={"num_beams": num_beams},
                )[0]
                #  TODO: make batch compatible

                for ix, segment in enumerate(segments):
                    segment["words"] = segment.pop("word_timestamps")
                    segment["start"] = segment.pop("start_time")
                    segment["end"] = segment.pop("end_time")
                    extra = {
                        "seek": 1,
                        "id": 1,
                        "tokens": [1],
                        "temperature": 0.0,
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": 0.0,
                    }
                    segments[ix] = Segment(**{**segment, **extra})

            _outputs = [segment._asdict() for segment in segments]
            outputs = TranscriptionOutput(segments=_outputs)

        else:
            outputs = []
            for audio_index, audio_file in enumerate(audio):
                outputs.append(
                    self.multi_channel(
                        audio_file,
                        source_lang=source_lang,
                        speaker_id=audio_index,
                        suppress_blank=suppress_blank,
                        word_timestamps=word_timestamps,
                        internal_vad=internal_vad,
                        repetition_penalty=repetition_penalty,
                        compression_ratio_threshold=compression_ratio_threshold,
                        log_prob_threshold=log_prob_threshold,
                        no_speech_threshold=no_speech_threshold,
                        prompt=prompt,
                    )
                )

        return outputs

    async def async_live_transcribe(
        self,
        audio: torch.Tensor,
        source_lang: str,
        model_index: int,
    ) -> Iterable[dict]:
        """Async generator for live transcriptions.

        This method wraps the live_transcribe method to make it async.

        Args:
            audio (torch.Tensor): Audio tensor.
            source_lang (str): Language of the audio file.
            model_index (int): Index of the model to use.

        Yields:
            Iterable[dict]: Iterable of transcribed segments.
        """
        for result in self.live_transcribe(audio, source_lang, model_index):
            yield result

    def live_transcribe(
        self,
        audio: torch.Tensor,
        source_lang: str,
        model_index: int,
    ) -> Iterable[dict]:
        """
        Transcribe audio from a WebSocket connection.

        Args:
            audio (torch.Tensor): Audio tensor.
            source_lang (str): Language of the audio file.
            model_index (int): Index of the model to use.

        Yields:
            Iterable[dict]: Iterable of transcribed segments.
        """
        segments, _ = self.model.transcribe(
            audio.numpy(),
            language=source_lang,
            suppress_blank=True,
            word_timestamps=False,
        )

        for segment in segments:
            yield segment._asdict()

    def multi_channel(
        self,
        audio: Union[str, torch.Tensor, TensorShare],
        source_lang: str,
        speaker_id: int,
        suppress_blank: bool = False,
        word_timestamps: bool = True,
        internal_vad: bool = True,
        repetition_penalty: float = 1.0,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = False,
        prompt: Optional[str] = None,
    ) -> MultiChannelTranscriptionOutput:
        """
        Transcribe an audio file using the faster-whisper original pipeline.

        Args:
            audio (Union[str, torch.Tensor, TensorShare]): Audio file path or loaded audio.
            source_lang (str): Language of the audio file.
            speaker_id (int): Speaker ID used in the diarization.
            suppress_blank (bool):
                Whether to suppress blank at the beginning of the sampling.
            word_timestamps (bool):
                Whether to return word timestamps.
            internal_vad (bool):
                Whether to use faster-whisper's VAD or not.
            repetition_penalty (float):
                Repetition penalty to use during generation beamed search.
            compression_ratio_threshold (float):
                If the gzip compression ratio is above this value, treat as failed.
            log_prob_threshold (float):
                If the average log probability over sampled tokens is below this value, treat as failed.
            no_speech_threshold (float):
                If the no_speech probability is higher than this value AND the average log probability
                over sampled tokens is below `log_prob_threshold`, consider the segment as silent.
            condition_on_previous_text (bool):
                If True, the previous output of the model is provided as a prompt for the next window;
                disabling may make the text inconsistent across windows, but the model becomes less prone
                to getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.
            prompt (Optional[str]): Initial prompt to use for the generation.

        Returns:
            MultiChannelTranscriptionOutput: Multi-channel transcription segments in a list.
        """
        if isinstance(audio, torch.Tensor):
            _audio = audio.numpy()
        elif isinstance(audio, TensorShare):
            ts = audio.to_tensors(backend=Backend.NUMPY)
            _audio = ts["audio"]

        final_segments = []

        segments, _ = self.model.transcribe(
            _audio,
            language=source_lang,
            initial_prompt=prompt,
            repetition_penalty=repetition_penalty,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            suppress_blank=suppress_blank,
            word_timestamps=word_timestamps,
            vad_filter=internal_vad,
            vad_parameters={
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 100,
                "speech_pad_ms": 30,
                "window_size_samples": 512,
            },
        )

        for segment in segments:
            _segment = MultiChannelSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=[Word(**word._asdict()) for word in segment.words],
                speaker=speaker_id,
            )
            final_segments.append(_segment)

        return MultiChannelTranscriptionOutput(segments=final_segments)
