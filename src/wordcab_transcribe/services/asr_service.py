# Copyright 2023 The Wordcab Team. All rights reserved.
#
# Licensed under the Wordcab Transcribe License 0.1 (the "License");
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
"""ASR Service module that handle all AI interactions."""

import asyncio
import time
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import aiohttp
import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict
from tensorshare import Backend, TensorShare

from wordcab_transcribe.logging import time_and_tell, time_and_tell_async
from wordcab_transcribe.models import (
    DiarizationOutput,
    DiarizationRequest,
    ProcessTimes,
    Timestamps,
    TranscribeRequest,
    TranscriptionOutput,
    Utterance,
)
from wordcab_transcribe.services.concurrency_services import GPUService, URLService
from wordcab_transcribe.services.diarization.diarize_service import DiarizeService
from wordcab_transcribe.services.post_processing_service import PostProcessingService
from wordcab_transcribe.services.transcribe_service import TranscribeService
from wordcab_transcribe.services.vad_service import VadService
from wordcab_transcribe.utils import early_return, format_segments, read_audio


class ExceptionSource(str, Enum):
    """Exception source enum."""

    diarization = "diarization"
    post_processing = "post_processing"
    transcription = "transcription"


class ProcessException(BaseModel):
    """Process exception model."""

    source: ExceptionSource
    message: str


class LocalExecution(BaseModel):
    """Local execution model."""

    index: Union[int, None]


class RemoteExecution(BaseModel):
    """Remote execution model."""

    url: str


class ASRTask(BaseModel):
    """ASR Task model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio: Union[torch.Tensor, List[torch.Tensor]]
    diarization: "DiarizationTask"
    duration: float
    multi_channel: bool
    offset_start: Union[float, None]
    post_processing: "PostProcessingTask"
    process_times: ProcessTimes
    timestamps_format: Timestamps
    transcription: "TranscriptionTask"
    word_timestamps: bool


class DiarizationTask(BaseModel):
    """Diarization Task model."""

    execution: Union[LocalExecution, RemoteExecution, None]
    num_speakers: int
    result: Union[ProcessException, DiarizationOutput, None] = None


class PostProcessingTask(BaseModel):
    """Post Processing Task model."""

    result: Union[ProcessException, List[Utterance], None] = None


class TranscriptionOptions(BaseModel):
    """Transcription options model."""

    compression_ratio_threshold: float
    condition_on_previous_text: bool
    internal_vad: bool
    log_prob_threshold: float
    no_speech_threshold: float
    repetition_penalty: float
    source_lang: str
    vocab: Union[List[str], None]


class TranscriptionTask(BaseModel):
    """Transcription Task model."""

    execution: Union[LocalExecution, RemoteExecution]
    options: TranscriptionOptions
    result: Union[
        ProcessException, TranscriptionOutput, List[TranscriptionOutput], None
    ] = None


class ASRService(ABC):
    """Base ASR Service module that handle all AI interactions and batch processing."""

    def __init__(self) -> None:
        """Initialize the ASR Service.

        This class is not meant to be instantiated. Use the subclasses instead.
        """
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Do we have a GPU? If so, use it!
        self.num_gpus = torch.cuda.device_count() if self.device == "cuda" else 0
        logger.info(f"NVIDIA GPUs available: {self.num_gpus}")

        if self.num_gpus > 1 and self.device == "cuda":
            self.device_index = list(range(self.num_gpus))
        else:
            self.device_index = [0]

        self.gpu_handler = GPUService(
            device=self.device, device_index=self.device_index
        )

    @abstractmethod
    async def process_input(self) -> None:
        """Process the input request by creating a task and adding it to the appropriate queues."""
        raise NotImplementedError("This method should be implemented in subclasses.")


class ASRAsyncService(ASRService):
    """ASR Service module for async endpoints."""

    def __init__(
        self,
        whisper_model: str,
        compute_type: str,
        window_lengths: List[int],
        shift_lengths: List[int],
        multiscale_weights: List[float],
        extra_languages: Union[List[str], None],
        extra_languages_model_paths: Union[List[str], None],
        transcribe_server_urls: Union[List[str], None],
        diarize_server_urls: Union[List[str], None],
        debug_mode: bool,
    ) -> None:
        """
        Initialize the ASRAsyncService class.

        Args:
            whisper_model (str):
                The path to the whisper model.
            compute_type (str):
                The compute type to use for inference.
            window_lengths (List[int]):
                The window lengths to use for diarization.
            shift_lengths (List[int]):
                The shift lengths to use for diarization.
            multiscale_weights (List[float]):
                The multiscale weights to use for diarization.
            extra_languages (Union[List[str], None]):
                The list of extra languages to support.
            extra_languages_model_paths (Union[List[str], None]):
                The list of paths to the extra language models.
            use_remote_servers (bool):
                Whether to use remote servers for transcription and diarization.
            transcribe_server_urls (Union[List[str], None]):
                The list of URLs to the remote transcription servers.
            diarize_server_urls (Union[List[str], None]):
                The list of URLs to the remote diarization servers.
            debug_mode (bool):
                Whether to run in debug mode.
        """
        super().__init__()

        self.services: dict = {
            "post_processing": PostProcessingService(),
            "vad": VadService(),
        }
        self.dual_channel_transcribe_options: dict = {
            "beam_size": 5,
            "patience": 1,
            "length_penalty": 1,
            "suppress_blank": False,
            "word_timestamps": True,
            "temperature": 0.0,
        }

        if transcribe_server_urls is not None:
            self.use_remote_transcription = True
            self.transcription_url_handler = URLService(
                remote_urls=transcribe_server_urls
            )
        else:
            self.use_remote_transcription = False
            self.services["transcription"] = TranscribeService(
                model_path=whisper_model,
                compute_type=compute_type,
                device=self.device,
                device_index=self.device_index,
                extra_languages=extra_languages,
                extra_languages_model_paths=extra_languages_model_paths,
            )

        if diarize_server_urls is not None:
            self.use_remote_diarization = True
            self.diarization_url_handler = URLService(remote_urls=diarize_server_urls)
        else:
            self.use_remote_diarization = False
            self.services["diarization"] = DiarizeService(
                device=self.device,
                device_index=self.device_index,
                window_lengths=window_lengths,
                shift_lengths=shift_lengths,
                multiscale_weights=multiscale_weights,
            )

        self.debug_mode = debug_mode

    async def inference_warmup(self) -> None:
        """Warmup the GPU by loading the models."""
        sample_path = Path(__file__).parent.parent / "assets/warmup_sample.wav"

        for gpu_index in self.gpu_handler.device_index:
            logger.info(f"Warmup GPU {gpu_index}.")
            await self.process_input(
                filepath=str(sample_path),
                offset_start=None,
                offset_end=None,
                num_speakers=1,
                diarization=True,
                multi_channel=False,
                source_lang="en",
                timestamps_format="s",
                vocab=None,
                word_timestamps=False,
                internal_vad=False,
                repetition_penalty=1.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
            )

    async def process_input(  # noqa: C901
        self,
        filepath: Union[str, List[str]],
        offset_start: Union[float, None],
        offset_end: Union[float, None],
        num_speakers: int,
        diarization: bool,
        multi_channel: bool,
        source_lang: str,
        timestamps_format: str,
        vocab: Union[List[str], None],
        word_timestamps: bool,
        internal_vad: bool,
        repetition_penalty: float,
        compression_ratio_threshold: float,
        log_prob_threshold: float,
        no_speech_threshold: float,
        condition_on_previous_text: bool,
    ) -> Union[Tuple[List[dict], ProcessTimes, float], Exception]:
        """Process the input request and return the results.

        This method will create a task and add it to the appropriate queues.
        All tasks are added to the transcription queue, but will be added to the
        diarization queues only if the user requested it.
        Each step will be processed asynchronously and the results will be returned
        and stored in separated keys in the task dictionary.

        Args:
            filepath (Union[str, List[str]]):
                Path to the audio file or list of paths to the audio files to process.
            offset_start (Union[float, None]):
                The start time of the audio file to process.
            offset_end (Union[float, None]):
                The end time of the audio file to process.
            num_speakers (int):
                The number of oracle speakers.
            diarization (bool):
                Whether to do diarization or not.
            multi_channel (bool):
                Whether to do multi-channel diarization or not.
            source_lang (str):
                Source language of the audio file.
            timestamps_format (str):
                Timestamps format to use.
            vocab (Union[List[str], None]):
                List of words to use for the vocabulary.
            word_timestamps (bool):
                Whether to return word timestamps or not.
            internal_vad (bool):
                Whether to use faster-whisper's VAD or not.
            repetition_penalty (float):
                The repetition penalty to use for the beam search.
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
            Union[Tuple[List[dict], ProcessTimes, float], Exception]:
                The results of the ASR pipeline or an exception if something went wrong.
                Results are returned as a tuple of the following:
                    * List[dict]: The final results of the ASR pipeline.
                    * ProcessTimes: The process times of each step of the ASR pipeline.
                    * float: The audio duration
        """
        if isinstance(filepath, list):
            audio, durations = [], []
            for path in filepath:
                _audio, _duration = read_audio(
                    path, offset_start=offset_start, offset_end=offset_end
                )

                audio.append(_audio)
                durations.append(_duration)

            duration = sum(durations) / len(durations)

        else:
            audio, duration = read_audio(
                filepath, offset_start=offset_start, offset_end=offset_end
            )

        gpu_index = None
        if self.use_remote_transcription:
            _url = await self.transcription_url_handler.next_url()
            transcription_execution = RemoteExecution(url=_url)
        else:
            gpu_index = await self.gpu_handler.get_device()
            transcription_execution = LocalExecution(index=gpu_index)

        if diarization and multi_channel is False:
            if self.use_remote_diarization:
                _url = await self.diarization_url_handler.next_url()
                diarization_execution = RemoteExecution(url=_url)
            else:
                if gpu_index is None:
                    gpu_index = await self.gpu_handler.get_device()

                diarization_execution = LocalExecution(index=gpu_index)
        else:
            diarization_execution = None

        task = ASRTask(
            audio=audio,
            diarization=DiarizationTask(
                execution=diarization_execution, num_speakers=num_speakers
            ),
            duration=duration,
            multi_channel=multi_channel,
            offset_start=offset_start,
            post_processing=PostProcessingTask(),
            process_times=ProcessTimes(),
            timestamps_format=timestamps_format,
            transcription=TranscriptionTask(
                execution=transcription_execution,
                options=TranscriptionOptions(
                    compression_ratio_threshold=compression_ratio_threshold,
                    condition_on_previous_text=condition_on_previous_text,
                    internal_vad=internal_vad,
                    log_prob_threshold=log_prob_threshold,
                    no_speech_threshold=no_speech_threshold,
                    repetition_penalty=repetition_penalty,
                    source_lang=source_lang,
                    vocab=vocab,
                ),
            ),
            word_timestamps=word_timestamps,
        )

        try:
            start_process_time = time.time()

            transcription_task = self.process_transcription(task, self.debug_mode)
            diarization_task = self.process_diarization(task, self.debug_mode)

            await asyncio.gather(transcription_task, diarization_task)

            if isinstance(task.diarization.result, ProcessException):
                return task.diarization.result

            if (
                diarization
                and task.diarization.result is None
                and multi_channel is False
            ):
                # Empty audio early return
                return early_return(duration=duration)

            if isinstance(task.transcription.result, ProcessException):
                return task.transcription.result

            await asyncio.get_event_loop().run_in_executor(
                None,
                self.process_post_processing,
                task,
            )

            if isinstance(task.post_processing.result, ProcessException):
                return task.post_processing.result

            task.process_times.total = time.time() - start_process_time

            return task.post_processing.result, task.process_times, duration

        except Exception as e:
            return e

        finally:
            del task

            if gpu_index is not None:
                self.gpu_handler.release_device(gpu_index)

    async def process_transcription(self, task: ASRTask, debug_mode: bool) -> None:
        """
        Process a task of transcription and update the task with the result.

        Args:
            task (ASRTask): The task and its parameters.
            debug_mode (bool): Whether to run in debug mode or not.

        Returns:
            None: The task is updated with the result.
        """
        try:
            if isinstance(task.transcription.execution, LocalExecution):
                out = await time_and_tell_async(
                    lambda: self.services["transcription"](
                        task.audio,
                        model_index=task.transcription.execution.index,
                        suppress_blank=False,
                        word_timestamps=True,
                        **task.transcription.options.model_dump(),
                    ),
                    func_name="transcription",
                    debug_mode=debug_mode,
                )
                result, process_time = out

            elif isinstance(task.transcription.execution, RemoteExecution):
                if isinstance(task.audio, list):
                    ts = [
                        TensorShare.from_dict({"audio": a}, backend=Backend.TORCH)
                        for a in task.audio
                    ]
                else:
                    ts = TensorShare.from_dict(
                        {"audio": task.audio}, backend=Backend.TORCH
                    )

                data = TranscribeRequest(
                    audio=ts,
                    **task.transcription.options.model_dump(),
                )
                out = await time_and_tell_async(
                    self.remote_transcription(
                        url=task.transcription.execution.url,
                        data=data,
                    ),
                    func_name="transcription",
                    debug_mode=debug_mode,
                )
                result, process_time = out

            else:
                raise NotImplementedError("No execution method specified.")

        except Exception as e:
            result = ProcessException(
                source=ExceptionSource.transcription,
                message=f"Error in transcription: {e}\n{traceback.format_exc()}",
            )
            process_time = None

        finally:
            task.process_times.transcription = process_time
            task.transcription.result = result

        return None

    async def process_diarization(self, task: ASRTask, debug_mode: bool) -> None:
        """
        Process a task of diarization.

        Args:
            task (ASRTask): The task and its parameters.
            debug_mode (bool): Whether to run in debug mode or not.

        Returns:
            None: The task is updated with the result.
        """
        try:
            if isinstance(task.diarization.execution, LocalExecution):
                out = await time_and_tell_async(
                    lambda: self.services["diarization"](
                        waveform=task.audio,
                        audio_duration=task.duration,
                        oracle_num_speakers=task.diarization.num_speakers,
                        model_index=task.diarization.execution.index,
                        vad_service=self.services["vad"],
                    ),
                    func_name="diarization",
                    debug_mode=debug_mode,
                )
                result, process_time = out

            elif isinstance(task.diarization.execution, RemoteExecution):
                ts = TensorShare.from_dict({"audio": task.audio}, backend=Backend.TORCH)

                data = DiarizationRequest(
                    audio=ts,
                    duration=task.duration,
                    num_speakers=task.diarization.num_speakers,
                )
                out = await time_and_tell_async(
                    self.remote_diarization(
                        url=task.diarization.execution.url,
                        data=data,
                    ),
                    func_name="diarization",
                    debug_mode=debug_mode,
                )
                result, process_time = out

            elif task.diarization.execution is None:
                result = None
                process_time = None

            else:
                raise NotImplementedError("No execution method specified.")

        except Exception as e:
            result = ProcessException(
                source=ExceptionSource.diarization,
                message=f"Error in diarization: {e}\n{traceback.format_exc()}",
            )
            process_time = None

        finally:
            task.process_times.diarization = process_time
            task.diarization.result = result

        return None

    def process_post_processing(self, task: ASRTask) -> None:
        """
        Process a task of post-processing.

        Args:
            task (ASRTask): The task and its parameters.

        Returns:
            None: The task is updated with the result.
        """
        try:
            total_post_process_time = 0

            if task.multi_channel:
                utterances, process_time = time_and_tell(
                    self.services["post_processing"].multi_channel_speaker_mapping(
                        task.transcription.result
                    ),
                    func_name="multi_channel_speaker_mapping",
                    debug_mode=self.debug_mode,
                )
                total_post_process_time += process_time

            else:
                formatted_segments, process_time = time_and_tell(
                    format_segments(
                        transcription_output=task.transcription.result,
                    ),
                    func_name="format_segments",
                    debug_mode=self.debug_mode,
                )
                total_post_process_time += process_time

                if task.diarization.execution is not None:
                    utterances, process_time = time_and_tell(
                        self.services["post_processing"].single_channel_speaker_mapping(
                            transcript_segments=formatted_segments,
                            speaker_timestamps=task.diarization.result,
                            word_timestamps=task.word_timestamps,
                        ),
                        func_name="single_channel_speaker_mapping",
                        debug_mode=self.debug_mode,
                    )
                    total_post_process_time += process_time
                else:
                    utterances = formatted_segments

            final_utterances, process_time = time_and_tell(
                self.services["post_processing"].final_processing_before_returning(
                    utterances=utterances,
                    offset_start=task.offset_start,
                    timestamps_format=task.timestamps_format,
                    word_timestamps=task.word_timestamps,
                ),
                func_name="final_processing_before_returning",
                debug_mode=self.debug_mode,
            )
            total_post_process_time += process_time

        except Exception as e:
            final_utterances = ProcessException(
                source=ExceptionSource.post_processing,
                message=f"Error in post-processing: {e}\n{traceback.format_exc()}",
            )
            total_post_process_time = None

        finally:
            task.process_times.post_processing = total_post_process_time
            task.post_processing.result = final_utterances

        return None

    async def remote_transcription(
        self,
        url: str,
        data: TranscribeRequest,
    ) -> TranscriptionOutput:
        """Remote transcription method."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f"{url}/api/v1/transcribe",
                data=data.model_dump_json(),
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    raise Exception(response.status)
                else:
                    return TranscriptionOutput(**await response.json())

    async def remote_diarization(
        self,
        url: str,
        data: DiarizationRequest,
    ) -> DiarizationOutput:
        """Remote diarization method."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f"{url}/api/v1/diarize",
                data=data.model_dump_json(),
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    r = await response.json()
                    raise Exception(r["detail"])
                else:
                    return DiarizationOutput(**await response.json())


class ASRLiveService(ASRService):
    """ASR Service module for live endpoints."""

    def __init__(self, whisper_model: str, compute_type: str, debug_mode: bool) -> None:
        """Initialize the ASRLiveService class."""
        super().__init__()

        self.transcription_service = TranscribeService(
            model_path=whisper_model,
            compute_type=compute_type,
            device=self.device,
            device_index=self.device_index,
        )
        self.debug_mode = debug_mode

    async def inference_warmup(self) -> None:
        """Warmup the GPU by loading the models."""
        sample_audio = Path(__file__).parent.parent / "assets/warmup_sample.wav"
        with open(sample_audio, "rb") as audio_file:
            async for _ in self.process_input(
                data=audio_file.read(),
                source_lang="en",
            ):
                pass

    async def process_input(self, data: bytes, source_lang: str) -> Iterable[dict]:
        """
        Process the input data and return the results as a tuple of text and duration.

        Args:
            data (bytes):
                The raw audio bytes to process.
            source_lang (str):
                The source language of the audio data.

        Yields:
            Iterable[dict]: The results of the ASR pipeline.
        """
        gpu_index = await self.gpu_handler.get_device()

        try:
            waveform, _ = read_audio(data)

            async for result in self.transcription_service.async_live_transcribe(
                audio=waveform, source_lang=source_lang, model_index=gpu_index
            ):
                yield result

        except Exception as e:
            logger.error(
                f"Error in transcription gpu {gpu_index}: {e}\n{traceback.format_exc()}"
            )

        finally:
            self.gpu_handler.release_device(gpu_index)


class ASRTranscriptionOnly(ASRService):
    """ASR Service module for transcription-only endpoint."""

    def __init__(
        self,
        whisper_model: str,
        compute_type: str,
        extra_languages: Union[List[str], None],
        extra_languages_model_paths: Union[List[str], None],
        debug_mode: bool,
    ) -> None:
        """Initialize the ASRTranscriptionOnly class."""
        super().__init__()

        self.transcription_service = TranscribeService(
            model_path=whisper_model,
            compute_type=compute_type,
            device=self.device,
            device_index=self.device_index,
            extra_languages=extra_languages,
            extra_languages_model_paths=extra_languages_model_paths,
        )
        self.debug_mode = debug_mode

    async def inference_warmup(self) -> None:
        """Warmup the GPU by doing one inference."""
        sample_audio = Path(__file__).parent.parent / "assets/warmup_sample.wav"

        audio, _ = read_audio(str(sample_audio))
        ts = TensorShare.from_dict({"audio": audio}, backend=Backend.TORCH)

        data = TranscribeRequest(
            audio=ts,
            source_lang="en",
            compression_ratio_threshold=2.4,
            condition_on_previous_text=True,
            internal_vad=False,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            repetition_penalty=1.0,
            vocab=None,
        )

        for gpu_index in self.gpu_handler.device_index:
            logger.info(f"Warmup GPU {gpu_index}.")
            await self.process_input(data=data)

    async def process_input(
        self, data: TranscribeRequest
    ) -> Union[TranscriptionOutput, List[TranscriptionOutput]]:
        """
        Process the input data and return the results as a list of segments.

        Args:
            data (TranscribeRequest):
                The input data to process.

        Returns:
            Union[TranscriptionOutput, List[TranscriptionOutput]]:
                The results of the ASR pipeline.
        """
        gpu_index = await self.gpu_handler.get_device()

        try:
            result = self.transcription_service(
                audio=data.audio,
                source_lang=data.source_lang,
                model_index=gpu_index,
                suppress_blank=False,
                word_timestamps=True,
                compression_ratio_threshold=data.compression_ratio_threshold,
                condition_on_previous_text=data.condition_on_previous_text,
                internal_vad=data.internal_vad,
                log_prob_threshold=data.log_prob_threshold,
                repetition_penalty=data.repetition_penalty,
                no_speech_threshold=data.no_speech_threshold,
                vocab=data.vocab,
            )

        except Exception as e:
            result = ProcessException(
                source=ExceptionSource.transcription,
                message=f"Error in transcription: {e}\n{traceback.format_exc()}",
            )

        finally:
            self.gpu_handler.release_device(gpu_index)

        return result


class ASRDiarizationOnly(ASRService):
    """ASR Service module for diarization-only endpoint."""

    def __init__(
        self,
        window_lengths: List[int],
        shift_lengths: List[int],
        multiscale_weights: List[float],
        debug_mode: bool,
    ) -> None:
        """Initialize the ASRDiarizationOnly class."""
        super().__init__()

        self.diarization_service = DiarizeService(
            device=self.device,
            device_index=self.device_index,
            window_lengths=window_lengths,
            shift_lengths=shift_lengths,
            multiscale_weights=multiscale_weights,
        )
        self.vad_service = VadService()
        self.debug_mode = debug_mode

    async def inference_warmup(self) -> None:
        """Warmup the GPU by doing one inference."""
        sample_audio = Path(__file__).parent.parent / "assets/warmup_sample.wav"

        audio, duration = read_audio(str(sample_audio))
        ts = TensorShare.from_dict({"audio": audio}, backend=Backend.TORCH)

        data = DiarizationRequest(
            audio=ts,
            duration=duration,
            num_speakers=1,
        )

        for gpu_index in self.gpu_handler.device_index:
            logger.info(f"Warmup GPU {gpu_index}.")
            await self.process_input(data=data)

    async def process_input(self, data: DiarizationRequest) -> DiarizationOutput:
        """
        Process the input data and return the results as a list of segments.

        Args:
            data (DiarizationRequest):
                The input data to process.

        Returns:
            DiarizationOutput:
                The results of the ASR pipeline.
        """
        gpu_index = await self.gpu_handler.get_device()

        try:
            result = self.diarization_service(
                waveform=data.audio,
                audio_duration=data.duration,
                oracle_num_speakers=data.num_speakers,
                model_index=gpu_index,
                vad_service=self.vad_service,
            )

        except Exception as e:
            result = ProcessException(
                source=ExceptionSource.diarization,
                message=f"Error in diarization: {e}\n{traceback.format_exc()}",
            )

        finally:
            self.gpu_handler.release_device(gpu_index)

        return result
