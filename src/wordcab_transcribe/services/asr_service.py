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
import functools
import os
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from loguru import logger

from wordcab_transcribe.logging import time_and_tell
from wordcab_transcribe.services.diarization.diarize_service import DiarizeService
from wordcab_transcribe.services.gpu_service import GPUService
from wordcab_transcribe.services.post_processing_service import PostProcessingService
from wordcab_transcribe.services.transcribe_service import TranscribeService
from wordcab_transcribe.services.vad_service import VadService
from wordcab_transcribe.utils import early_return, format_segments, read_audio


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
        self.num_cpus = os.cpu_count()

        self.sample_rate = (
            16000  # The sample rate to use for inference for all audio files (Hz)
        )

        self.queues = None  # the queue to store requests
        self.queue_locks = None  # the locks to access the queues
        self.needs_processing = (
            None  # the flag to indicate if the queue needs processing
        )
        self.needs_processing_timer = None  # the timer to schedule processing

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
        extra_languages: List[str],
        extra_languages_model_paths: List[str],
        debug_mode: bool,
    ) -> None:
        """
        Initialize the ASRAsyncService class.

        Args:
            whisper_model (str): The path to the whisper model.
            compute_type (str): The compute type to use for inference.
            window_lengths (List[int]): The window lengths to use for diarization.
            shift_lengths (List[int]): The shift lengths to use for diarization.
            multiscale_weights (List[float]): The multiscale weights to use for diarization.
            extra_languages (List[str]): The list of extra languages to support.
            extra_languages_model_paths (List[str]): The list of paths to the extra language models.
            debug_mode (bool): Whether to run in debug mode.
        """
        super().__init__()

        self.services: dict = {
            "transcription": TranscribeService(
                model_path=whisper_model,
                compute_type=compute_type,
                device=self.device,
                device_index=self.device_index,
                extra_languages=extra_languages,
                extra_languages_model_paths=extra_languages_model_paths,
            ),
            "diarization": DiarizeService(
                device=self.device,
                device_index=self.device_index,
                window_lengths=window_lengths,
                shift_lengths=shift_lengths,
                multiscale_weights=multiscale_weights,
            ),
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
                dual_channel=False,
                source_lang="en",
                timestamps_format="s",
                vocab=[],
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
        filepath: Union[str, Tuple[str, str]],
        offset_start: Union[float, None],
        offset_end: Union[float, None],
        num_speakers: int,
        diarization: bool,
        dual_channel: bool,
        source_lang: str,
        timestamps_format: str,
        vocab: List[str],
        word_timestamps: bool,
        internal_vad: bool,
        repetition_penalty: float,
        compression_ratio_threshold: float,
        log_prob_threshold: float,
        no_speech_threshold: float,
        condition_on_previous_text: bool,
    ) -> Union[Tuple[List[dict], Dict[str, float], float], Exception]:
        """Process the input request and return the results.

        This method will create a task and add it to the appropriate queues.
        All tasks are added to the transcription queue, but will be added to the
        diarization queues only if the user requested it.
        Each step will be processed asynchronously and the results will be returned
        and stored in separated keys in the task dictionary.

        Args:
            filepath (Union[str, Tuple[str, str]]):
                Path to the audio file or tuple of paths to the audio files.
            offset_start (Union[float, None]):
                The start time of the audio file to process.
            offset_end (Union[float, None]):
                The end time of the audio file to process.
            num_speakers (int):
                The number of oracle speakers.
            diarization (bool):
                Whether to do diarization or not.
            dual_channel (bool):
                Whether to do dual channel or not.
            source_lang (str):
                Source language of the audio file.
            timestamps_format (str):
                Timestamps format to use.
            vocab (List[str]):
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
            Union[Tuple[List[dict], Dict[str, float], float], Exception]:
                The results of the ASR pipeline or an exception if something went wrong.
                Results are returned as a tuple of the following:
                    * List[dict]: The final results of the ASR pipeline.
                    * Dict[str, float]: the process times for each step.
                    * float: The audio duration
        """
        if isinstance(filepath, tuple):
            audio, duration = [], []
            for path in filepath:
                _audio, _duration = read_audio(
                    path, offset_start=offset_start, offset_end=offset_end
                )

                audio.append(_audio)
                duration.append(_duration)

            audio = tuple(audio)
            duration = sum(duration) / len(duration)

        else:
            audio, duration = read_audio(
                filepath, offset_start=offset_start, offset_end=offset_end
            )

        task = {
            "input": audio,
            "offset_start": offset_start,
            "duration": duration,
            "num_speakers": num_speakers,
            "diarization": diarization,
            "dual_channel": dual_channel,
            "source_lang": source_lang,
            "timestamps_format": timestamps_format,
            "vocab": vocab,
            "word_timestamps": word_timestamps,
            "internal_vad": internal_vad,
            "repetition_penalty": repetition_penalty,
            "compression_ratio_threshold": compression_ratio_threshold,
            "log_prob_threshold": log_prob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "condition_on_previous_text": condition_on_previous_text,
            "transcription_result": None,
            "transcription_done": asyncio.Event(),
            "diarization_result": None,
            "diarization_done": asyncio.Event(),
            "post_processing_result": None,
            "post_processing_done": asyncio.Event(),
            "process_times": {},
        }

        gpu_index = -1  # Placeholder to track if GPU is acquired

        try:
            # If GPU is required, acquire one
            if self.device == "cuda":
                gpu_index = await self.gpu_handler.get_device()
                logger.info(f"Using GPU {gpu_index} for the task")

            start_process_time = time.time()

            asyncio.get_event_loop().run_in_executor(
                None,
                functools.partial(
                    self.process_transcription, task, gpu_index, self.debug_mode
                ),
            )

            if diarization and dual_channel is False:
                asyncio.get_event_loop().run_in_executor(
                    None,
                    functools.partial(
                        self.process_diarization, task, gpu_index, self.debug_mode
                    ),
                )
            else:
                task["process_times"]["diarization"] = None
                task["diarization_done"].set()

            await task["transcription_done"].wait()
            await task["diarization_done"].wait()

            if isinstance(task["diarization_result"], Exception):
                self.gpu_handler.release_device(gpu_index)
                gpu_index = -1
                return task["diarization_result"]

            if (
                diarization
                and task["diarization_result"] is None
                and dual_channel is False
            ):
                # Empty audio early return
                return early_return(duration=duration)

            if isinstance(task["transcription_result"], Exception):
                self.gpu_handler.release_device(gpu_index)
                gpu_index = -1
                return task["transcription_result"]

            self.gpu_handler.release_device(gpu_index)
            gpu_index = -1

            asyncio.get_event_loop().run_in_executor(
                None, functools.partial(self.process_post_processing, task)
            )

            await task["post_processing_done"].wait()

            if isinstance(task["post_processing_result"], Exception):
                return task["post_processing_result"]

            result: List[dict] = task.pop("post_processing_result")
            process_times: Dict[str, float] = task.pop("process_times")
            process_times["total"]: float = time.time() - start_process_time

            del task  # Delete the task to free up memory

            return result, process_times, duration
        except Exception as e:
            return e
        finally:
            # Ensure GPU is released if it was acquired
            if gpu_index != -1:
                self.gpu_handler.release_device(gpu_index)

    def process_transcription(
        self, task: dict, gpu_index: int, debug_mode: bool
    ) -> None:
        """
        Process a task of transcription and update the task with the result.

        Args:
            task (dict): The task and its parameters.
            gpu_index (int): The GPU index to use for the transcription.
            debug_mode (bool): Whether to run in debug mode or not.

        Returns:
            None: The task is updated with the result.
        """
        try:
            result, process_time = time_and_tell(
                lambda: self.services["transcription"](
                    task["input"],
                    source_lang=task["source_lang"],
                    model_index=gpu_index,
                    suppress_blank=False,
                    vocab=None if task["vocab"] == [] else task["vocab"],
                    word_timestamps=True,
                    internal_vad=task["internal_vad"],
                    repetition_penalty=task["repetition_penalty"],
                    compression_ratio_threshold=task["compression_ratio_threshold"],
                    log_prob_threshold=task["log_prob_threshold"],
                    no_speech_threshold=task["no_speech_threshold"],
                    condition_on_previous_text=task["condition_on_previous_text"],
                    vad_service=self.services["vad"] if task["dual_channel"] else None,
                ),
                func_name="transcription",
                debug_mode=debug_mode,
            )

        except Exception as e:
            result = Exception(
                f"Error in transcription gpu {gpu_index}: {e}\n{traceback.format_exc()}"
            )
            process_time = None

        finally:
            task["process_times"]["transcription"] = process_time
            task["transcription_result"] = result
            task["transcription_done"].set()

        return None

    def process_diarization(self, task: dict, gpu_index: int, debug_mode: bool) -> None:
        """
        Process a task of diarization.

        Args:
            task (dict): The task and its parameters.
            gpu_index (int): The GPU index to use for the diarization.
            debug_mode (bool): Whether to run in debug mode or not.

        Returns:
            None: The task is updated with the result.
        """
        try:
            result, process_time = time_and_tell(
                lambda: self.services["diarization"](
                    task["input"],
                    audio_duration=task["duration"],
                    oracle_num_speakers=task["num_speakers"],
                    model_index=gpu_index,
                    vad_service=self.services["vad"],
                ),
                func_name="diarization",
                debug_mode=debug_mode,
            )

        except Exception as e:
            result = Exception(f"Error in diarization: {e}\n{traceback.format_exc()}")
            process_time = None

        finally:
            task["process_times"]["diarization"] = process_time
            task["diarization_result"] = result
            task["diarization_done"].set()

        return None

    def process_post_processing(self, task: dict) -> None:
        """
        Process a task of post-processing.

        Args:
            task (dict): The task and its parameters.

        Returns:
            None: The task is updated with the result.
        """
        try:
            total_post_process_time = 0
            diarization = task["diarization"]
            dual_channel = task["dual_channel"]
            word_timestamps = task["word_timestamps"]

            if dual_channel:
                left_segments, right_segments = task["transcription_result"]
                utterances, process_time = time_and_tell(
                    lambda: self.services[
                        "post_processing"
                    ].dual_channel_speaker_mapping(
                        left_segments=left_segments,
                        right_segments=right_segments,
                    ),
                    func_name="dual_channel_speaker_mapping",
                    debug_mode=self.debug_mode,
                )
                total_post_process_time += process_time

            else:
                formatted_segments, process_time = time_and_tell(
                    lambda: format_segments(
                        segments=task["transcription_result"],
                        word_timestamps=True,
                    ),
                    func_name="format_segments",
                    debug_mode=self.debug_mode,
                )
                total_post_process_time += process_time

                if diarization:
                    utterances, process_time = time_and_tell(
                        lambda: self.services[
                            "post_processing"
                        ].single_channel_speaker_mapping(
                            transcript_segments=formatted_segments,
                            speaker_timestamps=task["diarization_result"],
                            word_timestamps=word_timestamps,
                        ),
                        func_name="single_channel_speaker_mapping",
                        debug_mode=self.debug_mode,
                    )
                    total_post_process_time += process_time
                else:
                    utterances = formatted_segments

            final_utterances, process_time = time_and_tell(
                lambda: self.services[
                    "post_processing"
                ].final_processing_before_returning(
                    utterances=utterances,
                    diarization=diarization,
                    dual_channel=task["dual_channel"],
                    offset_start=task["offset_start"],
                    timestamps_format=task["timestamps_format"],
                    word_timestamps=word_timestamps,
                ),
                func_name="final_processing_before_returning",
                debug_mode=self.debug_mode,
            )
            total_post_process_time += process_time

        except Exception as e:
            final_utterances = Exception(
                f"Error in post-processing: {e}\n{traceback.format_exc()}"
            )
            total_post_process_time = None

        finally:
            task["process_times"]["post_processing"] = total_post_process_time
            task["post_processing_result"] = final_utterances
            task["post_processing_done"].set()

        return None


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
            await self.process_input(
                data=audio_file.read(),
                source_lang="en",
            )

    async def process_input(self, data: bytes, source_lang: str) -> Tuple[str, float]:
        """
        Process the input data and return the results as a tuple of text and duration.

        Args:
            data (bytes):
                The raw audio bytes to process.
            source_lang (str):
                The source language of the audio data.

        Returns:
            Tuple[str, float]: The text and duration of the audio data.
        """
        gpu_index = await self.gpu_handler.get_device()

        try:
            waveform, duration = read_audio(data)

            result, process_time = time_and_tell(
                lambda: self.transcription_service(
                    waveform,
                    source_lang=source_lang,
                    model_index=gpu_index,
                    suppress_blank=False,
                    vocab=None,
                    word_timestamps=False,
                    internal_vad=False,
                ),
                func_name="live_transcription",
                debug_mode=self.debug_mode,
            )

        except Exception as e:
            result = None
            duration = None
            logger.error(
                f"Error in transcription gpu {gpu_index}: {e}\n{traceback.format_exc()}"
            )

        finally:
            logger.info(f"Operation took {process_time} seconds")
            self.gpu_handler.release_device(gpu_index)

        return result, duration
