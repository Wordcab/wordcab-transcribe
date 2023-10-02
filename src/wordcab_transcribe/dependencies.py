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
"""Dependencies for Wordcab Transcribe."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from wordcab_transcribe.config import settings
from wordcab_transcribe.services.asr_service import (
    ASRAsyncService,
    ASRLiveService,
    ASRTranscriptionOnly,
)
from wordcab_transcribe.utils import (
    check_ffmpeg,
    download_model,
    retrieve_user_platform,
)

# Define the maximum number of files to pre-download for the async ASR service
download_limit = asyncio.Semaphore(10)

# Define the ASR service to use depending on the settings
if settings.asr_type == "live":
    asr = ASRLiveService(
        whisper_model=settings.whisper_model,
        compute_type=settings.compute_type,
        debug_mode=settings.debug,
    )
elif settings.asr_type == "async":
    asr = ASRAsyncService(
        whisper_model=settings.whisper_model,
        compute_type=settings.compute_type,
        window_lengths=settings.window_lengths,
        shift_lengths=settings.shift_lengths,
        multiscale_weights=settings.multiscale_weights,
        extra_languages=settings.extra_languages,
        extra_languages_model_paths=settings.extra_languages_model_paths,
        transcribe_server_urls=settings.transcribe_server_urls,
        diarize_server_urls=settings.diarize_server_urls,
        debug_mode=settings.debug,
    )
elif settings.asr_type == "only_transcription":
    asr = ASRTranscriptionOnly(
        whisper_model=settings.whisper_model,
        compute_type=settings.compute_type,
        extra_languages=settings.extra_languages,
        extra_languages_model_paths=settings.extra_languages_model_paths,
        debug_mode=settings.debug,
    )
elif settings.asr_type == "only_diarization":
    asr = None
else:
    raise ValueError(f"Invalid ASR type: {settings.asr_type}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Context manager to handle the startup and shutdown of the application."""
    if retrieve_user_platform() != "linux":
        logger.warning(
            "You are not running the application on Linux.\nThe application was tested"
            " on Ubuntu 22.04, so we cannot guarantee that it will work on other"
            " OS.\nReport any issues with your env specs to:"
            " https://github.com/Wordcab/wordcab-transcribe/issues"
        )

    if settings.asr_type == "async" or settings.asr_type == "only_transcription":
        if check_ffmpeg() is False:
            logger.warning(
                "FFmpeg is not installed on the host machine.\n"
                "Please install it and try again: `sudo apt-get install ffmpeg`"
            )
            exit(1)

        if settings.extra_languages is not None:
            logger.info("Downloading models for extra languages...")
            for model in settings.extra_languages:
                try:
                    model_path = download_model(
                        compute_type=settings.compute_type, language=model
                    )

                    if model_path is not None:
                        settings.extra_languages_model_paths[model] = model_path
                    else:
                        raise Exception(f"Coudn't download model for {model}")

                except Exception as e:
                    logger.error(f"Error downloading model for {model}: {e}")

    logger.info("Warmup initialization...")
    await asr.inference_warmup()

    yield  # This is where the execution of the application starts
