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
"""Audio url endpoint for the Wordcab Transcribe API."""

import asyncio
import json
from datetime import datetime
from typing import List, Optional, Union

import boto3
import shortuuid
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi import status as http_status
from loguru import logger
from svix.api import MessageIn, SvixAsync

from wordcab_transcribe.config import settings
from wordcab_transcribe.dependencies import asr, download_limit
from wordcab_transcribe.models import (
    AudioRequest,
    AudioResponse,
)
from wordcab_transcribe.utils import (
    check_num_channels,
    delete_file,
    download_audio_file,
    process_audio_file,
)

router = APIRouter()


def retrieve_service(service, aws_creds):
    return boto3.client(
        service,
        aws_access_key_id=aws_creds.get("aws_access_key_id"),
        aws_secret_access_key=aws_creds.get("aws_secret_access_key"),
        region_name=aws_creds.get("region_name"),
    )


s3_client = retrieve_service(
    "s3",
    {
        "aws_access_key_id": settings.aws_access_key_id,
        "aws_secret_access_key": settings.aws_secret_access_key,
        "region_name": settings.aws_region_name,
    },
)


@router.post("", status_code=http_status.HTTP_202_ACCEPTED)
async def inference_with_audio_url(
    background_tasks: BackgroundTasks,
    url: str,
    data: Optional[AudioRequest] = None,
) -> dict:
    """Inference endpoint with audio url."""
    filename = f"audio_url_{shortuuid.ShortUUID().random(length=32)}"
    data = AudioRequest() if data is None else AudioRequest(**data.dict())

    async def process_audio():
        try:
            async with download_limit:
                _filepath = await download_audio_file("url", url, filename)

                num_channels = await check_num_channels(_filepath)
                if num_channels > 1 and data.multi_channel is False:
                    num_channels = 1  # Force mono channel if more than 1 channel

                try:
                    filepath: Union[str, List[str]] = await process_audio_file(
                        _filepath, num_channels=num_channels
                    )

                except Exception as e:
                    raise HTTPException(  # noqa: B904
                        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Process failed: {e}",
                    )

                background_tasks.add_task(delete_file, filepath=filename)

                task = asyncio.create_task(
                    asr.process_input(
                        filepath=filepath,
                        url=url,
                        url_type="url",
                        offset_start=data.offset_start,
                        offset_end=data.offset_end,
                        num_speakers=data.num_speakers,
                        diarization=data.diarization,
                        batch_size=data.batch_size,
                        multi_channel=data.multi_channel,
                        source_lang=data.source_lang,
                        timestamps_format=data.timestamps,
                        vocab=data.vocab,
                        word_timestamps=data.word_timestamps,
                        internal_vad=data.internal_vad,
                        repetition_penalty=data.repetition_penalty,
                        compression_ratio_threshold=data.compression_ratio_threshold,
                        log_prob_threshold=data.log_prob_threshold,
                        no_speech_threshold=data.no_speech_threshold,
                        condition_on_previous_text=data.condition_on_previous_text,
                    )
                )

                result = await task
                utterances, process_times, audio_duration = result
                result = AudioResponse(
                    utterances=utterances,
                    audio_duration=audio_duration,
                    offset_start=data.offset_start,
                    offset_end=data.offset_end,
                    num_speakers=data.num_speakers,
                    diarization=data.diarization,
                    batch_size=data.batch_size,
                    multi_channel=data.multi_channel,
                    source_lang=data.source_lang,
                    timestamps=data.timestamps,
                    vocab=data.vocab,
                    word_timestamps=data.word_timestamps,
                    internal_vad=data.internal_vad,
                    repetition_penalty=data.repetition_penalty,
                    compression_ratio_threshold=data.compression_ratio_threshold,
                    log_prob_threshold=data.log_prob_threshold,
                    no_speech_threshold=data.no_speech_threshold,
                    condition_on_previous_text=data.condition_on_previous_text,
                    job_name=data.job_name,
                    task_token=data.task_token,
                    process_times=process_times,
                )

                upload_file(
                    s3_client,
                    file=bytes(json.dumps(result.model_dump()).encode("UTF-8")),
                    bucket=settings.aws_storage_bucket_name,
                    object_name=f"responses/{data.task_token}_{data.job_name}.json",
                )

                background_tasks.add_task(delete_file, filepath=filepath)
                await send_update_with_svix(
                    data.job_name,
                    "finished",
                    {
                        "job_name": data.job_name,
                        "task_token": data.task_token,
                    },
                )
        except Exception as e:
            error_message = f"Error during transcription: {e}"
            logger.error(error_message)

            error_payload = {
                "error": error_message,
                "job_name": data.job_name,
                "task_token": data.task_token,
            }

            await send_update_with_svix(data.job_name, "error", error_payload)

    # Add the process_audio function to background tasks
    background_tasks.add_task(process_audio)

    # Return the job name and task token immediately
    return {"job_name": data.job_name, "task_token": data.task_token}


def upload_file(s3_client, file, bucket, object_name):
    try:
        s3_client.put_object(
            Body=file,
            Bucket=bucket,
            Key=object_name,
        )
    except Exception as e:
        logger.error(f"Exception while uploading results to S3: {e}")
        return False
    return True


async def send_update_with_svix(
    job_name: str,
    status: str,
    payload: dict,
    payload_retention_period: Optional[int] = 5,
) -> None:
    """
    Send the status update to Svix.

    Args:
        job_name (str): The name of the job.
        status (str): The status of the job.
        payload (dict): The payload to send.
        payload_retention_period (Optional[int], optional): The payload retention period. Defaults to 5.
    """
    if settings.svix_api_key and settings.svix_app_id:
        svix = SvixAsync(settings.svix_api_key)
        await svix.message.create(
            settings.svix_app_id,
            MessageIn(
                event_type=f"async_job.wordcab_transcribe.{status}",
                event_id=f"wordcab_transcribe_{status}_{job_name}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}",
                payload_retention_period=payload_retention_period,
                payload=payload,
            ),
        )
    else:
        logger.warning(
            "Svix API key and app ID are not set. Cannot send the status update to"
            " Svix."
        )
