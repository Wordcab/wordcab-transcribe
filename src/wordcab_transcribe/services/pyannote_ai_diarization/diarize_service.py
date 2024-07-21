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
"""Pyannote AI diarization service for audio files."""

import time
import requests
import shortuuid
import subprocess
from loguru import logger
from typing import List, Tuple, Dict, Any

from wordcab_transcribe.models import DiarizationOutput
from wordcab_transcribe.services.pyannote_ai_diarization.shared_state import diarization_results


def get_current_ip():
    try:
        # Use an external service to get the public IP
        response = requests.get('https://api.ipify.org')
        ip = response.text
        return f"http://{ip}/webhook"
    except requests.RequestException:
        # Fallback to local IP if public IP can't be retrieved
        try:
            ip = subprocess.check_output(['hostname', '-I']).decode('utf-8').split()[0]
            return f"http://{ip}/webhook"
        except (subprocess.CalledProcessError, IndexError):
            raise RuntimeError("Could not determine IP address")


WEBHOOK_BASE_URL = get_current_ip()
logger.info(f"Webhook base URL: {WEBHOOK_BASE_URL}")

pending_webhook_jobs: Dict[str, Any] = {}


def start_diarization(
        api_key: str,
        diarize_url: str,
        audio_url: str,
        job_id: str,
        num_speakers: int,
        timeout: int = 600,
        check_interval: int = 5
):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    WEBHOOK_BASE_URL = "dz5h2rderk0d.share.zrok.io"
    data = {
        "url": audio_url,
        "webhook": f"https://{WEBHOOK_BASE_URL}/diarization-webhook/{job_id}",
    }
    response = requests.post(diarize_url, json=data, headers=headers)
    response.raise_for_status()
    data = response.json()

    start_time = time.time()
    while time.time() - start_time < timeout:
        if job_id in diarization_results:
            result = diarization_results.pop(job_id)
            return result
        time.sleep(check_interval)

    raise TimeoutError("Diarization webhook not received in time")


class PyannoteAIDiarizeService:
    """Pyannote AI Diarize Service for audio files."""

    def __init__(self, pyannote_ai_api_key: str, pyannote_ai_api_url: str):
        self.pyannote_ai_api_key = pyannote_ai_api_key
        self.pyannote_ai_api_url = pyannote_ai_api_url

    def __call__(
            self,
            oracle_num_speakers: int,
            url: str,
    ) -> DiarizationOutput:
        """
        Run inference with the Pyannote AI API.

        Args:
            oracle_num_speakers (int):
                Number of speakers in the audio file.
            url (str):
                URL of the audio file.

        Returns:
            DiarizationOutput:
                List of segments with the following keys: "start", "end", "speaker".
        """

        job_id = f"pyannote-ai-{shortuuid.ShortUUID().random(length=32)}"
        diarize_url = f"{self.pyannote_ai_api_url}/diarize"

        result = start_diarization(
            api_key=self.pyannote_ai_api_key,
            diarize_url=diarize_url,
            audio_url=url,
            job_id=job_id,
            num_speakers=oracle_num_speakers,
        )
        annotation = result["annotation"]

        segments = self.convert_annotation_to_segments(annotation)
        segments = self.get_contiguous_timestamps(segments)
        segments = self.merge_timestamps(segments)

        return DiarizationOutput(segments=segments)

    def convert_annotation_to_segments(
        self, annotation
    ) -> List[Tuple[float, float, int]]:
        """
        Convert annotation to segments.

        Args:
            annotation: Annotation object.

        Returns:
            List[Tuple[float, float, int]]: List of segments containing the start time, end time and speaker.
        """
        segments = []
        speaker_mapping = {}
        current_speaker_id = 0

        for segment in annotation:
            speaker_label = segment["speaker"]
            if speaker_label not in speaker_mapping:
                speaker_mapping[speaker_label] = current_speaker_id
                current_speaker_id += 1

            start = round(segment["start"], 4)
            end = round(segment["end"], 4)
            speaker = speaker_mapping[speaker_label]

            segments.append((start, end, speaker))

        return segments

    @staticmethod
    def get_contiguous_timestamps(
        stamps: List[Tuple[float, float, int]]
    ) -> List[Tuple[float, float, int]]:
        """
        Return contiguous timestamps.

        Args:
            stamps (List[Tuple[float, float, int]]): List of segments containing the start time, end time and speaker.

        Returns:
            List[Tuple[float, float, int]]: List of segments containing the start time, end time and speaker.
        """
        contiguous_timestamps = []
        for i in range(len(stamps) - 1):
            start, end, speaker = stamps[i]
            next_start, next_end, next_speaker = stamps[i + 1]

            if end > next_start:
                avg = (next_start + end) / 2.0
                stamps[i + 1] = (avg, next_end, next_speaker)
                contiguous_timestamps.append((start, avg, speaker))
            else:
                contiguous_timestamps.append((start, end, speaker))

        start, end, speaker = stamps[-1]
        contiguous_timestamps.append((start, end, speaker))

        return contiguous_timestamps

    @staticmethod
    def merge_timestamps(
        stamps: List[Tuple[float, float, int]]
    ) -> List[Tuple[float, float, int]]:
        """
        Merge timestamps of the same speaker.

        Args:
            stamps (List[Tuple[float, float, int]]): List of segments containing the start time, end time and speaker.

        Returns:
            List[Tuple[float, float, int]]: List of segments containing the start time, end time and speaker.
        """
        overlap_timestamps = []
        for i in range(len(stamps) - 1):
            start, end, speaker = stamps[i]
            next_start, next_end, next_speaker = stamps[i + 1]

            if end == next_start and speaker == next_speaker:
                stamps[i + 1] = (start, next_end, next_speaker)
            else:
                overlap_timestamps.append((start, end, speaker))

        start, end, speaker = stamps[-1]
        overlap_timestamps.append((start, end, speaker))

        return overlap_timestamps
