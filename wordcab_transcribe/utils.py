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
"""Utils module of the Wordcab Transcribe."""
import asyncio
import json
import math
import mimetypes
import re
import subprocess  # noqa: S404
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp
from loguru import logger
from omegaconf import OmegaConf
from yt_dlp import YoutubeDL


async def run_subprocess(command: List[str]) -> tuple:
    """
    Run a subprocess asynchronously.

    Args:
        command (List[str]): Command to run.

    Returns:
        tuple: Tuple with the return code, stdout and stderr.
    """
    process = await asyncio.create_subprocess_exec(
        *command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    return process.returncode, stdout, stderr


def convert_seconds_to_hms(seconds: float) -> str:
    """
    Convert seconds to hours, minutes and seconds.

    Args:
        seconds (float): Seconds to convert.

    Returns:
        str: Hours, minutes and seconds.
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = math.floor((seconds % 1) * 1000)

    output = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"

    return output


async def convert_file_to_wav(filepath: str) -> str:
    """
    Convert a file to wav format using ffmpeg.

    Args:
        filepath (str): Path to the file to convert.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If there is an error converting the file.

    Returns:
        str: Path to the converted file.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")

    new_filepath = filepath.with_suffix(".wav")
    cmd = [
        "ffmpeg",
        "-i",
        str(filepath),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        str(new_filepath),
    ]
    result = await run_subprocess(cmd)

    if result[0] != 0:
        raise Exception(f"Error converting file {filepath} to wav format: {result[2]}")

    return str(new_filepath)


async def download_file_from_youtube(url: str, filename: str) -> str:
    """
    Download a file from YouTube using youtube-dl.

    Args:
        url (str): URL of the YouTube video.
        filename (str): Filename to save the file as.

    Returns:
        str: Path to the downloaded file.
    """
    with YoutubeDL(
        {
            "format": "bestaudio",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            "outtmpl": f"{filename}",
        }
    ) as ydl:
        ydl.download([url])

    return filename + ".wav"


async def download_audio_file(
    url: str, filename: str, url_headers: Optional[Dict[str, str]] = None
) -> str:
    """
    Download an audio file from a URL.

    Args:
        url (str): URL of the audio file.
        filename (str): Filename to save the file as.
        url_headers (Optional[Dict[str, str]]): Headers to send with the request. Defaults to None.

    Raises:
        Exception: If the file failed to download.

    Returns:
        str: Path to the downloaded file.
    """
    url_headers = url_headers or {}

    logger.info(f"Downloading audio file from {url} to {filename}...")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=url_headers) as response:
            if response.status == 200:
                content_type = response.headers.get("Content-Type")
                extension = mimetypes.guess_extension(content_type)

                filename = f"{filename}{extension}"

                logger.info(f"New file name: {filename}")
                async with aiofiles.open(filename, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024)

                        if not chunk:
                            break

                        await f.write(chunk)
            else:
                raise Exception(f"Failed to download file. Status: {response.status}")

    return filename


def delete_file(filepath: str) -> None:
    """
    Delete a file.

    Args:
        filepath (str): Path to the file to delete.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if filepath.exists():
        filepath.unlink()


def is_empty_string(text: str):
    """
    Checks if a string is empty after removing spaces and periods.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the string is empty, False otherwise.
    """
    text = text.replace(".", "")
    text = re.sub(r"\s+", "", text)
    if text.strip():
        return False
    return True


def format_punct(text: str):
    """
    Removes Whisper's '...' output, and checks for weird spacing in punctuation. Also removes extra spaces.

    Args:
        text (str): The text to format.

    Returns:
        str: The formatted text.
    """
    text = text.replace("...", "")
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")
    text = re.sub("/s+", " ", text)
    return text.strip()


def format_segments(
    segments: list,
    use_dict: Optional[bool] = False,
    include_words: Optional[bool] = False,
) -> List[dict]:
    """
    Format the segments to a list of dicts with start, end and text keys.

    Args:
        segments (list): List of segments.
        use_dict (bool, optional): Use dict instead of object. Defaults to False.
        include_words (bool, optional): Include words. Defaults to False.

    Returns:
        list: List of dicts with start, end and text keys.
    """
    formatted_segments = []

    for segment in segments:
        segment_dict = {}

        if use_dict:
            segment_dict["start"] = segment["start"]
            segment_dict["end"] = segment["end"]
            segment_dict["text"] = segment["text"].strip()

        else:
            segment_dict["start"] = segment.start
            segment_dict["end"] = segment.end
            segment_dict["text"] = segment.text.strip()

        if include_words:
            words = [
                {
                    "start": word.start,
                    "end": word.end,
                    "word": word.word.strip(),
                    "probability": word.probability,
                }
                for word in segment.words
            ]
            segment_dict["words"] = words

        formatted_segments.append(segment_dict)

    return formatted_segments


def get_segment_timestamp_anchor(start: float, end: float, option: str = "start"):
    if option == "end":
        return end
    elif option == "mid":
        return (start + end) / 2
    return start


def load_nemo_config(domain_type: str, storage_path: str, output_path: str) -> Dict[str, Any]:
    """
    Load NeMo config file based on a domain type.

    Args:
        domain_type (str): The domain type. Can be "general", "meeting" or "telephonic".
        storage_path (str): The path to the NeMo storage directory.
        output_path (str): The path to the NeMo output directory.

    Returns:
        Dict[str, Any]: The config file as a dict.
    """
    cfg_path = (
        Path(__file__).parent.parent
        / "config"
        / "nemo"
        / f"diar_infer_{domain_type}.yaml"
    )
    with open(cfg_path) as f:
        cfg = OmegaConf.load(f)

    storage_path = Path(__file__).parent.parent / storage_path
    if not storage_path.exists():
        storage_path.mkdir(parents=True, exist_ok=True)

    meta = {
        "audio_filepath": "/app/temp_outputs/mono_file.wav",
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }

    manifest_path = storage_path / "infer_manifest.json"
    with open(manifest_path, "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    output_path = Path(__file__).parent.parent / output_path
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    cfg.num_workers = 0
    cfg.diarizer.manifest_filepath = str(manifest_path)
    cfg.diarizer.out_dir = str(output_path)

    return cfg


def retrieve_user_platform() -> str:
    """
    Retrieve the user's platform.

    Returns:
        str: User's platform. Either 'linux', 'darwin' or 'win32'.
    """
    return sys.platform
