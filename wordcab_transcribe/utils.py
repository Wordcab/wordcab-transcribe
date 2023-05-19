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
from typing import Any, Dict, List, Optional, Tuple, Union

import aiofiles
import aiohttp
import pandas as pd
from loguru import logger
from num2words import num2words
from omegaconf import OmegaConf
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter, normalize
from yt_dlp import YoutubeDL


CURRENCIES_CHARACTERS = [
    "$",
    "€",
    "£",
    "¥",
    "₹",
    "₽",
    "₱",
    "฿",
    "₺",
    "₴",
    "₩",
    "₦",
    "₫",
    "₭",
    "₡",
    "₲",
    "₪",
    "₵",
    "₸",
    "₼",
    "₾",
    "₿",
]


# pragma: no cover
async def async_run_subprocess(command: List[str]) -> tuple:
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


# pragma: no cover
def run_subprocess(command: List[str]) -> tuple:
    """
    Run a subprocess synchronously.

    Args:
        command (List[str]): Command to run.

    Returns:
        tuple: Tuple with the return code, stdout and stderr.
    """
    process = subprocess.Popen(  # noqa: S603,S607
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    return process.returncode, stdout, stderr


def convert_timestamp(timestamp: float, target: str) -> Union[str, float]:
    """
    Use the right function to convert the timestamp.

    Args:
        timestamp (float): Timestamp to convert.
        target (str): Timestamp to convert.

    Returns:
        Union[str, float]: Converted timestamp.

    Raises:
        ValueError: If the target is invalid. Valid targets are: ms, hms, s.
    """
    if target == "ms":
        return timestamp
    elif target == "hms":
        return _convert_ms_to_hms(timestamp)
    elif target == "s":
        return _convert_ms_to_s(timestamp)
    else:
        raise ValueError(
            f"Invalid conversion target: {target}. Valid targets are: ms, hms, s."
        )


def _convert_ms_to_hms(timestamp: float) -> str:
    """
    Convert a timestamp from milliseconds to hours, minutes and seconds.

    Args:
        timestamp (float): Timestamp in milliseconds to convert.

    Returns:
        str: Hours, minutes and seconds.
    """
    hours, remainder = divmod(timestamp / 1000, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = math.floor((seconds % 1) * 1000)

    output = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"

    return output


def _convert_ms_to_s(timestamp: float) -> float:
    """
    Convert a timestamp from milliseconds to seconds.

    Args:
        timestamp (float): Timestamp in milliseconds to convert.

    Returns:
        float: Seconds.
    """
    return timestamp / 1000


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

    new_filepath = f"{filepath.stem}_{filepath.stat().st_mtime_ns}.wav"
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
    result = await async_run_subprocess(cmd)

    if result[0] != 0:
        raise Exception(f"Error converting file {filepath} to wav format: {result[2]}")

    return str(new_filepath)


# pragma: no cover
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


# pragma: no cover
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


def delete_file(filepath: Union[str, Tuple[str]]) -> None:
    """
    Delete a file or a list of files.

    Args:
        filepath (Union[str, Tuple[str]]): Path to the file to delete.
    """
    if isinstance(filepath, str):
        filepath = (filepath,)

    for path in filepath:
        Path(path).unlink(missing_ok=True)


def enhance_audio(
    filepath: str,
    speaker_label: Optional[int] = 0,
    apply_agc: Optional[bool] = True,
    apply_bandpass: Optional[bool] = False,
) -> str:
    """
    Enhance the audio by applying automatic gain control and bandpass filter.

    Args:
        filepath (str): Path to the audio file.
        speaker_label (Optional[str], optional): Speaker label. Defaults to "".
        apply_agc (Optional[bool], optional): Whether to apply automatic gain control. Defaults to True.
        apply_bandpass (Optional[bool], optional): Whether to apply bandpass filter. Defaults to False.

    Returns:
        str: Path to the enhanced audio file.
    """
    audio = AudioSegment.from_file(filepath)
    audio = audio.set_frame_rate(16000)

    if apply_agc:
        audio = normalize(audio)

    if apply_bandpass:
        audio = high_pass_filter(audio, 300)
        audio = low_pass_filter(audio, 3400)

    enhanced_filepath = filepath.replace(".wav", f"_enhanced_{speaker_label}.wav")
    audio.export(enhanced_filepath, format="wav")

    return enhanced_filepath


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
    text = text.strip()

    if text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in [".", "?", "!"]:
        text += "."

    text = text.replace("...", "")
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bi\b", "I", text)

    return text.strip()


def format_segments(segments: list) -> List[dict]:
    """
    Format the segments to a list of dicts with start, end and text keys.

    Args:
        segments (list): List of segments.

    Returns:
        list: List of dicts with start, end and word keys.
    """
    formatted_segments = []

    for segment in segments:
        segment_dict = {}

        segment_dict["start"] = segment["start"]
        segment_dict["end"] = segment["end"]
        segment_dict["word"] = segment["text"].strip()

        formatted_segments.append(segment_dict)

    return formatted_segments


def get_segment_timestamp_anchor(start: float, end: float, option: str = "start"):
    """Get the timestamp anchor for a segment."""
    if option == "end":
        return end
    elif option == "mid":
        return (start + end) / 2
    return start


def interpolate_nans(x: pd.Series, method="nearest") -> pd.Series:
    """
    Interpolate NaNs in a pandas Series using a given method.

    Args:
        x (pd.Series): The Series to interpolate.
        method (str, optional): The interpolation method. Defaults to "nearest".

    Returns:
        pd.Series: The interpolated Series.
    """
    if x.notnull().sum() > 1:
        return x.interpolate(method=method).ffill().bfill()
    else:
        return x.ffill().bfill()


def load_nemo_config(
    domain_type: str, storage_path: str, output_path: str
) -> Dict[str, Any]:
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


# pragma: no cover
def experimental_num_to_words(sentence: str, model_lang: str) -> str:
    """
    Convert numerical values to words. This is an experimental feature.

    Args:
        sentence (str): The sentence to convert.
        model_lang (str): The language of the model.

    Returns:
        str: The converted sentence.
    """
    for wdx, word in enumerate(sentence):
        if any([char.isdigit() for char in word]):
            logger.debug(f"Transcript contains digits: {word}")

            if any([char == "%" for char in word]):
                word = word.replace("%", "")
                to_ = "ordinal" if model_lang not in ["ja", "zh"] else "cardinal"
            elif any([char in CURRENCIES_CHARACTERS for char in word]):
                word = "".join(
                    [char for char in word if char not in CURRENCIES_CHARACTERS]
                )
                to_ = "currency"
            else:
                to_ = "cardinal"

            if word[-1] in [".", ",", "?", "!", ":", ";"]:
                punctuation = word[-1]
                word = word[:-1]
            else:
                punctuation = None

            if "-" in word:
                splitted_word = word.split("-")
            else:
                splitted_word = [word]

            reformatted_word = []
            for word in splitted_word:
                reformatted_word.append(num2words(word, lang=model_lang, to=to_))

            reformatted_word = (
                reformatted_word + [punctuation] if punctuation else reformatted_word
            )

            sentence = sentence[:wdx] + reformatted_word + sentence[wdx + 1 :]

    return " ".join(sentence)


# pragma: no cover
async def split_dual_channel_file(filepath: str) -> Tuple[str, str]:
    """
    Split a dual channel audio file into two mono files using ffmpeg.

    Args:
        filepath (str): The path to the dual channel audio file.

    Returns:
        Tuple[str, str]: The paths to the two mono files.

    Raises:
        Exception: If the file could not be split.
    """
    logger.debug(f"Splitting dual channel file: {filepath}")

    filename = Path(filepath).stem
    filename_left = f"{filename}_left.wav"
    filename_right = f"{filename}_right.wav"

    cmd = [
        "ffmpeg",
        "-i",
        str(filepath),
        "-map_channel",
        "0.0.0",
        str(filename_left),
        "-map_channel",
        "0.0.1",
        str(filename_right),
    ]
    result = await async_run_subprocess(cmd)

    if result[0] != 0:
        raise Exception(f"Error splitting dual channel file: {filepath}. {result[2]}")

    return filename_left, filename_right
