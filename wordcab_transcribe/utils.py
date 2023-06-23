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
import re
import subprocess  # noqa: S404
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import aiofiles
import aiohttp
import filetype
import huggingface_hub
import pandas as pd
import torch
import torchaudio
from fastapi import UploadFile
from loguru import logger
from num2words import num2words
from omegaconf import DictConfig, ListConfig, OmegaConf
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


def check_number_of_segments(chunk_size: int, duration: Union[int, float]) -> int:
    """
    Check the number of chunks considering the duration of the audio.

    Args:
        chunk_size (int): Size of each chunk.
        duration (Union[int, float]): Duration of the audio in milliseconds.

    Returns:
        int: Number of chunks after ceil the division of the duration by the chunk size.
    """
    _duration = _convert_ms_to_s(duration)

    return math.ceil(_duration / chunk_size)


def convert_timestamp(
    timestamp: float, target: str, round_digits: Optional[int] = 3
) -> Union[str, float]:
    """
    Use the right function to convert the timestamp.

    Args:
        timestamp (float): Timestamp to convert.
        target (str): Timestamp to convert.
        round_digits (int, optional): Number of digits to round the timestamp. Defaults to 3.

    Returns:
        Union[str, float]: Converted timestamp.

    Raises:
        ValueError: If the target is invalid. Valid targets are: ms, hms, s.
    """
    if target == "ms":
        return round(_convert_s_to_ms(timestamp), round_digits)
    elif target == "hms":
        return _convert_s_to_hms(timestamp)
    elif target == "s":
        return round(timestamp, round_digits)
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


def _convert_s_to_ms(timestamp: float) -> float:
    """
    Convert a timestamp from seconds to milliseconds.

    Args:
        timestamp (float): Timestamp in seconds to convert.

    Returns:
        float: Milliseconds.
    """
    return timestamp * 1000


def _convert_s_to_hms(timestamp: float) -> str:
    """
    Convert a timestamp from seconds to hours, minutes and seconds.

    Args:
        timestamp (float): Timestamp in seconds to convert.

    Returns:
        str: Hours, minutes and seconds.
    """
    hours, remainder = divmod(timestamp, 3600)
    minutes, seconds = divmod(remainder, 60)

    output = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.000"

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
        _filepath = Path(filepath)

    if not _filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")

    new_filepath = f"{_filepath.stem}_{_filepath.stat().st_mtime_ns}.wav"
    cmd = [
        "ffmpeg",
        "-i",
        filepath,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        new_filepath,
    ]
    result = await async_run_subprocess(cmd)

    if result[0] != 0:
        raise Exception(f"Error converting file {filepath} to wav format: {result[2]}")

    return new_filepath


# pragma: no cover
def download_file_from_youtube(url: str, filename: str) -> str:
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
    url: str,
    filename: str,
    url_headers: Optional[Dict[str, str]] = None,
    guess_extension: Optional[bool] = True,
) -> Tuple[str, str]:
    """
    Download an audio file from a URL.

    Args:
        url (str): URL of the audio file.
        filename (str): Filename to save the file as.
        url_headers (Optional[Dict[str, str]]): Headers to send with the request. Defaults to None.
        guess_extension (Optional[bool]): Whether to guess the file extension based on the
            Content-Type header. Defaults to True.

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
                parsed_url = urlparse(url)
                url_path = parsed_url.path
                possible_filename = url_path.split("/")[-1]
                logger.info(f"Possible filename: {possible_filename}")
                if "." not in possible_filename:
                    logger.info("No '.' found in path file, guessing file type")
                    file_content = await response.read()
                    extension = filetype.guess(file_content).extension
                else:
                    extension = possible_filename.split(".")[-1]
                    logger.info(f"Extension detected: {extension}")

                filename = f"{filename}.{extension}"

                logger.info(f"New file name: {filename}")
                async with aiofiles.open(filename, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024)

                        if not chunk:
                            break

                        await f.write(chunk)
            else:
                raise Exception(f"Failed to download file. Status: {response.status}")

    return filename, extension


# pragma: no cover
def download_model(compute_type: str, language: str) -> Optional[str]:
    """
    Download the special models during the warmup phase.

    Args:
        compute_type (str): The target compute type.
        language (str): The target language.

    Returns:
        Optional[str]: The path to the downloaded model.
    """
    if compute_type == "float16":
        repo_id = f"wordcab/whisper-large-fp16-{language}"
    elif compute_type == "int8_float16":
        repo_id = f"wordcab/whisper-large-int8-fp16-{language}"
    elif compute_type == "int8":
        repo_id = f"wordcab/whisper-large-int8-{language}"
    else:
        # No other models are supported
        return None

    allow_patterns = ["config.json", "model.bin", "tokenizer.json", "vocabulary.*"]
    snapshot_path = huggingface_hub.snapshot_download(
        repo_id, local_files_only=False, allow_patterns=allow_patterns
    )

    return snapshot_path


def delete_file(filepath: Union[str, Tuple[str, Optional[str]]]) -> None:
    """
    Delete a file or a list of files.

    Args:
        filepath (Union[str, Tuple[str]]): Path to the file to delete.
    """
    if isinstance(filepath, str):
        filepath = (filepath, None)

    for path in filepath:
        if path:
            Path(path).unlink(missing_ok=True)


def enhance_audio(
    audio: Union[str, torch.Tensor],
    apply_agc: Optional[bool] = True,
    apply_bandpass: Optional[bool] = False,
) -> torch.Tensor:
    """
    Enhance the audio by applying automatic gain control and bandpass filter.

    Args:
        audio (Union[str, torch.Tensor]): Path to the audio file or the waveform.
        apply_agc (Optional[bool], optional): Whether to apply automatic gain control. Defaults to True.
        apply_bandpass (Optional[bool], optional): Whether to apply bandpass filter. Defaults to False.

    Returns:
        torch.Tensor: The enhanced audio waveform.
    """
    if isinstance(audio, str):
        waveform, sample_rate = torchaudio.load(audio)
    else:
        waveform = audio
        sample_rate = 16000

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000,
        )
        waveform = transform(waveform)
        sample_rate = 16000

    if apply_agc:
        # Volmax normalization to mimic AGC
        waveform /= waveform.abs().max()

    if apply_bandpass:
        highpass = torchaudio.functional.highpass_biquad(waveform, sample_rate, 300)
        waveform = torchaudio.functional.lowpass_biquad(highpass, sample_rate, 3400)

    return waveform


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
    if text[-1] not in [".", "?", "!", ":", ";", ","]:
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


def format_segments(
    segments: list, alignment: bool, use_batch: bool, word_timestamps: bool
) -> List[dict]:
    """
    Format the segments to a list of dicts with start, end and text keys. Optionally include word timestamps.

    Args:
        segments (list): List of segments.
        alignment (bool): Whether the segments have been aligned. Used to format the word timestamps correctly.
        use_batch (bool): Whether the segments are from a batch. Used to format the word timestamps correctly.
        word_timestamps (bool): Whether to include word timestamps.

    Returns:
        list: List of dicts with start, end and word keys.
    """
    formatted_segments = []

    for segment in segments:
        segment_dict = {}

        segment_dict["start"] = segment["start"]
        segment_dict["end"] = segment["end"]
        segment_dict["text"] = segment["text"].strip()
        if word_timestamps:
            if alignment:
                segment_dict["words"] = segment["words"]
            else:
                if use_batch:
                    _words = [
                        {
                            "word": word["word"].strip(),
                            "start": word["start"],
                            "end": word["end"],
                            "score": round(word["probability"], 2),
                        }
                        for word in segment["words"]
                    ]
                else:
                    _words = [
                        {
                            "word": word.word.strip(),
                            "start": word.start,
                            "end": word.end,
                            "score": round(word.probability, 2),
                        }
                        for word in segment["words"]
                    ]
                segment_dict["words"] = _words

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
    domain_type: str, storage_path: str, output_path: str, temp_folder: Path
) -> Union[DictConfig, ListConfig]:
    """
    Load NeMo config file based on a domain type.

    Args:
        domain_type (str): The domain type. Can be "general", "meeting" or "telephonic".
        storage_path (str): The path to the NeMo storage directory.
        output_path (str): The path to the NeMo output directory.
        temp_folder (Path): The path to the temporary folder.

    Returns:
        DictConfig: The NeMo config loaded as a DictConfig.
    """
    cfg_path = (
        Path(__file__).parent.parent
        / "config"
        / "nemo"
        / f"diar_infer_{domain_type}.yaml"
    )
    with open(cfg_path) as f:
        cfg = OmegaConf.load(f)

    _storage_path = Path(__file__).parent.parent / storage_path
    if not _storage_path.exists():
        _storage_path.mkdir(parents=True, exist_ok=True)

    meta = {
        "audio_filepath": f"/app/{str(temp_folder)}/mono_file.wav",
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }

    manifest_path = _storage_path / "infer_manifest.json"
    with open(manifest_path, "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    _output_path = Path(__file__).parent.parent / output_path
    if not _output_path.exists():
        _output_path.mkdir(parents=True, exist_ok=True)

    cfg.num_workers = 0
    cfg.diarizer.manifest_filepath = str(manifest_path)
    cfg.diarizer.out_dir = str(output_path)

    return cfg


def remove_words_for_svix(dict_payload: dict) -> dict:
    """
    Remove the words from the utterances for SVIX.

    Args:
        dict_payload (dict): The dict payload.

    Returns:
        dict: The dict payload with the words removed.
    """
    for utterance in dict_payload["utterances"]:
        utterance.pop("words", None)

    return dict_payload


def retrieve_user_platform() -> str:
    """
    Retrieve the user's platform.

    Returns:
        str: User's platform. Either 'linux', 'darwin' or 'win32'.
    """
    return sys.platform


async def save_file_locally(filename: str, file: UploadFile) -> bool:
    """
    Save a file locally from an UploadFile object.

    Args:
        filename (str): The filename to save the file as.
        file (UploadFile): The UploadFile object.

    Returns:
        bool: Whether the file was saved successfully.
    """
    async with aiofiles.open(filename, "wb") as f:
        audio_bytes = await file.read()
        await f.write(audio_bytes)

    return True


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
