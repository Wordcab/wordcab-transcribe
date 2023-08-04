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

"""Automate module of the Wordcab Transcribe."""


import json
from typing import List, Optional

import requests

from wordcab_transcribe.models import AudioResponse, BaseResponse, YouTubeResponse


def run_api_youtube(
    url: str,
    source_lang: str = "en",
    timestamps: str = "s",
    word_timestamps: bool = False,
    alignment: bool = False,
    diarization: bool = False,
    server_url: Optional[str] = None,
    vocab: Optional[List[str]] = None,
    timeout: int = 900,
) -> YouTubeResponse:
    """
    Run API call for Youtube videos.

    Args:
        url (str): URL source of the Youtube video.
        source_lang: language of the URL source (defaulted to English)
        timestamps: time unit of the timestamps (defaulted to seconds)
        word_timestamps: associated words and their timestamps (defaulted to False)
        alignment: re-align timestamps (defaulted to False)
        diarization: speaker labels for utterances (defaulted to False)
        server_url: the URL used to reach out the API
        vocab: defaulted to empty list
        timeout: defaulted to 90 seconds (15 minutes)

    Returns:
        AudioResponse
    """
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    params = {"url": url}
    data = {
        "alignment": alignment,
        "diarization": diarization,
        "source_lang": source_lang,
        "timestamps": timestamps,
        "word_timestamps": word_timestamps,
    }
    if vocab:
        data["vocab"] = vocab

    if server_url is None:
        response = requests.post(
            "http://localhost:5001/api/v1/youtube",
            headers=headers,
            params=params,
            data=json.dumps(data),
            timeout=timeout,
        )
    else:
        response = requests.post(
            f"http://localhost:5001/api/{server_url}",
            # f"{server_url}/api/v1/youtube",
            headers=headers,
            params=params,
            data=json.dumps(data),
            timeout=timeout,
        )

    try:
        r_json = response.json()
        url_name = url.split("https://")[-1]
        with open(f"{url_name}.json", "w", encoding="utf-8") as f:
            json.dump(r_json, f, indent=4, ensure_ascii=False)
        return response
    except Exception:
        print("An exception occurred")


def run_audio_url(
    url: str,
    source_lang: str = "en",
    timestamps: str = "s",
    word_timestamps: bool = False,
    alignment: bool = False,
    diarization: bool = False,
    dual_channel: bool = False,
    server_url: Optional[str] = None,
    vocab: Optional[List[str]] = None,
    timeout: int = 900,
) -> AudioResponse:
    """
    Run API call for audio URLs.

    Args:
        url (str): URL source of the audio URL.
        source_lang: language of the URL source (defaulted to English)
        timestamps: time unit of the timestamps (defaulted to seconds)
        word_timestamps: associated words and their timestamps (defaulted to False)
        alignment: re-align timestamps (defaulted to False)
        diarization: speaker labels for utterances (defaulted to False)
        dual_channel: defaulted to False
        server_url: the URL used to reach out the API
        vocab: defaulted to empty list
        timeout: defaulted to 90 seconds (15 minutes)

    Returns:
        AudioResponse
    """
    headers = {"accept": "application/json", "content-type": "application/json"}

    params = {"url": url}

    data = {
        "alignment": alignment,
        "diarization": diarization,
        "source_lang": source_lang,
        "timestamps": timestamps,
        "word_timestamps": word_timestamps,
        "dual_channel": dual_channel,
    }
    if vocab:
        data["vocab"] = vocab

    if server_url is None:
        response = requests.post(
            "https://wordcab.com/api/v1/audio-url",
            headers=headers,
            params=params,
            data=data,
            timeout=timeout,
        )
    else:
        response = requests.post(
            f"{server_url}/api/v1/audio-url",
            headers=headers,
            params=params,
            data=data,
            timeout=timeout,
        )

    try:
        r_json = response.json()
        url_name = url.split("https://")[-1]
        with open(f"{url_name}.json", "w", encoding="utf-8") as f:
            json.dump(r_json, f, indent=4, ensure_ascii=False)
        return response
    except Exception:
        print("An exception occurred")


def run_api_audio_file(
    file: str,
    source_lang: str = "en",
    timestamps: str = "s",
    word_timestamps: bool = False,
    alignment: bool = False,
    diarization: bool = False,
    dual_channel: bool = False,
    server_url: Optional[str] = None,
    vocab: Optional[List[str]] = None,
    timeout: int = 900,
) -> AudioResponse:
    """
    Run API call for audio files.

    Args:
        file (str): source of the audio file.
        source_lang: language of the URL source (defaulted to English)
        timestamps: time unit of the timestamps (defaulted to seconds)
        word_timestamps: associated words and their timestamps (defaulted to False)
        alignment: re-align timestamps (defaulted to False)
        diarization: speaker labels for utterances (defaulted to False)
        dual_channel: defaulted to False
        server_url: the URL used to reach out the API
        vocab: defaulted to empty list
        timeout: defaulted to 90 seconds (15 minutes)

    Returns:
        AudioResponse
    """
    data = {
        "alignment": alignment,
        "diarization": diarization,
        "source_lang": source_lang,
        "timestamps": timestamps,
        "word_timestamps": word_timestamps,
        "dual_channel": dual_channel,
    }
    if vocab:
        data["vocab"] = vocab

    with open(file, "rb") as f:
        files = {"file": f}
        if server_url is None:
            response = requests.post(
                "http://localhost:5001/api/v1/audio",
                files=files,
                data=data,
                timeout=timeout,
            )
        else:
            response = requests.post(
                f"{server_url}/api/v1/audio",
                files=files,
                data=data,
                timeout=timeout,
            )

    try:
        r_json = response.json()
        filename = file.split(".")[0]
        with open(f"{filename}.json", "w", encoding="utf-8") as f:
            json.dump(r_json, f, indent=4, ensure_ascii=False)
        return response
    except Exception:
        print("An exception occurred")


# run API function that will delegate to other functions based on the endpoint
def run_api(
    endpoint: str,
    source: str,
    source_lang: str = "en",
    timestamps: str = "s",
    word_timestamps: bool = False,
    alignment: bool = False,
    diarization: bool = False,
    dual_channel: bool = False,
    server_url: Optional[str] = None,
    vocab: Optional[List[str]] = None,
    timeout: int = 900,
) -> BaseResponse:
    """
    Automated function for API calls for 3 endpoints: audio files, youtube videos, and audio URLs.

    Args:
        endpoint (str): audio_file or youtube or audioURL
        source: source of the endpoint (either a URL or a filepath)
        source_lang: language of the source (defaulted to English)
        timestamps: time unit of the timestamps (defaulted to seconds)
        word_timestamps: whether the timestamps are represented by words (defaulted to False)
        alignment: re-align timestamps (defaulted to False)
        diarization: speaker labels for utterances (defaulted to False)
        dual_channel: defaulted to False
        server_url: the URL used to reach out the API
        vocab: defaulted to empty list
        timeout: defaulted to 900 seconds (15 minutes)

    Returns:
        BaseResponse
    """
    if endpoint == "youtube":
        response = run_api_youtube(
            source,
            source_lang,
            timestamps,
            word_timestamps,
            alignment,
            diarization,
            server_url,
            vocab,
            timeout,
        )
    elif endpoint == "audio_file":
        response = run_api_audio_file(
            source,
            source_lang,
            timestamps,
            word_timestamps,
            alignment,
            diarization,
            dual_channel,
            server_url,
            vocab,
            timeout,
        )
    elif endpoint == "audio_url":
        response = run_audio_url(
            source,
            source_lang,
            timestamps,
            word_timestamps,
            alignment,
            diarization,
            dual_channel,
            server_url,
            vocab,
            timeout,
        )
    return response
