# https://github.com/guillaumekln/faster-whisper/blob/master/faster_whisper/utils.py
import re
from pathlib import PosixPath
from typing import List, Optional

import huggingface_hub
import requests

_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}


def available_models() -> List[str]:
    """Returns the names of available models."""
    return list(_MODELS.keys())


def download_model(
    size_or_id: str,
    output_dir: Optional[PosixPath] = None,
    local_files_only: bool = False,
):
    """Downloads a CTranslate2 Whisper model from the Hugging Face Hub.

    Args:
      size_or_id: Size of the model to download from https://huggingface.co/guillaumekln
        (tiny, tiny.en, base, base.en, small, small.en medium, medium.en, large-v1, large-v2,
        large), or a CTranslate2-converted model ID from the Hugging Face Hub
        (e.g. guillaumekln/faster-whisper-large-v2).
      output_dir: Directory where the model should be saved. If not set, the model is saved in
        the cache directory.
      local_files_only:  If True, avoid downloading the file and return the path to the local
        cached file if it exists.

    Returns:
      The path to the downloaded model.

    Raises:
      ValueError: if the model size is invalid.
    """
    if re.match(r".*/.*", size_or_id):
        repo_id = size_or_id
    else:
        repo_id = _MODELS.get(size_or_id)
        if repo_id is None:
            raise ValueError(
                "Invalid model size '%s', expected one of: %s"
                % (size_or_id, ", ".join(_MODELS.keys()))
            )

    allow_patterns = [
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]

    kwargs = {
        "local_files_only": local_files_only,
        "allow_patterns": allow_patterns,
    }

    if output_dir is not None:
        kwargs["local_dir"] = str(output_dir)
        kwargs["local_dir_use_symlinks"] = False

    try:
        return huggingface_hub.snapshot_download(repo_id, **kwargs)
    except (
        huggingface_hub.utils.HfHubHTTPError,
        requests.exceptions.ConnectionError,
    ):
        kwargs["local_files_only"] = True
        return huggingface_hub.snapshot_download(repo_id, **kwargs)
