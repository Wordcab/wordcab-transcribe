import asyncio
import io
import json
from typing import List, Tuple, Union

import aiohttp
import soundfile as sf
import torch
import torchaudio
from pydantic import BaseModel
from tensorshare import Backend, TensorShare


def read_audio(
    audio: Union[str, bytes],
    offset_start: Union[float, None] = None,
    offset_end: Union[float, None] = None,
    sample_rate: int = 16000,
) -> Tuple[torch.Tensor, float]:
    """
    Read an audio file and return the audio tensor.

    Args:
        audio (Union[str, bytes]):
            Path to the audio file or the audio bytes.
        offset_start (Union[float, None], optional):
            When to start reading the audio file. Defaults to None.
        offset_end (Union[float, None], optional):
            When to stop reading the audio file. Defaults to None.
        sample_rate (int):
            The sample rate of the audio file. Defaults to 16000.

    Returns:
        Tuple[torch.Tensor, float]: The audio tensor and the audio duration.
    """
    if isinstance(audio, str):
        wav, sr = torchaudio.load(audio)
    elif isinstance(audio, bytes):
        with io.BytesIO(audio) as buffer:
            wav, sr = sf.read(
                buffer, format="RAW", channels=1, samplerate=16000, subtype="PCM_16"
            )
        wav = torch.from_numpy(wav).unsqueeze(0)
    else:
        raise ValueError(
            f"Invalid audio type. Must be either str or bytes, got: {type(audio)}."
        )

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        wav = transform(wav)
        sr = sample_rate

    audio_duration = float(wav.shape[1]) / sample_rate

    # Convert start and end times to sample indices based on the new sample rate
    if offset_start is not None:
        start_sample = int(offset_start * sr)
    else:
        start_sample = 0

    if offset_end is not None:
        end_sample = int(offset_end * sr)
    else:
        end_sample = wav.shape[1]

    # Trim the audio based on the new start and end samples
    wav = wav[:, start_sample:end_sample]

    return wav.squeeze(0), audio_duration


class TranscribeRequest(BaseModel):
    """Request model for the transcribe endpoint."""

    audio: Union[TensorShare, List[TensorShare]]
    compression_ratio_threshold: float
    condition_on_previous_text: bool
    internal_vad: bool
    log_prob_threshold: float
    no_speech_threshold: float
    repetition_penalty: float
    source_lang: str
    vocab: Union[List[str], None]


async def main():

    audio, _ = read_audio("data/HL_Podcast_1.mp3")
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

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url="http://0.0.0.0:5002/api/v1/transcribe",
            data=data.model_dump_json(),
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status != 200:
                raise Exception(
                    f"Remote transcription failed with status {response.status}."
                )
            else:
                r = await response.json()

    with open("remote_test.json", "w") as f:
        f.write(json.dumps(r, indent=4))


asyncio.run(main())
