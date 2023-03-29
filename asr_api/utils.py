# Copyright (c) 2023, The Wordcab team. All rights reserved.
"""Utils module of the Wordcab ASR API."""

import io
import soundfile as sf


def get_duration(audio_bytes: io.BytesIO) -> float:
    """
    Get the duration of the audio file.

    Args:
        audio_bytes (io.BytesIO): Audio file.

    Returns:
        float: Duration of the audio file.
    """
    data, samplerate = sf.read(audio_bytes)

    return len(data) / samplerate


def format_segments(segments: list, use_dict: bool = False, include_words: bool = False) -> list:
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
                    "probability": word.probability
                } 
                for word in segment.words
            ]
            segment_dict["words"] = words

        formatted_segments.append(segment_dict)

    return formatted_segments
