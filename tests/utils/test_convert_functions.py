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
"""Tests the conversion functions."""
from typing import Union

import pytest

from wordcab_transcribe.utils import (
    _convert_ms_to_hms,
    _convert_ms_to_s,
    _convert_s_to_hms,
    _convert_s_to_ms,
    convert_timestamp,
)


@pytest.mark.parametrize(
    "timestamp, target, expected",
    [
        (1, "ms", 1000),
        (1, "s", 1),
        (1, "hms", "00:00:01.000"),
        (3600, "hms", "01:00:00.000"),
        (3661, "hms", "01:01:01.000"),
    ],
)
def test_convert_timestamp(
    timestamp: float, target: str, expected: Union[str, float]
) -> None:
    """Test the convert_timestamp function."""
    assert convert_timestamp(timestamp, target) == expected


def test_convert_timestamp_raises_error() -> None:
    """Test the convert_timestamp function raises error."""
    with pytest.raises(ValueError):
        convert_timestamp(1000, "invalid_target", False)


@pytest.mark.parametrize("ms, expected", [(1000, 1), (3600000, 3600), (3661000, 3661)])
def test_convert_ms_to_s(ms: float, expected: float) -> None:
    """Test the _convert_ms_to_s function."""
    assert _convert_ms_to_s(ms) == expected


@pytest.mark.parametrize(
    "ms, expected",
    [(1000, "00:00:01.000"), (3600000, "01:00:00.000"), (3661000, "01:01:01.000")],
)
def test_convert_ms_to_hms(ms: float, expected: str) -> None:
    """Test the _convert_ms_to_hms function."""
    assert _convert_ms_to_hms(ms) == expected


@pytest.mark.parametrize("s, expected", [(1, 1000), (3600, 3600000), (3661, 3661000)])
def test_convert_s_to_ms(s: float, expected: float) -> None:
    """Test the _convert_s_to_ms function."""
    assert _convert_s_to_ms(s) == expected


@pytest.mark.parametrize(
    "s, expected",
    [(1, "00:00:01.000"), (3600, "01:00:00.000"), (3661, "01:01:01.000")],
)
def test_convert_s_to_hms(s: float, expected: str) -> None:
    """Test the _convert_s_to_hms function."""
    assert _convert_s_to_hms(s) == expected
