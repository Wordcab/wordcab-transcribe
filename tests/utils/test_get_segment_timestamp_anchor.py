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
"""Tests the get_segment_timestamp_anchor function."""
from wordcab_transcribe.utils import get_segment_timestamp_anchor


def test_get_segment_timestamp_anchor_start() -> None:
    """Test the get_segment_timestamp_anchor function for start."""
    assert get_segment_timestamp_anchor(10.0, 20.0, "start") == 10.0


def test_get_segment_timestamp_anchor_end() -> None:
    """Test the get_segment_timestamp_anchor function for end."""
    assert get_segment_timestamp_anchor(10.0, 20.0, "end") == 20.0


def test_get_segment_timestamp_anchor_mid() -> None:
    """Test the get_segment_timestamp_anchor function for mid."""
    assert get_segment_timestamp_anchor(10.0, 20.0, "mid") == 15.0


def test_get_segment_timestamp_anchor_default() -> None:
    """Test the get_segment_timestamp_anchor function for default."""
    assert get_segment_timestamp_anchor(10.0, 20.0) == 10.0
