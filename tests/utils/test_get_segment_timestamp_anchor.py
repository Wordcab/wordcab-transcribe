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
