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
"""Tests the is_empty_string function."""
from wordcab_transcribe.utils import is_empty_string


def test_is_empty_string_empty() -> None:
    """Test the is_empty_string function for empty."""
    assert is_empty_string("") is True


def test_is_empty_string_only_spaces() -> None:
    """Test the is_empty_string function for spaces."""
    assert is_empty_string("   ") is True


def test_is_empty_string_only_periods() -> None:
    """Test the is_empty_string function for periods."""
    assert is_empty_string("...") is True


def test_is_empty_string_spaces_and_periods() -> None:
    """Test the is_empty_string function for spaces and periods."""
    assert is_empty_string(" . . .  ") is True


def test_is_empty_string_non_empty() -> None:
    """Test the is_empty_string function for non-empty."""
    assert is_empty_string("Hello, world!") is False


def test_is_empty_string_non_empty_with_spaces() -> None:
    """Test the is_empty_string function for non-empty with spaces."""
    assert is_empty_string(" Wordcab Transcribe ") is False


def test_is_empty_string_non_empty_with_periods() -> None:
    """Test the is_empty_string function for non-empty with periods."""
    assert is_empty_string("Hello. World.") is False


def test_is_empty_string_non_empty_with_spaces_and_periods() -> None:
    """Test the is_empty_string function for non-empty with spaces and periods."""
    assert is_empty_string("  Hello.  World.  ") is False
