# Copyright 2024 The Wordcab Team. All rights reserved.
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
