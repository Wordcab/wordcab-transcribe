# Copyright 2024 The Wordcab Team. All rights reserved.
#
# Licensed under the MIT License (the "License");
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
"""Tests the format_punct function."""
from wordcab_transcribe.utils import format_punct


def test_format_punct_remove_ellipsis() -> None:
    """Test the format_punct function for ellipsis."""
    assert format_punct("Hello... World...") == "Hello World"


def test_format_punct_remove_space_before_question_mark() -> None:
    """Test the format_punct function for question mark."""
    assert format_punct("What ?") == "What?"


def test_format_punct_remove_space_before_exclamation_mark() -> None:
    """Test the format_punct function for exclamation mark."""
    assert format_punct("Wow !") == "Wow!"


def test_format_punct_remove_space_before_period() -> None:
    """Test the format_punct function for period."""
    assert format_punct("Hello .") == "Hello."


def test_format_punct_remove_space_before_comma() -> None:
    """Test the format_punct function for comma."""
    assert format_punct("Hello ,") == "Hello,"


def test_format_punct_remove_space_before_colon() -> None:
    """Test the format_punct function for colon."""
    assert format_punct("Hello :") == "Hello:"


def test_format_punct_remove_space_before_semicolon() -> None:
    """Test the format_punct function for semicolon."""
    assert format_punct("Hello ;") == "Hello;"


def test_format_punct_remove_extra_spaces() -> None:
    """Test the format_punct function for extra spaces."""
    assert format_punct("  Hello World  ") == "Hello World."
