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
"""Test important files are present in the project."""

import pathlib

important_files = [
    ".darglint",
    ".flake8",
    ".github",
    ".gitignore",
    ".pre-commit-config.yaml",
    "tests",
    "wordcab_transcribe",
    "poetry.lock",
    "pyproject.toml",
    "Dockerfile",
    "LICENSE",
    "README.md",
]


def test_important_files_present():
    """Test important files are present in the project."""
    for file in important_files:
        assert pathlib.Path(file).exists()
