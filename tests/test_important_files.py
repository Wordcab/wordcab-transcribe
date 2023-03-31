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
