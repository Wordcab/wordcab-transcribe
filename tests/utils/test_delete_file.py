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
"""Tests the delete_file function."""
from wordcab_transcribe.utils import delete_file


def test_delete_file(tmp_path) -> None:
    """Test the delete_file function by creating a temporary file and deleting it."""
    file_path = tmp_path / "test_file.txt"
    file_path.touch()

    assert file_path.exists()

    delete_file(str(file_path))

    assert not file_path.exists()
