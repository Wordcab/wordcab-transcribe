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
"""Tests the delete_file function."""
from wordcab_transcribe.utils import delete_file


def test_delete_file(tmp_path) -> None:
    """Test the delete_file function by creating a temporary file and deleting it."""
    file_path = tmp_path / "test_file.txt"
    file_path.touch()

    assert file_path.exists()

    delete_file(str(file_path))

    assert not file_path.exists()
