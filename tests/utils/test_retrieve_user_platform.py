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
"""Tests the retrieve_user_platform function."""

import platform
from wordcab_transcribe.utils import retrieve_user_platform


def test_retrieve_user_platform_windows() -> None:
    """Test the retrieve_user_platform function on Windows."""
    platform.platform = lambda: "Windows-10-10.0.19041-SP0"
    assert retrieve_user_platform() == "Windows"

def test_retrieve_user_platform_ubuntu() -> None:
    """Test the retrieve_user_platform function on Ubuntu."""
    platform.platform = lambda: "Linux-5.11.0-27-generic-x86_64-with-Ubuntu-20.04-focal"
    assert retrieve_user_platform() == "Ubuntu"

def test_retrieve_user_platform_linux() -> None:
    """Test the retrieve_user_platform function on Linux."""
    platform.platform = lambda: "Linux-5.16.0-rc3+ x86_64"
    assert retrieve_user_platform() == "Linux"

def test_retrieve_user_platform_macos() -> None:
    """Test the retrieve_user_platform function on MacOS."""
    platform.platform = lambda: "Darwin-21.2.0-x86_64-i386-64bit"
    assert retrieve_user_platform() == "MacOS"

def test_retrieve_user_platform_unknown() -> None:
    """Test the retrieve_user_platform function on an unknown platform."""
    platform.platform = lambda: "FooBar-1.0"
    assert retrieve_user_platform() == "Unknown"
