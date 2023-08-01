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
"""Dependencies for the API."""

import asyncio

from wordcab_transcribe.config import settings
from wordcab_transcribe.services.asr_service import ASRAsyncService, ASRLiveService


# Define the ASR service to use depending on the settings
if settings.asr_type == "live":
    asr = ASRLiveService()
elif settings.asr_type == "async":
    asr = ASRAsyncService()
else:
    raise ValueError(f"Invalid ASR type: {settings.asr_type}")


# Define the maximum number of files to pre-download for the async ASR service
download_limit = asyncio.Semaphore(10)
