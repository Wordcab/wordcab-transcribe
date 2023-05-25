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
"""Dependencies for the API."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from wordcab_transcribe.config import settings
from wordcab_transcribe.services.asr_service import ASRAsyncService, ASRLiveService


# Define the ASR service to use depending on the settings
if settings.asr_type == "live":
    asr = ASRLiveService()
elif settings.asr_type == "async":
    asr = ASRAsyncService()
else:
    asr = None
    raise ValueError(f"Invalid ASR type: {settings.asr_type}")


hardware_num_cores = multiprocessing.cpu_count()

# Define the number of workers for the I/O bound tasks
io_executor = ThreadPoolExecutor(max_workers=hardware_num_cores)

# TODO: Define the number of workers for the CPU bound tasks: Transcription, Alignment, Diarization
