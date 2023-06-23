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
"""GPU service class to handle gpu availability for models."""

import asyncio
from typing import Any, Dict, List


class GPUService:
    """GPU service class to handle gpu availability for models."""
    def __init__(self, device: str, device_index: List[int]) -> None:
        """
        Initialize the GPU service.

        Args:
            device (str): Device to use for inference. Can be "cpu" or "cuda".
            device_index (List[int]): Index of the device to use for inference.
        """
        self.device: str = device
        self.device_index: List[int] = device_index

        # Initialize the models dictionary that will hold the models for each GPU.
        self.models: Dict[int, Any] = {}

        self.queue = asyncio.Queue(maxsize=len(self.device_index))
        for idx in self.device_index:
            self.queue.put_nowait(idx)
        

    async def get_device(self) -> int:
        """
        Get the next available device.

        Returns:
            int: Index of the next available device.
        """
        return await self.queue.get()

    def release_device(self, device_index: int) -> None:
        """
        Return a device to the available devices list.

        Args:
            device_index (int): Index of the device to add to the available devices list.
        """
        self.queue.put_nowait(device_index)
