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
        while True:
            try:
                device_index = self.queue.get_nowait()
                return device_index
            except asyncio.QueueEmpty:
                await asyncio.sleep(1.0)

    def release_device(self, device_index: int) -> None:
        """
        Return a device to the available devices list.

        Args:
            device_index (int): Index of the device to add to the available devices list.
        """
        self.queue.put_nowait(device_index)
