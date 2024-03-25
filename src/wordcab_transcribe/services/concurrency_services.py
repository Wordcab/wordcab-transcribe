# Copyright 2024 The Wordcab Team. All rights reserved.
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
from typing import List


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
        if not any(item == device_index for item in self.queue._queue):
            self.queue.put_nowait(device_index)


class URLService:
    """URL service class to handle multiple remote URLs."""

    def __init__(self, remote_urls: List[str]) -> None:
        """
        Initialize the URL service.

        Args:
            remote_urls (List[str]): List of remote URLs to use.
        """
        self.remote_urls: List[str] = remote_urls
        self._init_queue()

    def _init_queue(self) -> None:
        """Initialize the queue with the available URLs."""
        self.queue = asyncio.Queue(maxsize=len(self.remote_urls))
        for url in self.remote_urls:
            self.queue.put_nowait(url)

    def get_queue_size(self) -> int:
        """
        Get the current queue size.

        Returns:
            int: Current queue size.
        """
        return self.queue.qsize()

    def get_urls(self) -> List[str]:
        """
        Get the list of available URLs.

        Returns:
            List[str]: List of available URLs.
        """
        return self.remote_urls

    async def next_url(self) -> str:
        """
        We use this to iterate equally over the available URLs.

        Returns:
            str: Next available URL.
        """
        url = self.queue.get_nowait()
        # Unlike GPU we don't want to block remote ASR requests.
        # So we re-insert the URL back into the queue after getting it.
        self.queue.put_nowait(url)

        return url

    async def add_url(self, url: str) -> None:
        """
        Add a URL to the pool of available URLs.

        Args:
            url (str): URL to add to the queue.
        """
        if url not in self.remote_urls:
            self.remote_urls.append(url)

            # Re-initialize the queue with the new URL.
            self._init_queue()

    async def remove_url(self, url: str) -> None:
        """
        Remove a URL from the pool of available URLs.

        Args:
            url (str): URL to remove from the queue.
        """
        if url in self.remote_urls:
            self.remote_urls.remove(url)

            # Re-initialize the queue without the removed URL.
            self._init_queue()
