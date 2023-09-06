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
"""Live endpoints for the Wordcab Transcribe API."""

import asyncio
from datetime import datetime
from enum import Enum
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, SecretStr

from wordcab_transcribe.dependencies import asr

router = APIRouter()


class LiveConnectionStatus(str, Enum):
    """Live connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    TRANSCRIBING = "transcribing"


class LiveConsumer(BaseModel):
    """Manage live transcription consumers."""

    client_id: int
    status: LiveConnectionStatus
    usage: float = 0.0
    last_connection: datetime = datetime.now()


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self.active_connections: List[WebSocket] = []
        self.live_consumers: List[LiveConsumer] = []

    async def connect(self, websocket: WebSocket, client_id: int) -> None:
        """Connect a WebSocket."""
        if len(self.active_connections) > 1:
            await websocket.close(code=1001)

        await websocket.accept()
        self.active_connections.append(websocket)
        self.live_consumers.append(
            LiveConsumer(client_id=client_id, status=LiveConnectionStatus.CONNECTED)
        )

    def disconnect(self, websocket: WebSocket, client_id: int) -> None:
        """Disconnect a WebSocket."""
        self.active_connections.remove(websocket)
        consumer_index = self.live_consumers.index(
            next(filter(lambda x: x.client_id == client_id, self.live_consumers))
        )
        self.live_consumers.index(consumer_index).status = LiveConnectionStatus.DISCONNECTED


manager = ConnectionManager()

@router.websocket("")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket connections."""
    # await manager.connect(websocket, client_id)
    await manager.connect(websocket, 1)

    try:
        while True:
            # data = await websocket.receive_bytes()
            await asyncio.sleep(1)
            await websocket.send_text("Hello, world!")
            await asyncio.sleep(3)
            await websocket.send_text("Hello, world!")
            await asyncio.sleep(5)
            raise WebSocketDisconnect

    except WebSocketDisconnect:
        manager.disconnect(websocket, 1)
