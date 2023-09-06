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


class LiveConsumer(BaseModel):
    """Manage live transcription consumers."""

    client_id: str
    api_key: SecretStr
    status: LiveConnectionStatus
    usage: float = 0.0
    last_connection: datetime = datetime.now()


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self.active_connections: List[WebSocket] = []
        self.live_consumers: List[LiveConsumer] = []

    async def connect(self, websocket: WebSocket, client_id: str, api_key: str) -> LiveConsumer:
        """Connect a WebSocket."""
        if len(self.active_connections) > 1:
            await websocket.close(code=1001, reason="Too many connections, try again later.")

        if True:  # TODO: Check API key for real
            await websocket.accept()
            self.active_connections.append(websocket)

            return LiveConsumer(client_id=client_id, api_key=api_key, status=LiveConnectionStatus.CONNECTED)
        else:
            await websocket.close(code=1008, reason="Invalid API key.")

    def disconnect(self, websocket: WebSocket, consumer: LiveConsumer) -> None:
        """Disconnect a WebSocket."""
        self.active_connections.remove(websocket)
        consumer.status = LiveConnectionStatus.DISCONNECTED


manager = ConnectionManager()

@router.websocket("")
async def websocket_endpoint(
    client_id: str, api_key: str, source_lang:str, websocket: WebSocket
) -> None:
    """Handle WebSocket connections."""
    consumer = await manager.connect(websocket, client_id=client_id, api_key=api_key)
    await websocket.send_text(f"Welcome back {consumer.client_id}!")

    try:
        while True:
            data = await websocket.receive_bytes()

            result = await asr.process_input(data=data, source_lang=source_lang)
            transcription, duration = result

            await websocket.send_text(transcription[0]["text"])
            consumer.usage += duration

    except WebSocketDisconnect:
        manager.disconnect(websocket, consumer=consumer)
