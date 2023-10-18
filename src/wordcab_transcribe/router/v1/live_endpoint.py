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

from typing import List

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from loguru import logger

from wordcab_transcribe.config import settings
from wordcab_transcribe.dependencies import asr
from wordcab_transcribe.services.vad_service import VadService

router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self.max_live_connections = settings.max_live_connections
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Connect a WebSocket."""
        if len(self.active_connections) >= self.max_live_connections:
            await websocket.close(
                code=1001, reason="Too many connections, try again later."
            )
        else:
            await websocket.accept()
            self.active_connections.append(websocket)
            logger.info(
                f"Connected: {websocket}. Active connections:"
                f" {len(self.active_connections)}"
            )

    def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect a WebSocket."""
        self.active_connections.remove(websocket)
        logger.info(
            f"Disconnected: {websocket}. Active connections:"
            f" {len(self.active_connections)}"
        )


manager = ConnectionManager()

vad_service = VadService(live=True)


@router.websocket("")
async def websocket_endpoint(source_lang: str, websocket: WebSocket) -> None:
    """Handle WebSocket connections."""
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_bytes()

            speech_probability = vad_service.get_speech_probability(data)
            speech_probability = speech_probability if speech_probability else 0
            logger.info(f"Speech probability: {speech_probability}")

            if speech_probability > vad_service.options.threshold:
                try:
                    async for result in asr.process_input(
                        data=data, source_lang=source_lang
                    ):
                        await websocket.send_json(result)
                except Exception as e:
                    logger.error(f"Error processing data: {str(e)}")
                    raise HTTPException(detail=str(e)) from e
            else:
                continue

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await websocket.close(code=1000)  # Close WebSocket gracefully

    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        await websocket.close(code=1002, reason="Unexpected error.")
