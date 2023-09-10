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

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from wordcab_transcribe.dependencies import asr

router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Connect a WebSocket."""
        if len(self.active_connections) > 1:
            await websocket.close(
                code=1001, reason="Too many connections, try again later."
            )

        else:
            await websocket.accept()
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect a WebSocket."""
        self.active_connections.remove(websocket)


manager = ConnectionManager()


@router.websocket("")
async def websocket_endpoint(source_lang: str, websocket: WebSocket) -> None:
    """Handle WebSocket connections."""
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_bytes()

            result = await asr.process_input(data=data, source_lang=source_lang)

            full_transcript = ""
            for out in result[0]:
                print(out)
                if out["no_speech_prob"] < 0.5:
                    full_transcript += out["transcript"]

            if full_transcript != "":
                await websocket.send_text(full_transcript.strip())
            else:
                del result
                continue

    except WebSocketDisconnect:
        manager.disconnect(websocket)
