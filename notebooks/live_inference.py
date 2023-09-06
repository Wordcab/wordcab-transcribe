"""Test the live endpoint."""

import asyncio
import websockets

async def test_websocket_endpoint():
    uri = "ws://localhost:5001/api/v1/live"  # Replace with the actual WebSocket URL
    async with websockets.connect(uri) as websocket:
        try:
            while True:
                message = await websocket.recv()
                print(f"Received message: {message}")
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_websocket_endpoint())
