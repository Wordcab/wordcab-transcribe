"""Test the live endpoint."""

import asyncio
import websockets

async def test_websocket_endpoint():
    uri = "ws://localhost:5001/api/v1/live?source_lang=en"  # Replace with the actual WebSocket URL
    async with websockets.connect(uri) as websocket:
        try:
            with open("src/wordcab_transcribe/assets/warmup_sample.wav", "rb") as audio_file:
                await websocket.send(audio_file.read())
                print("Sent audio file")
            while True:
                message = await websocket.recv()
                print(f"Received message: {message}")
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_websocket_endpoint())
