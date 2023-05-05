# adapted from https://github.com/Softcatala/whisper-ctranslate2/

"""Live transcription testing."""

import asyncio
import json

import numpy as np
import websockets


config = {
    "sample_rate": 16000,
    "block_size": 30,
    "vocals": [50, 1000],
    "end_blocks": 33 * 3,
    "flush_blocks": 33 * 8,
}

config["duration"] = config["block_size"] / 1000
config["chunk_size"] = int(config["sample_rate"] * config["duration"])

try:
    import sounddevice as sd

    sounddevice_available = True

except Exception as e:
    sounddevice_available = False
    sounddevice_exception = e


class Live:
    """Live transcription class."""

    def __init__(
        self,
        threshold: float,
        config: dict,
    ):
        """Initialize live transcription class."""
        self.threshold = threshold
        self.config = config
        self.running = True
        self.waiting = 0
        self.prevblock = self.buffer = np.zeros((0, 1))
        self.speaking = False
        self.blocks_speaking = 0
        self.buffers_to_process = []

    @staticmethod
    def is_available():
        """Check if sounddevice is available."""
        return sounddevice_available

    @staticmethod
    def force_not_available_exception():
        """Force sounddevice not available exception."""
        raise (sounddevice_exception)

    def _is_there_voice(self, indata, frames):
        """Check if there is voice in the audio."""
        freq = (
            np.argmax(np.abs(np.fft.rfft(indata[:, 0])))
            * self.config["sample_rate"]
            / frames
        )
        volume = np.sqrt(np.mean(indata**2))

        return (
            volume > self.threshold
            and self.config["vocals"][0] <= freq <= self.config["vocals"][1]
        )

    def _save_to_process(self):
        """Save audio to process."""
        self.buffers_to_process.append(self.buffer.copy())
        self.buffer = np.zeros((0, 1))
        self.speaking = False

    def callback(self, indata, frames, _time, status):
        """Callback function."""
        if not any(indata):
            return

        voice = self._is_there_voice(indata, frames)

        # Silence and no nobody has started speaking
        if not voice and not self.speaking:
            return

        if voice:  # User speaking
            print(".", end="", flush=True)
            if self.waiting < 1:
                self.buffer = self.prevblock.copy()

            self.buffer = np.concatenate((self.buffer, indata))
            self.waiting = self.config["end_blocks"]

            if not self.speaking:
                self.blocks_speaking = self.config["flush_blocks"]

            self.speaking = True
        else:  # Silence after user has spoken
            self.waiting -= 1
            if self.waiting < 1:
                self._save_to_process()
                return
            else:
                self.buffer = np.concatenate((self.buffer, indata))

        self.blocks_speaking -= 1
        # User spoken for a long time and we need to flush
        if self.blocks_speaking < 1:
            self._save_to_process()

    def process(self):
        """Process audio."""
        if len(self.buffers_to_process) > 0:
            _buffer = self.buffers_to_process.pop()
            return _buffer


async def send_audio(websocket, live):
    """Send audio to the server."""
    while True:
        _buffer = live.process()
        if _buffer is not None:
            audio_bytes = _buffer.astype(np.int16)
            audio_bytes_list = audio_bytes.tolist()
            data = {"audio_array": audio_bytes_list, "source_lang": "en"}
            await websocket.send(json.dumps(data))
        else:
            await asyncio.sleep(0.1)  # Avoid high CPU usage


async def test_websocket_transcription():
    """Test websocket transcription."""
    uri = "ws://0.0.0.0:5001/api/v1/live"

    try:
        async with websockets.connect(uri) as websocket:
            # Send the source language to the server
            initial_data = {"source_lang": "en"}
            await websocket.send(json.dumps(initial_data))

            live = Live(threshold=0.2, config=config)
            # Record audio from the microphone and send it to the server
            with sd.InputStream(
                channels=1,
                callback=live.callback,
                blocksize=int(config["sample_rate"] * config["block_size"] / 1000),
                samplerate=config["sample_rate"],
            ):
                print("Recording... Press Ctrl+C to stop.")
                await send_audio(websocket, live)
    except Exception as e:
        print(f"Error connecting to the server. Is it running? Error: {e}")
        return


loop = asyncio.get_event_loop()
audio_queue = asyncio.Queue()
loop.run_until_complete(test_websocket_transcription())
