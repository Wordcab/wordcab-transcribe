import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from faster_whisper import WhisperModel


num_workers = 4
model = WhisperModel("large-v2", device="cuda", num_workers=num_workers, cpu_threads=2)

files = [
    # "data/nissan_sample_5.mp3",
    # "data/test.wav",
    "data/GMT20230619_150026_Recording.mp3",
    # "data/2023-06-12 Rvampp Session (Audio).m4a",
]


def transcribe_file(file_path):
    start_time = time.time()

    segments, _ = model.transcribe(file_path, language="en", word_timestamps=True)
    segments = list(segments)

    end_time = time.time()
    
    return segments, end_time - start_time


with ThreadPoolExecutor(num_workers) as executor:
    futures = [executor.submit(transcribe_file, file) for file in files]

    for future in as_completed(futures):
        result, time_taken = future.result()
        path = files[futures.index(future)]
        print(
            "Transcription for %s:%s (Time taken: %.2f seconds)"
            % (path, "".join(segment.text for segment in result), time_taken)
        )