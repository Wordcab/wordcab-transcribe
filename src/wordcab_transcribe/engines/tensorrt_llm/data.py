import numpy as np
import torch

from wordcab_transcribe.engines.tensorrt_llm.audio import (
    convert_audio_tensor,
    pad_or_trim,
)


def stitch_speech_segments(start_ends, max_len=27.0, max_silent_region=None):
    speech_duration = [end - start for start, end in start_ends]

    stitched_speech_segments = []

    curr_seg = [0]
    curr_dur = speech_duration[0]
    idx = 1
    while idx < len(start_ends):
        if curr_dur + speech_duration[idx] > max_len:
            stitched_speech_segments.append([start_ends[_] for _ in curr_seg])
            curr_seg = [idx]
            curr_dur = speech_duration[idx]
        else:
            curr_dur += speech_duration[idx]
            curr_seg.append(idx)

        idx += 1

    stitched_speech_segments.append([start_ends[_] for _ in curr_seg])

    if max_silent_region is None:
        return stitched_speech_segments

    stitched_speech_segments_joined = []
    for segs in stitched_speech_segments:
        _segs = []
        curr_seg_start_time, curr_seg_end_time = segs[0]
        for i in range(1, len(segs)):
            if (segs[i][0] - curr_seg_end_time) >= max_silent_region:
                _segs.append((curr_seg_start_time, curr_seg_end_time))
                curr_seg_start_time = segs[i][0]

            curr_seg_end_time = segs[i][1]

        _segs.append((curr_seg_start_time, curr_seg_end_time))

        stitched_speech_segments_joined.append(_segs)

    return stitched_speech_segments_joined


class BasicSegmenter:
    """Basic segmenter that segments audio into chunks of max_seg_len seconds."""

    def __init__(self, max_seg_len=29.0, sampling_rate=16000):
        self.max_seg_len = max_seg_len
        self.sampling_rate = sampling_rate

    def __call__(self, audio_data=None):
        audio_duration = len(audio_data) / self.sampling_rate

        start_ends = []
        for i in range(0, int(audio_duration), int(self.max_seg_len)):
            start_ends.append([i, i + self.max_seg_len])

        start_ends[-1][1] = min(audio_duration, start_ends[-1][1])  # fix edge

        return start_ends, audio_data


class WhisperTRTDataLoader:
    """Data loader for WhisperTRT."""

    def __init__(
        self,
        device,
        tokenizer,
        speech_segmenter,
        dta_padding=3.0,
        without_timestamps=False,
        max_speech_len=29.0,
        max_initial_prompt_len=223,
        merge_chunks=True,
        use_dynamic_time_axis=False,
    ):
        """
        Initialize the WhisperDataLoader.

        Args:
            device: The device to use for processing.
            tokenizer: The tokenizer to use for encoding prompts.
            speech_segmenter: The speech segmenter to use for segmenting audio.
            dta_padding: The padding to use for dynamic time axis.
            without_timestamps: Whether to exclude timestamps from the prompts.
            max_speech_len: The maximum length of speech segments.
            max_initial_prompt_len: The maximum length of initial prompts.
            merge_chunks: Whether to merge chunks of audio.
            use_dynamic_time_axis: Whether to use dynamic time axis.
        """
        self.device = device
        self.tokenizer = tokenizer
        self.speech_segmenter = speech_segmenter
        self.basic_segmenter = BasicSegmenter(max_seg_len=max_speech_len)
        self.dta_padding = int(dta_padding * 16000)
        self.without_timestamps = without_timestamps
        self.max_speech_len = max_speech_len
        self.max_initial_prompt_len = max_initial_prompt_len
        self.use_dynamic_time_axis = use_dynamic_time_axis
        self.merge_chunks = merge_chunks

    def data_collate_fn(self, batch):
        """Collate a batch of data."""
        n_samples = 30 * 16000  # chunk length * sampling rate
        max_len = (
            min(max([_[3] for _ in batch]) + self.dta_padding, n_samples)
            if self.use_dynamic_time_axis
            else n_samples
        )

        signal_batch = torch.stack(
            [
                torch.from_numpy(pad_or_trim(_[0], length=max_len)).to(self.device)
                for _ in batch
            ]
        )
        seq_len = torch.tensor([_[3] for _ in batch]).to(self.device)

        prompt_batch = []
        initial_prompt_max_len = max([len(_[2]) for _ in batch])
        for _ in batch:
            prompt = (
                [self.tokenizer.sot_prev]
                + (initial_prompt_max_len - len(_[2])) * [self.tokenizer.silent_token]
                + _[2]
                + _[1]
                if initial_prompt_max_len
                else _[1]
            )
            prompt_batch.append(prompt)

        return (
            (signal_batch, prompt_batch, seq_len, [_[4] for _ in batch])
            if len(batch[0]) == 5
            else (signal_batch, prompt_batch, seq_len)
        )

    def get_segmented_audio_signal(
        self, start_ends, audio_signal, file_id, lang, task, initial_prompt, sr=16000
    ):
        """Segment the audio signal based on start and end times."""
        initial_prompt_tokens = (
            self.tokenizer.encode(" " + initial_prompt.strip())[
                -self.max_initial_prompt_len :
            ]
            if initial_prompt
            else []
        )
        prompt = self.tokenizer.sot_sequence(task=task, lang=lang) + (
            [self.tokenizer.no_timestamps]
            if self.without_timestamps
            else [self.tokenizer.timestamp_begin]
        )

        segmented_audio_signal = []
        if self.merge_chunks:
            for stitched_seg in stitch_speech_segments(
                start_ends, max_len=self.max_speech_len
            ):
                audio = np.concatenate(
                    [
                        audio_signal[int(st * sr) : int(et * sr)]
                        for st, et in stitched_seg
                    ]
                )
                seq_len = audio.shape[-1]
                seg_metadata = {
                    "file_id": file_id,
                    "start_time": stitched_seg[0][0],
                    "end_time": stitched_seg[-1][1],
                    "stitched_seg": stitched_seg,
                    "lang_code": lang,
                }
                segmented_audio_signal.append(
                    (audio, prompt, initial_prompt_tokens, seq_len, seg_metadata)
                )
        else:
            for st, et in start_ends:
                audio = audio_signal[int(st * sr) : int(et * sr)]
                seq_len = audio.shape[-1]
                segmented_audio_signal.append(
                    (
                        audio,
                        prompt,
                        initial_prompt_tokens,
                        seq_len,
                        {"file_id": file_id, "start_time": st, "end_time": et},
                    )
                )

        return segmented_audio_signal

    def get_data_loader(
        self,
        audio_data,
        lang_codes,
        tasks,
        initial_prompts,
        batch_size,
        use_vad,
    ):
        """Get the data loader for the given audio files."""
        segmented_audio_signal = []

        if all(isinstance(item, torch.Tensor) for item in audio_data):
            audio_data = [convert_audio_tensor(item) for item in audio_data]
        for file_id, (audio_signal, lang, task, initial_prompt) in enumerate(
            zip(audio_data, lang_codes, tasks, initial_prompts)
        ):
            start_ends, audio_signal = (
                self.speech_segmenter(audio_data=audio_signal)
                if use_vad
                else self.basic_segmenter(audio_data=audio_signal)
            )
            segmented_audio_signal.extend(
                self.get_segmented_audio_signal(
                    start_ends, audio_signal, file_id, lang, task, initial_prompt
                )
            )

            if not use_vad:
                while len(segmented_audio_signal) >= batch_size:
                    batch = segmented_audio_signal[:batch_size]
                    segmented_audio_signal = segmented_audio_signal[batch_size:]
                    yield self.data_collate_fn(batch)

        if use_vad:
            while len(segmented_audio_signal) >= batch_size:
                batch = segmented_audio_signal[:batch_size]
                segmented_audio_signal = segmented_audio_signal[batch_size:]
                yield self.data_collate_fn(batch)

        if segmented_audio_signal:
            yield self.data_collate_fn(segmented_audio_signal)

    def __call__(
        self,
        audio_data,
        lang_codes,
        tasks,
        initial_prompts,
        batch_size,
        use_vad,
    ):
        """Call the data loader with the given parameters."""
        return self.get_data_loader(
            audio_data,
            lang_codes,
            tasks,
            initial_prompts,
            batch_size,
            use_vad,
        )
