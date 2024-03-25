from abc import ABC, abstractmethod

import numpy as np
import torch


class VADBaseClass(ABC):
    def __init__(self, sampling_rate=16000):
        self.sampling_rate = sampling_rate

    @abstractmethod
    def update_params(self, params):
        pass

    @abstractmethod
    def __call__(self, audio_signal, batch_size=4):
        pass


class FrameVAD(VADBaseClass):
    def __init__(
        self,
        device=None,
        chunk_size=15.0,
        margin_size=1.0,
        frame_size=0.02,
        batch_size=4,
        sampling_rate=16000,
    ):
        super().__init__(sampling_rate=sampling_rate)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        if self.device == "cpu":
            # This is a JIT Scripted model of Nvidia's NeMo Framewise Marblenet Model: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_frame_marblenet
            self.vad_pp = torch.jit.load("assets/vad_pp_cpu.ts").to(self.device)
            self.vad_model = torch.jit.load("assets/frame_vad_model_cpu.ts").to(
                self.device
            )
        else:
            self.vad_pp = torch.jit.load("assets/vad_pp_gpu.ts").to(self.device)
            self.vad_model = torch.jit.load("assets/frame_vad_model_gpu.ts").to(
                self.device
            )

        self.vad_pp.eval()
        self.vad_model.eval()

        self.batch_size = batch_size
        self.frame_size = frame_size
        self.chunk_size = chunk_size
        self.margin_size = margin_size

        self._init_params()

    def _init_params(self):
        self.signal_chunk_len = int(self.chunk_size * self.sampling_rate)
        self.signal_stride = int(
            self.signal_chunk_len - 2 * int(self.margin_size * self.sampling_rate)
        )

        self.margin_logit_len = int(self.margin_size / self.frame_size)
        self.signal_to_logit_len = int(self.frame_size * self.sampling_rate)

        self.vad_pp.to(self.device)
        self.vad_model.to(self.device)

    def update_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)

        self._init_params()

    def prepare_input_batch(self, audio_signal):
        input_signal = []
        input_signal_length = []
        for s_idx in range(0, len(audio_signal), self.signal_stride):
            _signal = audio_signal[s_idx : s_idx + self.signal_chunk_len]
            _signal_len = len(_signal)
            input_signal.append(_signal)
            input_signal_length.append(_signal_len)

            if _signal_len < self.signal_chunk_len:
                input_signal[-1] = np.pad(
                    input_signal[-1], (0, self.signal_chunk_len - _signal_len)
                )
                break

        return input_signal, input_signal_length

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def forward(self, input_signal, input_signal_length):
        all_logits = []
        for s_idx in range(0, len(input_signal), self.batch_size):
            input_signal_pt = torch.stack(
                [
                    torch.tensor(_, device=self.device)
                    for _ in input_signal[s_idx : s_idx + self.batch_size]
                ]
            )
            input_signal_length_pt = torch.tensor(
                input_signal_length[s_idx : s_idx + self.batch_size], device=self.device
            )

            x, x_len = self.vad_pp(input_signal_pt, input_signal_length_pt)
            logits = self.vad_model(x, x_len)

            for _logits, _len in zip(logits, input_signal_length_pt):
                all_logits.append(_logits[: int(_len / self.signal_to_logit_len)])

        if len(all_logits) > 1 and self.margin_logit_len > 0:
            all_logits[0] = all_logits[0][: -self.margin_logit_len]
            all_logits[-1] = all_logits[-1][self.margin_logit_len :]

            for i in range(1, len(all_logits) - 1):
                all_logits[i] = all_logits[i][
                    self.margin_logit_len : -self.margin_logit_len
                ]

        all_logits = torch.concatenate(all_logits)
        all_logits = torch.softmax(all_logits, dim=-1)

        return all_logits[:, 1].detach().cpu().numpy()

    def __call__(self, audio_signal):
        audio_duration = len(audio_signal) / self.sampling_rate

        input_signal, input_signal_length = self.prepare_input_batch(audio_signal)
        speech_probs = self.forward(input_signal, input_signal_length)

        vad_times = []
        for idx, prob in enumerate(speech_probs):
            s_time = idx * self.frame_size
            e_time = min(audio_duration, (idx + 1) * self.frame_size)

            if s_time >= e_time:
                break

            vad_times.append([prob, s_time, e_time])

        return np.array(vad_times)


class SegmentVAD(VADBaseClass):
    def __init__(
        self,
        device=None,
        win_len=0.32,
        win_step=0.08,
        batch_size=512,
        sampling_rate=16000,
    ):
        super().__init__(sampling_rate=sampling_rate)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        if self.device == "cpu":
            # This is a JIT Scripted model of Nvidia's NeMo Marblenet Model: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_marblenet
            self.vad_pp = torch.jit.load("assets/vad_pp_cpu.ts").to(self.device)
            self.vad_model = torch.jit.load("assets/seg_vad_model_cpu.ts").to(
                self.device
            )
        else:
            self.vad_pp = torch.jit.load("assets/vad_pp_gpu.ts").to(self.device)
            self.vad_model = torch.jit.load("assets/seg_vad_model_gpu.ts").to(
                self.device
            )

        self.vad_pp = torch.jit.load("assets/vad_pp.ts")
        self.vad_model = torch.jit.load("assets/segment_vad_model.ts")

        self.vad_model.eval()
        self.vad_model.to(self.device)

        self.vad_pp.eval()
        self.vad_pp.to(self.device)

        self.batch_size = batch_size
        self.win_len = win_len
        self.win_step = win_step

        self._init_params()

    def _init_params(self):
        self.signal_win_len = int(self.win_len * self.sampling_rate)
        self.signal_win_step = int(self.win_step * self.sampling_rate)

    def update_params(self, params: dict):
        for key, value in params.items():
            setattr(self, key, value)

        self._init_params()

    def prepare_input_batch(self, audio_signal):
        num_chunks = (
            self.signal_win_len // 2 + len(audio_signal)
        ) // self.signal_win_step
        if (
            num_chunks
            < (self.signal_win_len // 2 + len(audio_signal)) / self.signal_win_step
        ):
            num_chunks += 1

        input_signal = np.zeros((num_chunks, self.signal_win_len), dtype=np.float32)
        input_signal_length = np.zeros(num_chunks, dtype=np.int64)

        chunk_idx = 0
        for idx in range(
            -1 * self.signal_win_len // 2, len(audio_signal), self.signal_win_step
        ):
            s_idx = max(idx, 0)
            e_idx = min(idx + self.signal_win_len, len(audio_signal))
            input_signal[chunk_idx][: e_idx - s_idx] = audio_signal[s_idx:e_idx]
            input_signal_length[chunk_idx] = e_idx - s_idx
            chunk_idx += 1

        return input_signal, input_signal_length

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def forward(self, input_signal, input_signal_length):
        x, x_len = self.vad_pp(
            torch.Tensor(input_signal).to(self.device),
            torch.Tensor(input_signal_length).to(self.device),
        )
        logits = self.vad_model(x, x_len)
        logits = torch.softmax(logits, dim=-1)
        return logits[:, 1].detach().cpu().numpy()

    def __call__(self, audio_signal):
        audio_duration = len(audio_signal) / self.sampling_rate

        input_signal, input_signal_length = self.prepare_input_batch(audio_signal)

        speech_probs = np.zeros(len(input_signal))
        for s_idx in range(0, len(input_signal), self.batch_size):
            speech_probs[s_idx : s_idx + self.batch_size] = self.forward(
                input_signal=input_signal[s_idx : s_idx + self.batch_size],
                input_signal_length=input_signal_length[
                    s_idx : s_idx + self.batch_size
                ],
            )

        vad_times = []
        for idx, prob in enumerate(speech_probs):
            s_time = max(0, (idx - 0.5) * self.win_step)
            e_time = min(audio_duration, (idx + 0.5) * self.win_step)
            vad_times.append([prob, s_time, e_time])

        return np.array(vad_times)


class SpeechSegmenter:
    def __init__(
        self,
        vad_model=None,
        device=None,
        frame_size=0.02,
        min_seg_len=0.08,
        max_seg_len=29.0,
        max_silent_region=0.6,
        padding=0.2,
        eos_thresh=0.3,
        bos_thresh=0.3,
        cut_factor=2,
        sampling_rate=16000,
    ):
        if vad_model is None:
            vad_model = FrameVAD(device=device)

        self.vad_model = vad_model

        self.sampling_rate = sampling_rate
        self.padding = padding
        self.frame_size = frame_size
        self.min_seg_len = min_seg_len
        self.max_seg_len = max_seg_len
        self.max_silent_region = max_silent_region

        self.eos_thresh = eos_thresh
        self.bos_thresh = bos_thresh

        self.cut_factor = cut_factor
        self.cut_idx = int(self.max_seg_len / (self.cut_factor * self.frame_size))
        self.max_idx_in_seg = self.cut_factor * self.cut_idx

    def update_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)

        self.cut_idx = int(self.max_seg_len / (self.cut_factor * self.frame_size))
        self.max_idx_in_seg = self.cut_factor * self.cut_idx

    def update_vad_model_params(self, params):
        self.vad_model.update_params(params=params)

    def okay_to_merge(self, speech_probs, last_seg, curr_seg):
        conditions = [
            (speech_probs[curr_seg["start"]][1] - speech_probs[last_seg["end"]][2])
            < self.max_silent_region,
            (speech_probs[curr_seg["end"]][2] - speech_probs[last_seg["start"]][1])
            <= self.max_seg_len,
        ]

        return all(conditions)

    def get_speech_segments(self, speech_probs):
        speech_flag, start_idx = False, 0
        speech_segments = []
        for idx, (speech_prob, _st, _et) in enumerate(speech_probs):
            if speech_flag:
                if speech_prob < self.eos_thresh:
                    speech_flag = False
                    curr_seg = {"start": start_idx, "end": idx - 1}

                    if len(speech_segments) and self.okay_to_merge(
                        speech_probs, speech_segments[-1], curr_seg
                    ):
                        speech_segments[-1]["end"] = curr_seg["end"]
                    else:
                        speech_segments.append(curr_seg)

            elif speech_prob >= self.bos_thresh:
                speech_flag = True
                start_idx = idx

        if speech_flag:
            curr_seg = {"start": start_idx, "end": len(speech_probs) - 1}

            if len(speech_segments) and self.okay_to_merge(
                speech_probs, speech_segments[-1], curr_seg
            ):
                speech_segments[-1]["end"] = curr_seg["end"]
            else:
                speech_segments.append(curr_seg)

        speech_segments = [
            _
            for _ in speech_segments
            if (speech_probs[_["end"]][2] - speech_probs[_["start"]][1])
            > self.min_seg_len
        ]

        start_ends = []
        for _ in speech_segments:
            first_idx = len(start_ends)
            start_idx, end_idx = _["start"], _["end"]
            while (end_idx - start_idx) > self.max_idx_in_seg:
                _start_idx = int(start_idx + self.cut_idx)
                _end_idx = int(min(end_idx, start_idx + self.max_idx_in_seg))

                new_end_idx = _start_idx + np.argmin(
                    speech_probs[_start_idx:_end_idx, 0]
                )
                start_ends.append(
                    [speech_probs[start_idx][1], speech_probs[new_end_idx][2]]
                )
                start_idx = new_end_idx + 1

            start_ends.append(
                [speech_probs[start_idx][1], speech_probs[end_idx][2] + self.padding]
            )
            start_ends[first_idx][0] = start_ends[first_idx][0] - self.padding

        return start_ends

    def __call__(self, audio_data=None):
        if audio_data is not None:
            if isinstance(audio_data, np.ndarray):
                audio_signal = audio_data
                audio_duration = len(audio_signal) / self.sampling_rate
            elif isinstance(audio_data, torch.Tensor):
                audio_tensor = audio_data
                audio_signal = audio_tensor.squeeze().cpu().numpy()
                audio_duration = len(audio_signal) / self.sampling_rate
            else:
                raise ValueError("`audio_data` must be a numpy array or torch tensor.")
        else:
            raise ValueError("`audio_data` must be a numpy array or torch tensor.")

        speech_probs = self.vad_model(audio_signal)
        start_ends = self.get_speech_segments(speech_probs)

        if len(start_ends) == 0:
            start_ends = [[0.0, self.max_seg_len]]  # Quick fix for silent audio.

        start_ends[0][0] = max(0.0, start_ends[0][0])  # fix edges
        start_ends[-1][1] = min(audio_duration, start_ends[-1][1])  # fix edges

        return start_ends, audio_signal
