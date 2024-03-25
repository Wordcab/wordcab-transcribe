import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def convert_audio_tensor(audio_tensor, sr=16000, return_duration=False):
    audio_tensor = audio_tensor.cpu()
    audio_signal = audio_tensor.squeeze().numpy()
    audio_signal = audio_signal / torch.max(torch.abs(audio_signal))
    if sr != audio_tensor.shape[-1]:
        resampler = torchaudio.transforms.Resample(audio_tensor.shape[-1], sr)
        audio_signal = resampler(audio_tensor).squeeze().numpy()

    audio_signal = audio_signal.astype("float32")
    audio_duration = len(audio_signal) / sr

    if return_duration:
        return audio_signal, audio_duration
    else:
        return audio_signal


def pad_or_trim(array, length: int = 16000, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """

    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


class TorchSTFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    def forward(self, x):
        return torch.stft(
            x, self.n_fft, self.hop_length, window=self.window, return_complex=True
        )


class LogMelSpectogram(nn.Module):
    def __init__(self, n_mels=80, n_fft=400, hop_length=160, padding=0):
        super().__init__()

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.padding = padding

        mel_filters = np.load("assets/mel_filters.npz")
        mel_filters = torch.from_numpy(mel_filters[f"mel_{n_mels}"])
        self.register_buffer("mel_filters", mel_filters)
        self.stft = TorchSTFT(n_fft, hop_length)

    def get_seq_len(self, seq_len):
        seq_len = torch.floor(seq_len / self.hop_length)
        return seq_len.to(dtype=torch.long)

    @torch.no_grad()
    def forward(self, x, seq_len):
        seq_len = self.get_seq_len(seq_len.float())

        if self.padding > 0:
            x = F.pad(x, (0, self.padding))

        x = self.stft(x)

        x = x[..., :-1].abs() ** 2
        x = self.mel_filters @ x  # mels

        x = torch.clamp(x, min=1e-10).log10()  # log_mels
        x = torch.maximum(x, torch.amax(x, dim=(1, 2), keepdims=True) - 8.0)  # clip
        x = (x + 4.0) / 4.0  # scale

        return x, seq_len
