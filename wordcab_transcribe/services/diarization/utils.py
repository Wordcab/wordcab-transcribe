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
"""Utils functions for the diarization service and modules."""

from typing import List, Tuple

import torch


def segmentation_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function used by the dataloader of the SegmentationModule.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): List of audio segments and their lengths.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of the audio segments and their lengths.
    """
    _, audio_lengths = zip(*batch)

    if not audio_lengths[0]:
        return None, None

    fixed_length = int(max(audio_lengths))

    audio_signal, new_audio_lengths = [], []
    for sig, sig_len in batch:
        sig_len = sig_len.item()
        chunck_len = sig_len - fixed_length

        if chunck_len < 0:
            repeat = fixed_length // sig_len
            rem = fixed_length % sig_len
            sub = sig[-rem:] if rem > 0 else torch.tensor([])
            rep_sig = torch.cat(repeat * [sig])
            sig = torch.cat((rep_sig, sub))
        new_audio_lengths.append(torch.tensor(fixed_length))

        audio_signal.append(sig)

    audio_signal = torch.stack(audio_signal)
    audio_lengths = torch.stack(new_audio_lengths)

    return audio_signal, audio_lengths