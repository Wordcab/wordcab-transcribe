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


def cosine_similarity(
    emb_a: torch.Tensor, emb_b: torch.Tensor, eps: torch.Tensor = torch.tensor(3.5e-4)
) -> torch.Tensor:
    """Calculate cosine similarities of the given two set of tensors.

    The output is an N by N matrix where N is the number of feature vectors.

    Args:
        emb_a (torch.Tensor): Matrix containing speaker representation vectors. (N x embedding_dim)
        emb_b (torch.Tensor): Matrix containing speaker representation vectors. (N x embedding_dim)

    Returns:
        torch.Tensor: Matrix containing cosine similarities. (N x N)
    """
    if emb_a.shape[0] == 1 or emb_b.shape[0] == 1:
        raise ValueError(f"Number of feature vectors should be greater than 1 but got {emb_a.shape} and {emb_b.shape}")

    a_norm = emb_a / (torch.norm(emb_a, dim=1).unsqueeze(1) + eps)
    b_norm = emb_b / (torch.norm(emb_b, dim=1).unsqueeze(1) + eps)

    cosine_similarity = torch.mm(a_norm, b_norm.transpose(0, 1)).fill_diagonal_(1)

    return cosine_similarity


def get_argmin_mapping_list(timestamps_in_scales: List[torch.Tensor]) -> List[torch.Tensor]:
    """Calculate the mapping between the base scale and other scales.

    A segment from a longer scale is repeatedly mapped to a segment from a shorter scale or the base scale.

    Args:
        timestamps_in_scales (list):
            List containing timestamp tensors for each scale.
            Each tensor has dimensions of (Number of base segments) x 2.

    Returns:
        session_scale_mapping_list (list):
            List containing argmin arrays indexed by scale index.
    """
    scale_list = list(range(len(timestamps_in_scales)))
    segment_anchor_list = [torch.mean(timestamps_in_scales[scale_idx], dim=1) for scale_idx in scale_list]

    base_scale_anchor = segment_anchor_list[max(scale_list)].view(-1, 1)

    session_scale_mapping_list = []
    for scale_idx in scale_list:
        current_scale_anchor = segment_anchor_list[scale_idx].view(1, -1)
        distance = torch.abs(current_scale_anchor - base_scale_anchor)
        argmin_mat = torch.argmin(distance, dim=1)
        session_scale_mapping_list.append(argmin_mat)

    return session_scale_mapping_list


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


def getEnhancedSpeakerCount(
    emb: torch.Tensor,
    random_test_count: int = 5,
    anchor_spk_n: int = 3,
    anchor_sample_n: int = 10,
    sigma: float = 50,
    cuda: bool = False,
) -> torch.Tensor:
    """
    Calculate the number of speakers using NME analysis with anchor embeddings. Add dummy speaker
    embedding vectors and run speaker counting multiple times to enhance the speaker counting accuracy
    for the short audio samples.

    Args:
        emb (Tensor):
            The input embedding from the embedding extractor.
        cuda (bool):
            Use cuda for the operations if cuda==True.
        random_test_count (int):
            Number of trials of the enhanced counting with randomness.
            The higher the count, the more accurate the enhanced counting is.
        anchor_spk_n (int):
            Number of speakers for synthetic embedding.
            anchor_spk_n = 3 is recommended.
        anchor_sample_n (int):
            Number of embedding samples per speaker.
            anchor_sample_n = 10 is recommended.
        sigma (float):
            The amplitude of synthetic noise for each embedding vector.
            If the sigma value is too small, under-counting could happen.
            If the sigma value is too large, over-counting could happen.
            sigma = 50 is recommended.

    Returns:
        comp_est_num_of_spk (Tensor):
            The estimated number of speakers. `anchor_spk_n` is subtracted from the estimated
            number of speakers to factor out the dummy speaker embedding vectors.
    """
    est_num_of_spk_list: List[int] = []
    for seed in range(random_test_count):
        torch.manual_seed(seed)
        emb_aug = addAnchorEmb(emb, anchor_sample_n, anchor_spk_n, sigma)
        mat = getCosAffinityMatrix(emb_aug)
        nmesc = NMESC(
            mat,
            max_num_speakers=emb.shape[0],
            max_rp_threshold=0.15,
            sparse_search=True,
            sparse_search_volume=10,
            fixed_thres=-1.0,
            nme_mat_size=300,
            cuda=cuda,
        )
        est_num_of_spk, _ = nmesc.forward()
        est_num_of_spk_list.append(est_num_of_spk.item())
    comp_est_num_of_spk = torch.tensor(max(torch.mode(torch.tensor(est_num_of_spk_list))[0].item() - anchor_spk_n, 1))
    return comp_est_num_of_spk
