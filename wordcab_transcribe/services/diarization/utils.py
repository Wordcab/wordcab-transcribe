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

# BSD 3-Clause License
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# NME-SC clustering is based on the implementation from the paper
# https://arxiv.org/pdf/2003.02405.pdf and the implementation from
# https://github.com/tango4j/Auto-Tuning-Spectral-Clustering.

# Code from this module is heavily based on the NVIDIA NeMo implementation of
# the SpeakerClustering class and the associated classes and functions from
# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/utils/offline_clustering.py

"""Utils functions for the diarization service and modules."""

from typing import List, Optional, Tuple

import torch


def add_anchor_embeddings(
    embeddings: torch.Tensor, anchor_sample_n: int, anchor_spk_n: int, sigma: float
) -> torch.Tensor:
    """Add randomly generated synthetic embeddings to make eigenanalysis more stable.

    We refer to these embeddings as anchor embeddings.

    emb (Tensor):
        The input embedding from the embedding extractor.
    anchor_sample_n (int):
        Number of embedding samples per speaker.
        anchor_sample_n = 10 is recommended.
    anchor_spk_n (int):
        Number of speakers for synthetic embedding.
        anchor_spk_n = 3 is recommended.
    sigma (int):
        The amplitude of synthetic noise for each embedding vector.
        If the sigma value is too small, under-counting could happen.
        If the sigma value is too large, over-counting could happen.
        sigma = 50 is recommended.
    """
    emb_dim = embeddings.shape[1]
    std_org = torch.std(embeddings, dim=0)
    sigma = torch.tensor(sigma).to(embeddings.device)

    new_emb_list = []
    for _ in range(anchor_spk_n):
        emb_m = torch.tile(torch.randn(1, emb_dim), (anchor_sample_n, 1)).to(
            embeddings.device
        )
        emb_noise = torch.randn(anchor_sample_n, emb_dim).T.to(embeddings.device)
        emb_noise = torch.matmul(
            torch.diag(std_org),
            emb_noise / torch.max(torch.abs(emb_noise), dim=0)[0].unsqueeze(0),
        ).T
        emb_gen = emb_m + sigma * emb_noise
        new_emb_list.append(emb_gen)

    new_emb_list.append(embeddings)
    new_emb_np = torch.vstack(new_emb_list)

    return new_emb_np


def cosine_similarity(
    emb_a: torch.Tensor, emb_b: torch.Tensor, eps: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Calculate cosine similarities of the given two set of tensors.

    The output is an N by N matrix where N is the number of feature vectors.

    Args:
        emb_a (torch.Tensor): Matrix containing speaker representation vectors. (N x embedding_dim)
        emb_b (torch.Tensor): Matrix containing speaker representation vectors. (N x embedding_dim)
        eps (torch.Tensor): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Matrix containing cosine similarities. (N x N)
    """
    if emb_a.shape[0] == 1 or emb_b.shape[0] == 1:
        raise ValueError(
            f"Number of feature vectors should be greater than 1 but got {emb_a.shape} and {emb_b.shape}"
        )

    if eps is None:
        eps = torch.tensor(3.5e-4).to(emb_a.device)
    else:
        eps = eps.to(emb_a.device)

    a_norm = emb_a / (torch.norm(emb_a, dim=1).unsqueeze(1) + eps)
    b_norm = emb_b / (torch.norm(emb_b, dim=1).unsqueeze(1) + eps)

    cosine_similarity = torch.mm(a_norm, b_norm.transpose(0, 1)).fill_diagonal_(1)

    return cosine_similarity


def eigen_decompose(
    laplacian: torch.Tensor, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate eigenvalues and eigenvectors from the Laplacian matrix.

    Args:
        laplacian (torch.Tensor): Laplacian matrix
        device (str): Device to use for eigendecomposition.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing eigenvalues and eigenvectors.
    """
    laplacian = laplacian.float()

    if laplacian.device != device:
        laplacian = laplacian.to(device)

    lambdas, diffusion_map = torch.linalg.eigh(laplacian)

    return lambdas, diffusion_map


def eigen_value_sh(laplacian: torch.Tensor, device: str) -> torch.Tensor:
    """
    Calculate only eigenvalues from the Laplacian matrix.

    Args:
        laplacian (torch.Tensor): Laplacian matrix
        device (str): Device to use for eigendecomposition.

    Returns:
        torch.Tensor: Eigenvalues of the Laplacian matrix.
    """
    laplacian = laplacian.float().to(device)

    lambdas = torch.linalg.eigvalsh(laplacian)

    return lambdas


def estimate_number_of_speakers(
    affinity_matrix: torch.Tensor,
    max_num_speakers: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate the number of speakers using eigendecomposition on the Laplacian Matrix.

    Args:
        affinity_matrix (torch.Tensor): N by N affinity matrix
        max_num_speakers (int): Maximum number of clusters to consider for each session
        device (str): Device to use for eigendecomposition.

    Returns:
        num_of_spk (torch.Tensor):
            The estimated number of speakers
        lambdas (Tensor):
            The lambda values from eigendecomposition
        lambda_gap (Tensor):
            The gap between the lambda values from eigendecomposition
    """
    laplacian = get_laplacian(affinity_matrix)

    lambdas = eigen_value_sh(laplacian, device=device)
    lambdas = torch.sort(lambdas)[0]

    lambda_gap = get_lamda_gap_list(lambdas)

    number_of_speakers = (
        torch.argmax(lambda_gap[: min(max_num_speakers, lambda_gap.shape[0])]) + 1
    )

    return number_of_speakers, lambdas, lambda_gap


def get_euclidean_distance(
    spectral_embeddings_a: torch.Tensor,
    spectral_embeddings_b: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Calculate Euclidean distances from the given feature tensors.

    Args:
        spectral_embeddings_a (torch.Tensor):
            Matrix containing spectral embedding vectors from eigenvalue decomposition (N x embedding_dim).
        spectral_embeddings_b (torch.Tensor):
            Matrix containing spectral embedding vectors from eigenvalue decomposition (N x embedding_dim).
        device (str): Device to use for eigendecomposition.

    Returns:
        distance (torch.Tensor):
            Euclidean distance values of the two sets of spectral embedding vectors.
    """
    A = spectral_embeddings_a.to(device).unsqueeze(dim=1)
    B = spectral_embeddings_b.to(device).unsqueeze(dim=0)

    distance = (A - B) ** 2.0
    distance = distance.sum(dim=-1).squeeze()

    return distance


def get_affinity_graph_matrix(
    affinity_matrix_raw: torch.Tensor, p_value: int
) -> torch.Tensor:
    """
    Calculate a binarized graph matrix and symmetrize the binarized graph matrix.

    Args:
        affinity_matrix_raw (torch.Tensor): Matrix containing cosine similarities. (N x N)
        p_value (int): Number of nearest neighbors to consider.

    Returns:
        torch.Tensor: Matrix containing cosine similarities. (N x N)
    """
    X = (
        affinity_matrix_raw
        if p_value <= 0
        else get_k_neighbors_connections(affinity_matrix_raw, p_value)
    )

    symm_affinity_mat = 0.5 * (X + X.T)

    return symm_affinity_mat


def get_argmin_mapping_list(
    timestamps_in_scales: List[torch.Tensor],
) -> List[torch.Tensor]:
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
    segment_anchor_list = [
        torch.mean(timestamps_in_scales[scale_idx], dim=1) for scale_idx in scale_list
    ]

    base_scale_anchor = segment_anchor_list[max(scale_list)].view(-1, 1)

    session_scale_mapping_list = []
    for scale_idx in scale_list:
        current_scale_anchor = segment_anchor_list[scale_idx].view(1, -1)
        distance = torch.abs(current_scale_anchor - base_scale_anchor)
        argmin_mat = torch.argmin(distance, dim=1)
        session_scale_mapping_list.append(argmin_mat)

    return session_scale_mapping_list


def get_cosine_affinity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Get cosine affinity matrix.

    Calculate cosine similarity values among speaker embeddings then min-max normalize
    the affinity matrix.

    Args:
        embeddings (torch.Tensor):
            Matrix containing embedding vectors. emb variable should be float(FP32) type to make the data-type
            compatible with torch.mm operation for both CPU and GPU(CUDA).
            dimension: (Number of embedding vectors) x (embedding dimension)

    Returns:
        torch.Tensor: The normalized matrix containing cosine similarity values among the given embedding vectors,
            with the dimension: (Number of embedding vectors) x (Number of embedding vectors)
    """
    if embeddings.shape[0] == 1:
        return torch.tensor([[1]]).to(embeddings.device)

    else:
        if embeddings.dtype != torch.float32:
            embeddings = embeddings.float()

        cosine_affinity_matix = cosine_similarity(embeddings, embeddings)

        v_min, v_max = cosine_affinity_matix.min(), cosine_affinity_matix.max()
        normalized_cosine_affinity_matrix = (cosine_affinity_matix - v_min) / (
            v_max - v_min
        )

    return normalized_cosine_affinity_matrix


def get_k_neighbors_connections(
    affinity_matrix: torch.Tensor, p_value: int, mask_method: str = "binary"
) -> torch.Tensor:
    """
    Binarize top-p values for each row from the given affinity matrix.

    Args:
        affinity_matrix (torch.Tensor): A square matrix (tensor) containing normalized cosine similarity values
        p_value (int): The number of top values that are selected from each row.
        mask_method (str): The method that is used to manipulate the affinity matrix. The default method is "binary".

    Returns:
        binarized_affinity_mat (torch.Tensor):
            A binarized affinity matrix based on the given mask method.
    """
    binarized_affinity_matrix = torch.zeros_like(affinity_matrix).half()
    sorted_matrix = torch.argsort(affinity_matrix, dim=1, descending=True)[:, :p_value]
    binarized_affinity_matrix[
        sorted_matrix.T, torch.arange(affinity_matrix.shape[0])
    ] = (torch.ones(1).to(affinity_matrix.device).half())
    indices_row = sorted_matrix[:, :p_value].flatten()
    indices_col = torch.arange(affinity_matrix.shape[1]).repeat(p_value, 1).T.flatten()

    if mask_method == "binary" or mask_method is None:
        binarized_affinity_matrix[indices_row, indices_col] = (
            torch.ones(indices_row.shape[0]).to(affinity_matrix.device).half()
        )
    elif mask_method == "drop":
        binarized_affinity_matrix[indices_row, indices_col] = affinity_matrix[
            indices_row, indices_col
        ].half()
    elif mask_method == "sigmoid":
        binarized_affinity_matrix[indices_row, indices_col] = torch.sigmoid(
            affinity_matrix[indices_row, indices_col]
        ).half()
    else:
        raise ValueError(f"Unknown mask method: {mask_method}")

    return binarized_affinity_matrix


def get_lamda_gap_list(lambdas: torch.Tensor) -> torch.Tensor:
    """
    Calculate the gaps between lambda values from eigendecomposition.

    Args:
        lambdas (torch.Tensor): A tensor containing lambda values from eigendecomposition.

    Returns:
        torch.Tensor: A tensor containing the gaps between lambda values.
    """
    if torch.is_complex(lambdas):
        lambdas = torch.real(lambdas)

    return lambdas[1:] - lambdas[:-1]


def get_laplacian(X: torch.Tensor) -> torch.Tensor:
    """
    Calculate a laplacian matrix from an affinity matrix X.

    Args:
        X (torch.Tensor): A square matrix containing normalized cosine distance values.

    Returns:
        torch.Tensor: A laplacian matrix.
    """
    X.fill_diagonal_(0)

    D = torch.sum(torch.abs(X), dim=1)
    D = torch.diag_embed(D)

    return D - X


def get_minimum_connection(
    matrix: torch.Tensor, max_N: torch.Tensor, n_list: torch.Tensor, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate connections until fully connect all the nodes in the graph.

    If the graph is not fully connected, it might generate inaccurate results.

    Args:
        matrix (torch.Tensor): A square matrix containing normalized cosine distance values.
        max_N (torch.Tensor): The maximum number of connections to be generated.
        n_list (torch.Tensor): The list of number of connections to be generated.
        device (str): The device that is used to generate the connections.

    Returns:
        affinity_matrix (torch.Tensor): A square matrix containing normalized cosine distance values.
        p_value (torch.Tensor): The number of connections that are generated.
    """
    p_value = torch.tensor(1)
    affinity_matrix = get_affinity_graph_matrix(matrix, p_value)

    for p_value in n_list:
        fully_connected = is_graph_fully_connected(affinity_matrix, device)
        affinity_matrix = get_affinity_graph_matrix(matrix, p_value)

        if fully_connected or p_value > max_N:
            break

    return affinity_matrix, p_value


def get_the_largest_component(
    affinity_matrix: torch.Tensor, seg_index: int, device: str
) -> torch.Tensor:
    """
    Find the largest affinity_mat connected components for each given node.

    Args:
        affinity_matrix (torch.Tensor): A square matrix containing normalized cosine distance values.
        seg_index (int): The segment index that is targeted to be explored.
        device (str): The device that is used to generate the connections.

    Returns:
        connected_nodes (Tensor):
            A tensor containing booleans that indicate whether the node is connected.
    """
    num_of_segments = affinity_matrix.shape[0]

    connected_nodes = torch.zeros(num_of_segments, dtype=torch.bool).to(device)
    nodes_to_explore = torch.zeros(num_of_segments, dtype=torch.bool).to(device)

    nodes_to_explore[seg_index] = True
    nodes_to_explore = nodes_to_explore.to(device)

    for _ in range(num_of_segments):
        last_num_component = connected_nodes.sum()
        torch.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)

        if last_num_component >= connected_nodes.sum():
            break

        indices = (nodes_to_explore == torch.tensor(True)).nonzero().t().squeeze()

        if len(indices.size()) == 0:
            indices = indices.unsqueeze(0)

        for i in indices:
            neighbors = affinity_matrix[i].to(device)
            torch.logical_or(
                nodes_to_explore, neighbors.squeeze(0), out=nodes_to_explore
            )

    return connected_nodes


def is_graph_fully_connected(
    affinity_matrix: torch.Tensor, device: str
) -> torch.Tensor:
    """
    Check whether the given affinity matrix is a fully connected graph.

    Args:
        affinity_matrix (torch.Tensor): A square matrix (tensor) containing normalized cosine distance values
        device (str): The device that is used to run the model.

    Returns:
        bool: A boolean value indicating whether the graph is fully connected.
    """
    return (
        get_the_largest_component(affinity_matrix, 0, device).sum()
        == affinity_matrix.shape[0]
    )


def kmeans_plusplus_torch(
    X: torch.Tensor,
    n_clusters: int,
    random_state: int,
    device: str,
    n_local_trials: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose initial centroids for initializing k-means algorithm.

    The performance of k-means algorithm can vary significantly by the initial centroids. To alleviate
    this problem, k-means++ algorithm chooses initial centroids based on the probability
    proportional to the distance from the formally chosen centroids. The centroids
    selected by k-means++ algorithm improve the chance of getting more accurate and
    stable clustering results. The overall implementation of k-means++ algorithm is
    inspired by the numpy based k-means++ implementation in: https://github.com/scikit-learn/scikit-learn

    Originally, the implementation of the k-means++ algorithm in scikit-learn is based
    on the following research article:
    Arthur, David, and Sergei Vassilvitskii. k-means++: The advantages of careful
    seeding. Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete
    algorithms, Society for Industrial and Applied Mathematics (2007)

    Args:
        X (torch.Tensor):
            Matrix containing cosine similarity values among embedding vectors (N x N)
        n_clusters (int):
            Maximum number of speakers for estimating number of speakers. Shows stable performance under 20.
        random_state (int):
            Seed variable for setting up a random state.
        device (str)
            Torch device variable.
        n_local_trials (int):
            Number of trials for creating initial values of the center points.

    Returns:
        centers (torch.Tensor):
            The coordinates for center points that are used for initializing k-means algorithm.
        indices (torch.Tensor):
            The indices of the best candidate center points.
    """
    torch.manual_seed(random_state)
    X = X.to(device)
    n_samples, n_features = X.shape

    centers = torch.zeros(n_clusters, n_features, dtype=X.dtype)
    center_id = torch.randint(0, n_samples, (1,)).long()
    indices = torch.full(
        [
            n_clusters,
        ],
        -1,
        dtype=torch.int,
    )

    centers[0] = X[center_id].squeeze(0)
    indices[0] = center_id.squeeze(0)

    centers = centers.to(device)
    closest_dist_diff = centers[0, None].repeat(1, X.shape[0]).view(X.shape[0], -1) - X
    closest_dist_sq = closest_dist_diff.pow(2).sum(dim=1).unsqueeze(dim=0)
    current_pot = closest_dist_sq.sum()

    for c in range(1, n_clusters):
        rand_vals = torch.rand(n_local_trials) * current_pot.item()

        if len(closest_dist_sq.shape) > 1:
            torch_cumsum = torch.cumsum(closest_dist_sq, dim=1)[0]
        else:
            torch_cumsum = torch.cumsum(closest_dist_sq, dim=0)

        candidate_ids = torch.searchsorted(torch_cumsum, rand_vals.to(device))

        N_ci = candidate_ids.shape[0]
        distance_diff = X[candidate_ids].repeat(1, X.shape[0]).view(
            X.shape[0] * N_ci, -1
        ) - X.repeat(N_ci, 1)
        distance = distance_diff.pow(2).sum(dim=1).view(N_ci, -1)
        distance_to_candidates = torch.minimum(closest_dist_sq, distance)
        candidates_pot = distance_to_candidates.sum(dim=1)

        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices


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
