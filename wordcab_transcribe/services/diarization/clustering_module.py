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

"""Clustering module for the diarization service."""

from typing import Dict, List, Tuple

import torch
from nemo.collections.asr.parts.utils.offline_clustering import SpeakerClustering

from wordcab_transcribe.services.diarization.models import MultiscaleEmbeddingsAndTimestamps
from wordcab_transcribe.services.diarization.utils import (
    cosine_similarity,
    get_argmin_mapping_list,
)


# https://github.com/NVIDIA/NeMo/blob/4d78737f612964ff9cce2120bb28ec94f874ab07/nemo/collections/asr/parts/utils/offline_clustering.py#L850
class NMESC:
    """
    Normalized Maximum Eigengap based Spectral Clustering (NME-SC)
    uses Eigengap analysis to get an estimated p-value for
    affinity binarization and an estimated number of speakers.

    p_value (also referred to as p_neighbors) is for taking
    top p number of affinity values and convert those to 1 while
    convert the rest of values to 0.

    p_value can be also tuned on a development set without performing
    NME-analysis. Fixing p_value brings about significantly faster clustering
    speed, but the performance is limited to the development set.

    References:
        Tae Jin Park et al., Auto-Tuning Spectral Clustering for Speaker Diarization
        Using Normalized Maximum Eigengap, IEEE Signal Processing Letters 27 (2019),
        https://arxiv.org/abs/2003.02405

    Args:
        Please refer to def __init__().

    Methods:
        forward():
            Performs NME-analysis to estimate p_value and the number of speakers
        subsample_affinity_matrix(nme_mat_size):
            Subsamples the number of speakers to reduce the computational load
        get_p_value_list():
            Generates a list containing p-values that need to be examined.
        get_eigengap_ratio(p_neighbors):
            Calculates g_p, which is a ratio between p_neighbors and the maximum eigengap
        getLamdaGaplist(lambdas):
            Calculates lambda gap values from an array contains lambda values
        estimateNumofSpeakers(affinity_mat):
            Estimates the number of speakers using lambda gap list
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        max_num_speakers: int = 10,
        max_rp_threshold: float = 0.15,
        sparse_search: bool = True,
        sparse_search_volume: int = 30,
        nme_mat_size: int = 512,
        use_subsampling_for_nme: bool = True,
        fixed_thres: float = -1.0,
        maj_vote_spk_count: bool = False,
        parallelism: bool = True,
        cuda: bool = False,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            matrix (Tensor):
                Cosine similarity matrix calculated from the provided speaker embeddings.
            max_num_speakers (int):
                Maximum number of speakers for estimating number of speakers.
                Shows stable performance under 20.
            max_rp_threshold (float):
                Limits the range of parameter search.
                Clustering performance can vary depending on this range.
                Default is 0.25.
            sparse_search (bool):
                To increase the speed of parameter estimation, sparse_search=True
                limits the number of p_values we search.
            sparse_search_volume (int):
                Number of p_values we search during NME analysis.
                Default is 30. The lower the value, the faster NME-analysis becomes.
                However, a value lower than 20 might cause a poor parameter estimation.
            nme_mat_size (int):
                Targeted size of matrix for NME analysis.
            use_subsampling_for_nme (bool):
                Use subsampling to reduce the calculational complexity.
                Default is True.
            fixed_thres (float or None):
                A fixed threshold which can be used instead of estimating the
                threshold with NME analysis. If fixed_thres is float,
                it skips the NME analysis part.
            maj_vote_spk_count (bool):
                If True, take a majority vote on all p-values in the given range to estimate the number of speakers.
                The majority voting may contribute to surpress overcounting of the speakers and improve speaker
                counting accuracy.
            parallelism (bool):
                If True, turn on parallelism based on torch.jit.script library.
            cuda (bool):
                Use cuda for Eigen decomposition if cuda=True.
            device (torch.device):
                Torch device variable

        """
        self.max_num_speakers: int = max_num_speakers
        self.max_rp_threshold: float = max_rp_threshold
        self.use_subsampling_for_nme: bool = use_subsampling_for_nme
        self.nme_mat_size: int = nme_mat_size
        self.sparse_search: bool = sparse_search
        self.sparse_search_volume: int = sparse_search_volume
        self.min_p_value = torch.tensor(2)
        self.fixed_thres: float = fixed_thres
        self.eps = 1e-10
        self.max_N = torch.tensor(0)
        self.mat: torch.Tensor = matrix
        self.p_value_list: torch.Tensor = self.min_p_value.unsqueeze(0)
        self.cuda: bool = cuda
        self.device: torch.device = device
        self.maj_vote_spk_count: bool = maj_vote_spk_count
        self.parallelism: bool = parallelism

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Subsample the input matrix to reduce the computational load.

        Returns:
            est_num_of_spk (Tensor):
                Estimated number of speakers from NMESC approach
            p_hat_value (Tensor):
                Estimated p-value (determines how many neighboring values to be selected)
        """
        if self.use_subsampling_for_nme:
            subsample_ratio = self.subsample_affinity_matrix(self.nme_mat_size)
        else:
            subsample_ratio = torch.tensor(1)

        # Scans p_values and find a p_value that generates the smallest g_p value.
        results: List[torch.Tensor] = []
        est_spk_n_dict: Dict[int, torch.Tensor] = {}
        self.p_value_list = self.get_p_value_list()
        p_volume = self.p_value_list.shape[0]
        eig_ratio_list = torch.zeros(p_volume,)
        est_num_of_spk_list = torch.zeros(p_volume,)

        if self.parallelism:
            futures: List[torch.jit.Future[torch.Tensor]] = []
            for p_idx, p_value in enumerate(self.p_value_list):
                futures.append(torch.jit.fork(self.get_eigengap_ratio, p_value))
            for future in futures:
                results.append(torch.jit.wait(future))

        else:
            for p_idx, p_value in enumerate(self.p_value_list):
                results.append(self.get_eigengap_ratio(p_value))

        # Retrieve the eigen analysis results
        for p_idx, p_value in enumerate(self.p_value_list):
            output = results[p_idx]
            g_p, est_num_of_spk = output[0], output[1].int()
            eig_ratio_list[p_idx] = g_p
            est_spk_n_dict[p_value.item()] = est_num_of_spk
            est_num_of_spk_list[p_idx] = est_num_of_spk

        index_nn = torch.argmin(eig_ratio_list)
        rp_p_value = self.p_value_list[index_nn]
        affinity_mat = getAffinityGraphMat(self.mat, rp_p_value)

        # Checks whether the affinity graph is fully connected.
        # If not, it adds a minimum number of connections to make it fully connected.
        if not isGraphFullyConnected(affinity_mat, device=self.device):
            affinity_mat, rp_p_value = getMinimumConnection(
                self.mat, self.max_N, self.p_value_list, device=self.device
            )

        p_hat_value = (subsample_ratio * rp_p_value).type(torch.int)
        if self.maj_vote_spk_count:
            est_num_of_spk = torch.mode(torch.tensor(est_num_of_spk_list))[0]
        else:
            est_num_of_spk = est_spk_n_dict[rp_p_value.item()]
        return est_num_of_spk, p_hat_value

    def subsample_affinity_matrix(self, nme_mat_size: int) -> torch.Tensor:
        """Perform subsampling of affinity matrix.

        This subsampling is for calculational complexity, not for performance.
        The smaller nme_mat_size is,
            - the bigger the chance of missing a speaker.
            - the faster p-value estimation speed (based on eigen decomposition).

        The recommended nme_mat_size is 250~750.
        However, if there are speakers who speak for very short period of time in the recording,
        this subsampling might make the system miss underrepresented speakers.
        Use this variable with caution.

        Args:
            nme_mat_size (int):
                The targeted matrix size

        Returns:
            subsample_ratio (float):
                The ratio between nme_mat_size and the original matrix size
        """
        subsample_ratio = torch.max(torch.tensor(1), torch.tensor(self.mat.shape[0] / nme_mat_size)).type(torch.int)
        self.mat = self.mat[:: subsample_ratio.item(), :: subsample_ratio.item()]

        return subsample_ratio

    def get_eigengap_ratio(self, p_neighbors: int) -> torch.Tensor:
        """
        For a given p_neighbors value, calculate g_p, which is a ratio between p_neighbors and the
        maximum eigengap values.
        References:
            Tae Jin Park et al., Auto-Tuning Spectral Clustering for Speaker Diarization Using
            Normalized Maximum Eigengap, IEEE Signal Processing Letters 27 (2019),
            https://arxiv.org/abs/2003.02405

        Args:
            p_neighbors (int):
                Determines how many binary graph connections we want to keep for each row.

        Returns:
            est_num_of_spk (int):
                Estimated number of speakers
            g_p (float):
                The ratio between p_neighbors value and the maximum eigen gap value.
        """
        affinity_mat = getAffinityGraphMat(self.mat, p_neighbors)
        est_num_of_spk, lambdas, lambda_gap_list = estimateNumofSpeakers(
            affinity_mat, self.max_num_speakers, self.cuda
        )
        arg_sorted_idx = torch.argsort(lambda_gap_list[: self.max_num_speakers], descending=True)
        max_key = arg_sorted_idx[0]
        max_eig_gap = lambda_gap_list[max_key] / (torch.max(lambdas).item() + self.eps)
        g_p = (p_neighbors / self.mat.shape[0]) / (max_eig_gap + self.eps)

        return torch.stack([g_p, est_num_of_spk])

    def get_p_value_list(self) -> torch.Tensor:
        """Generates a p-value (p_neighbour) list for searching.

        p_value_list must include 2 (min_p_value) since at least one neighboring 
        segment should be selected other than itself.

        If fixed_thres value is specified, then only one p-value is specified.
        If fixed_thres is not provided, multiple p-values are searched.
            If sparse_search is True:
                - Limit the number of p-values to be searched to sparse_search_volume.
                - N should be at least 2 to include a number greater than 1.
            If sparse_search is False:
                - Scan all the p_values from 1 to max_N
                - If sparse_search is False, NMESC analysis could take more time compared to sparse_search = True.

        Returns:
            p_value_list (Tensor):
                Tensor containing the p_values to be searched.
        """
        if self.fixed_thres is not None and self.fixed_thres > 0.0:
            self.max_N = torch.max(
                torch.floor(torch.tensor(self.mat.shape[0] * self.fixed_thres)).type(torch.int), self.min_p_value
            )
            p_value_list = self.max_N.unsqueeze(0).int()
        else:
            self.max_N = torch.max(
                torch.floor(torch.tensor(self.mat.shape[0] * self.max_rp_threshold)).type(torch.int), self.min_p_value
            )
            if self.sparse_search:
                search_volume = torch.min(self.max_N, torch.tensor(self.sparse_search_volume).type(torch.int))
                # search at least two values
                N = torch.max(search_volume, torch.tensor(2))
                # avoid repeating values by limiting the step size
                steps = min(self.max_N, N)
                p_value_list = torch.linspace(start=1, end=self.max_N, steps=steps).type(torch.int)
            else:
                p_value_list = torch.arange(1, self.max_N + 1)

        if p_value_list.shape[0] == 0:
            raise ValueError("p_value_list should not be empty.")

        return p_value_list


class SpeakerClustering(torch.nn.Module):
    """Clustering module for speaker diarization."""
    def __init__(
        self,
        device: str,
        min_samples_for_nmesc: int = 6,
        nme_mat_size: int = 512,
        sparse_search: bool = True,
        maj_vote_spk_count: bool = False,
    ):
        """
        Clustering method for speaker diarization based on cosine similarity.

        Args:
            device (str): Device to use for inference. Can be "cpu" or "cuda".
            min_samples_for_nmesc (int):
                The minimum number of samples required for NME clustering. This avoids
                zero p_neighbour_lists. If the input has fewer segments than min_samples,
                it is directed to the enhanced speaker counting mode.
            sparse_search (bool):
                Toggle sparse search mode. If True, limit the size of p_value_list to sparse_search_volume.
            maj_vote_spk_count (bool):
                If True, take a majority vote on all p-values in the given range to estimate the number of speakers.
                The majority voting may contribute to surpress overcounting of the speakers and improve speaker
                counting accuracy.
        """
        super().__init__()

        self.device = device
        self.maj_vote_spk_count: bool = maj_vote_spk_count
        self.min_samples_for_nmesc: int = min_samples_for_nmesc
        self.nme_mat_size: int = nme_mat_size
        self.sparse_search: bool = sparse_search

    def forward(
        self,
        embeddings_in_scales: List[torch.Tensor],
        timestamps_in_scales: List[torch.Tensor],
        multiscale_weights: torch.Tensor,
        oracle_num_speakers: int = -1,
        max_rp_threshold: float = 0.15,
        max_num_speakers: int = 8,
        enhanced_count_thres: int = 40,
        sparse_search_volume: int = 30,
        fixed_thres: float = -1.0,
        kmeans_random_trials: int = 1,
    ) -> torch.LongTensor:
        """Forward pass of the speaker clustering module.

        Calculate affinity matrix using timestamps and speaker embeddings, run NME analysis to estimate the best
        p-value and perform spectral clustering based on the estimated p-value and the calculated affinity matrix.

        Caution:
            For the sake of compatibility with libtorch, python boolean `False` is replaced with `torch.LongTensor(-1)`.

        Args:
            Dict containing following keys associated with tensors.
            embeddings (Tensor):
                Concatenated Torch tensor containing embeddings in multiple scales
                This tensor has dimensions of (Number of base segments) x (Embedding Dimension)
            timestamps (Tensor):
                Concatenated Torch tensor containing timestamps in multiple scales.
                This tensor has dimensions of (Total number of segments all scales) x 2
                Example:
                    >>> timestamps_in_scales = \
                    >>> torch.tensor([0.4, 1.4], [0.9, 1.9], [1.4, 2.4], ... [121.2, 122.2]])

            multiscale_weights (Tensor):
                Multi-scale weights that are used when affinity scores are merged.
                Example:
                    >>> multiscale_weights = torch.tensor([1.4, 1.3, 1.2, 1.1, 1.0])

            oracle_num_speakers (int):
                The number of speakers in a session from the reference transcript
            max_num_speakers (int):
                The upper bound for the number of speakers in each session
            max_rp_threshold (float):
                Limits the range of parameter search.
                Clustering performance can vary depending on this range.
                Default is 0.15.
            enhanced_count_thres (int):
                For the short audio recordings, clustering algorithm cannot
                accumulate enough amount of speaker profile for each cluster.
                Thus, function `getEnhancedSpeakerCount` employs anchor embeddings
                (dummy representations) to mitigate the effect of cluster sparsity.
                enhanced_count_thres = 80 is recommended.
            sparse_search_volume (int):
                Number of p_values we search during NME analysis.
                Default is 30. The lower the value, the faster NME-analysis becomes.
                Lower than 20 might cause a poor parameter estimation.
            fixed_thres (float):
                If fixed_thres value is provided, NME-analysis process will be skipped.
                This value should be optimized on a development set to obtain a quality result.
                Default is None and performs NME-analysis to estimate the threshold.
            kmeans_random_trials (int):
                Number of random trials for initializing k-means clustering. More trials
                will result in a more stable clustering result. Default is 1.

        Returns:
            Y (LongTensor):
                Speaker labels for the segments in the given input embeddings.
        """
        emb = embeddings_in_scales[-1]

        if emb.shape[0] == 1:
            return torch.zeros((1,), dtype=torch.int64)

        elif emb.shape[0] <= max(enhanced_count_thres, self.min_samples_for_nmesc) and oracle_num_speakers < 0:
            est_num_of_spk_enhanced = getEnhancedSpeakerCount(emb=emb, cuda=self.cuda)

        else:
            est_num_of_spk_enhanced = torch.tensor(-1)

        if oracle_num_speakers > 0:
            max_num_speakers = oracle_num_speakers

        multiscale_cosine_affinity_matrix = self.get_multiscale_cosine_affinity_matrix(
            embeddings_in_scales, timestamps_in_scales, multiscale_weights,
        )

        nmesc = NMESC(
            multiscale_cosine_affinity_matrix,
            max_num_speakers=max_num_speakers,
            max_rp_threshold=max_rp_threshold,
            sparse_search=self.sparse_search,
            sparse_search_volume=sparse_search_volume,
            fixed_thres=fixed_thres,
            nme_mat_size=self.nme_mat_size,
            maj_vote_spk_count=self.maj_vote_spk_count,
            parallelism=self.parallelism,
            cuda=self.cuda,
            device=self.device,
        )

        # If there are less than `min_samples_for_nmesc` segments, est_num_of_spk is 1.
        if mat.shape[0] > self.min_samples_for_nmesc:
            est_num_of_spk, p_hat_value = nmesc.forward()
            affinity_mat = getAffinityGraphMat(mat, p_hat_value)
        else:
            nmesc.fixed_thres = max_rp_threshold
            est_num_of_spk, p_hat_value = nmesc.forward()
            affinity_mat = mat

        # n_clusters is number of speakers estimated from spectral clustering.
        if oracle_num_speakers > 0:
            n_clusters = int(oracle_num_speakers)
        elif est_num_of_spk_enhanced > 0:
            n_clusters = int(est_num_of_spk_enhanced.item())
        else:
            n_clusters = int(est_num_of_spk.item())

        spectral_model = SpectralClustering(
            n_clusters=n_clusters, n_random_trials=kmeans_random_trials, cuda=self.cuda, device=self.device
        )
        Y = spectral_model.forward(affinity_mat)
        return Y

    def get_multiscale_cosine_affinity_matrix(
        self,
        embeddings_in_scales: List[torch.Tensor],
        timestamps_in_scales: List[torch.Tensor],
        multiscale_weights: List[float],
    ) -> torch.Tensor:
        """Get multiscale cosine affinity matrix.

        Calculate cosine similarity values among speaker embeddings for each scale then
        apply multiscale weights to calculate the fused similarity matrix.

        Args:
            embeddings_in_scales (List[torch.Tensor]): List containing split embedding tensors by each scale.
            timestamps_in_scales (List[torch.Tensor]): List containing split timestamps tensors by each scale.
            multiscale_weights (List[float]): List of weights for each scale.

        Returns:
            torch.Tensor: Fused similarity matrix, obtained by calculating the weighted sum of 
                the multiple affinity matrices from the different scales.
        """
        session_scale_mapping_list = get_argmin_mapping_list(timestamps_in_scales)

        fused_sim_d = torch.zeros(
            len(timestamps_in_scales[-1]), len(timestamps_in_scales[-1])
        ).to(self.device)

        for embeddings, weight, map_argmin in zip(
            embeddings_in_scales, multiscale_weights, session_scale_mapping_list
        ):
            cosine_affinity_matrix = self.get_cosine_affinity_matrix(embeddings.to(self.device))

            repeat_list = self.get_repeated_list(
                map_argmin, torch.tensor(cosine_affinity_matrix.shape[0])
            ).to(self.device)

            repeated_tensor_0 = torch.repeat_interleave(
                cosine_affinity_matrix, repeats=repeat_list, dim=0,
            ).to(self.device)
            repeated_tensor_1 = torch.repeat_interleave(
                repeated_tensor_0, repeats=repeat_list, dim=1,
            ).to(self.device)

            fused_sim_d += weight * repeated_tensor_1

        return fused_sim_d

    @staticmethod
    def get_cosine_affinity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
        """ Get cosine affinity matrix.

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
            normalized_cosine_affinity_matrix = (cosine_affinity_matix - v_min) / (v_max - v_min)

        return normalized_cosine_affinity_matrix

    @staticmethod
    def get_repeated_list(map_argmin: torch.Tensor, matrix_size: torch.Tensor) -> torch.Tensor:
        """Get repeated list for the affinity matrix.

        Count the numbers in the mapping dictionary and create lists that contain
        repeated indices that will be used for creating a repeated affinity matrix.
        This repeated matrix is then used for fusing multiple affinity values.

        Args:
            map_argmin (torch.Tensor): Mapping dictionary that contains the mapping between the base scale 
                and other scales.
            matrix_size (torch.Tensor): Size of the affinity matrix.

        Returns:
            torch.Tensor: List containing repeated indices.
        """
        repeat_list = torch.zeros(matrix_size, dtype=torch.int32)

        idxs, counts = torch.unique(map_argmin, return_counts=True)
        repeat_list[idxs] = counts.int()

        return repeat_list


class ClusteringModule:
    """Clustering module for diariation."""

    def __init__(self, device: str, max_num_speakers: int = 8) -> None:
        """Initialize the clustering module."""
        self.params = dict(
            oracle_num_speakers=False,
            max_num_speakers=max_num_speakers,
            enhanced_count_thres=80,
            max_rp_threshold=0.25,
            sparse_search_volume=30,
            maj_vote_spk_count=False,
        )
        self.clustering_model = SpeakerClustering(parallelism=True, cuda=True)
        self.clustering_model.device = device

    def __call__(
        self, ms_emb_ts: MultiscaleEmbeddingsAndTimestamps
    ) -> List[Tuple[float, float, int]]:
        """
        Run the clustering module and return the speaker segments.

        Args:
            ms_emb_ts (MultiscaleEmbeddingsAndTimestamps): Embeddings and timestamps of the audio file in multiscale.
                The multiscale embeddings and timestamps are from the SegmentationModule.

        Returns:
            List[Tuple[float, float, int]]: List of segments with the following keys: "start", "end", "speaker".
        """
        base_scale_idx = ms_emb_ts.multiscale_segment_counts.shape[0] - 1
        cluster_labels = self.clustering_model.forward_infer(
            embeddings_in_scales=ms_emb_ts.embeddings,
            timestamps_in_scales=ms_emb_ts.timestamps,
            multiscale_segment_counts=ms_emb_ts.multiscale_segment_counts,
            multiscale_weights=ms_emb_ts.multiscale_weights,
            oracle_num_speakers=-1,
            max_num_speakers=self.params["max_num_speakers"],
            max_rp_threshold=self.params["max_rp_threshold"],
            sparse_search_volume=self.params["sparse_search_volume"],
        )

        del ms_emb_ts
        torch.cuda.empty_cache()

        timestamps = self.clustering_model.timestamps_in_scales[base_scale_idx]
        cluster_labels = cluster_labels.cpu().numpy()

        if len(cluster_labels) != timestamps.shape[0]:
            raise ValueError(
                "Mismatch of length between cluster_labels and timestamps."
            )

        clustering_labels = []
        for idx, label in enumerate(cluster_labels):
            start, end = timestamps[idx]
            clustering_labels.append((float(start), float(start + end), int(label)))

        return clustering_labels
