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

from wordcab_transcribe.services.diarization.models import (
    MultiscaleEmbeddingsAndTimestamps,
)
from wordcab_transcribe.services.diarization.utils import (
    add_anchor_embeddings,
    eigen_decompose,
    estimate_number_of_speakers,
    get_affinity_graph_matrix,
    get_argmin_mapping_list,
    get_context_embeddings,
    get_cosine_affinity_matrix,
    get_euclidean_distance,
    get_laplacian,
    get_merge_quantity,
    get_minimum_connection,
    is_graph_fully_connected,
    kmeans_plusplus_torch,
    run_reducer,
    split_embs_to_windows,
)


# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/utils/offline_clustering.py#L618
def get_enhanced_speaker_count(
    emb: torch.Tensor,
    random_test_count: int = 5,
    anchor_spk_n: int = 3,
    anchor_sample_n: int = 10,
    sigma: float = 50,
    device: str = "cuda",
) -> torch.Tensor:
    """Calculate the number of speakers using NME analysis with anchor embeddings.

    Add dummy speaker embedding vectors and run speaker counting multiple times
    to enhance the speaker counting accuracy for the short audio samples.

    Args:
        emb (torch.Tensor):
            The input embedding from the embedding extractor.
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
        device (str):
            The device to run the operations on ('cuda' or 'cpu').

    Returns:
        comp_est_num_of_spk (torch.Tensor):
            The estimated number of speakers. `anchor_spk_n` is subtracted from the estimated
            number of speakers to factor out the dummy speaker embedding vectors.
    """
    estimations: List[int] = []
    for seed in range(random_test_count):
        torch.manual_seed(seed)

        embeddings_aug = add_anchor_embeddings(
            emb, anchor_sample_n, anchor_spk_n, sigma
        )
        matrix = get_cosine_affinity_matrix(embeddings_aug)

        nmesc = NMESC(
            matrix,
            max_num_speakers=emb.shape[0],
            max_rp_threshold=0.15,
            sparse_search=True,
            sparse_search_volume=10,
            fixed_thres=-1.0,
            nme_mat_size=300,
            device=device,
        )
        estimation_number_of_speakers, _ = nmesc.forward()
        estimations.append(estimation_number_of_speakers.item())

    comp_est_num_of_spk = torch.tensor(
        max(torch.mode(torch.tensor(estimations))[0].item() - anchor_spk_n, 1)
    )

    return comp_est_num_of_spk


# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/utils/offline_clustering.py#L850
class NMESC:
    """Normalized Maximum Eigengap based Spectral Clustering (NME-SC).

    NME-SC uses Eigengap analysis to get an estimated p-value for
    affinity binarization and an estimated number of speakers.

    p_value (also referred to as p_neighbors) is for taking top p number of affinity values
    and convert those to 1 while convert the rest of values to 0.

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
        device: str,
        max_num_speakers: int = 10,
        max_rp_threshold: float = 0.15,
        sparse_search: bool = True,
        sparse_search_volume: int = 30,
        nme_mat_size: int = 512,
        use_subsampling_for_nme: bool = True,
        fixed_thres: float = -1.0,
        maj_vote_spk_count: bool = False,
        parallelism: bool = True,
    ) -> None:
        """
        Initialize the NMESC class.

        Args:
            matrix (torch.Tensor):
                Cosine similarity matrix calculated from the provided speaker embeddings.
            device (str):
                Device to run the NME analysis.
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
            fixed_thres (float):
                A fixed threshold which can be used instead of estimating the
                threshold with NME analysis. If fixed_thres is float,
                it skips the NME analysis part.
            maj_vote_spk_count (bool):
                If True, take a majority vote on all p-values in the given range to estimate the number of speakers.
                The majority voting may contribute to surpress overcounting of the speakers and improve speaker
                counting accuracy.
            parallelism (bool):
                If True, turn on parallelism based on torch.jit.script library.
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
        self.device: str = device
        self.maj_vote_spk_count: bool = maj_vote_spk_count
        self.parallelism: bool = parallelism

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Subsample the input matrix to reduce the computational load.

        Returns:
            est_num_of_spk (torch.Tensor): Estimated number of speakers from NMESC approach
            p_hat_value (torch.Tensor): Estimated p-value (determines how many neighboring values to be selected)
        """
        if self.use_subsampling_for_nme:
            subsample_ratio = self.subsample_affinity_matrix(self.nme_mat_size)
        else:
            subsample_ratio = torch.tensor(1)

        # Scans p_values and find a p_value that generates the smallest g_p value.
        est_spk_n_dict: Dict[int, torch.Tensor] = {}
        self.p_value_list = self.get_p_value_list()

        p_volume = self.p_value_list.shape[0]
        eig_ratio_list = torch.zeros(p_volume)
        est_num_of_spk_list = torch.zeros(p_volume)

        results: List[torch.Tensor] = []
        if self.parallelism:
            futures: List[torch.jit.Future[torch.Tensor]] = []
            for p_value in self.p_value_list:
                futures.append(torch.jit.fork(self.get_eigengap_ratio, p_value))
            for future in futures:
                results.append(torch.jit.wait(future))

        else:
            for p_value in self.p_value_list:
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
        affinity_matrix = get_affinity_graph_matrix(self.mat, rp_p_value)

        # Checks whether the affinity graph is fully connected.
        # If not, it adds a minimum number of connections to make it fully connected.
        if not is_graph_fully_connected(affinity_matrix, device=self.device):
            affinity_matrix, rp_p_value = get_minimum_connection(
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
        subsample_ratio = torch.max(
            torch.tensor(1), torch.tensor(self.mat.shape[0] / nme_mat_size)
        ).type(torch.int)
        self.mat = self.mat[:: subsample_ratio.item(), :: subsample_ratio.item()]

        return subsample_ratio

    def get_eigengap_ratio(self, p_neighbors: int) -> torch.Tensor:
        """Get the ratio between p_neighbors and the maximum eigen gap value.

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
        affinity_matrix = get_affinity_graph_matrix(self.mat, p_neighbors)
        est_num_of_spk, lambdas, lambda_gap_list = estimate_number_of_speakers(
            affinity_matrix, self.max_num_speakers, self.device
        )
        arg_sorted_idx = torch.argsort(
            lambda_gap_list[: self.max_num_speakers], descending=True
        )

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
                torch.floor(torch.tensor(self.mat.shape[0] * self.fixed_thres)).type(
                    torch.int
                ),
                self.min_p_value,
            )
            p_value_list = self.max_N.unsqueeze(0).int()

        else:
            self.max_N = torch.max(
                torch.floor(
                    torch.tensor(self.mat.shape[0] * self.max_rp_threshold)
                ).type(torch.int),
                self.min_p_value,
            )

            if self.sparse_search:
                search_volume = torch.min(
                    self.max_N, torch.tensor(self.sparse_search_volume).type(torch.int)
                )
                # search at least two values
                N = torch.max(search_volume, torch.tensor(2))
                # avoid repeating values by limiting the step size
                steps = min(self.max_N, N)
                p_value_list = torch.linspace(
                    start=1, end=self.max_N, steps=steps
                ).type(torch.int)

            else:
                p_value_list = torch.arange(1, self.max_N + 1)

        if p_value_list.shape[0] == 0:
            raise ValueError("p_value_list should not be empty.")

        return p_value_list


# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/utils/offline_clustering.py#L737
class SpectralClustering:
    """Speaker diarization based on spectral clustering.

    Perform spectral clustering by calculating spectral embeddings then run k-means clustering
    algorithm on the spectral embeddings.
    """

    def __init__(
        self,
        device: str,
        n_clusters: int = 8,
        random_state: int = 0,
        n_random_trials: int = 1,
    ):
        """
        Initialize the variables needed for spectral clustering and k-means++.

        Args:
            device (str):
                Torch device variable
            n_clusters (int):
                Number of the estimated (or oracle) number of speakers
            random_state (int):
                Random seed that determines a random state of k-means initialization.
            n_random_trials (int):
                Number of trials with different random seeds for k-means initialization.
                k-means++ algorithm is executed for multiple times then the final result
                is obtained by taking a majority vote.
        """
        self.device = device
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_random_trials = max(n_random_trials, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Perform k-means clustering on spectral embeddings.

        To alleviate the effect of randomness, k-means clustering is performed for (self.n_random_trials) times
        then the final labels are obtained by taking a majority vote. If speed is the major concern,
        self.n_random_trials should be set to 1. n_random_trials=30 is recommended to see an improved result.

        Args:
            X (torch.Tensor):
                Affinity matrix input

        Returns:
            labels (torch.Tensor):
                clustering label output
        """
        if X.shape[0] != X.shape[1]:
            raise ValueError("The affinity matrix is not a square matrix.")

        if X.device != self.device:
            X = X.to(self.device)

        spectral_embeddings = self.get_spectral_embeddings(
            X, n_speakers=self.n_clusters, device=self.device
        )
        labels_set = []

        for random_state_seed in range(
            self.random_state, self.random_state + self.n_random_trials
        ):
            _labels = self.kmeans_torch(
                X=spectral_embeddings,
                num_clusters=self.n_clusters,
                random_state=random_state_seed,
                device=self.device,
            )
            labels_set.append(_labels)

        stacked_labels = torch.stack(labels_set)
        label_index = torch.mode(torch.mode(stacked_labels, 0)[1])[0]
        labels = stacked_labels[label_index]

        return labels

    @staticmethod
    def get_spectral_embeddings(
        affinity_matrix: torch.Tensor, n_speakers: int = 8, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Calculate eigenvalues and eigenvectors to extract spectral embeddings.

        Args:
            affinity_matrix (torch.Tensor):
                Affinity matrix input
            n_speakers (int):
                Number of the estimated (or oracle) number of speakers
            device (str):
                Torch device variable. Default is "cpu".

        Returns:
            labels (torch.Tensor): clustering label output
        """
        laplacian = get_laplacian(affinity_matrix)

        _, _diffusion_map = eigen_decompose(laplacian, device)
        diffusion_map = _diffusion_map[:, :n_speakers]

        inv_index = torch.arange(diffusion_map.size(1) - 1, -1, -1).long()
        embedding = diffusion_map.T[inv_index, :]

        return embedding[:n_speakers].T

    @staticmethod
    def kmeans_torch(
        X: torch.Tensor,
        num_clusters: int,
        device: str,
        threshold: float = 1e-4,
        iter_limit: int = 15,
        random_state: int = 0,
    ) -> torch.Tensor:
        """Run k-means algorithm on the given set of spectral embeddings in X.

        The threshold and iter_limit variables are set to show the best performance on speaker diarization
        tasks. The overall implementation of k-means algorithm is inspired by the k-means
        algorithm implemented in https://github.com/scikit-learn/scikit-learn.

        References:
            Arthur, David, and Sergei Vassilvitskii. k-means++: The advantages of careful
            seeding. Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete
            algorithms, Society for Industrial and Applied Mathematics (2007).

        Args:
            X (torch.Tensor):
                Cosine similarity matrix calculated from speaker embeddings
            num_clusters (int):
                The estimated number of speakers.
            device (str):
                Torch device to be used for the k-means algorithm.
            threshold (float):
                This threshold limits the change of center values. If the square of
                the center shift values are bigger than this threshold, the iteration stops.
            iter_limit (int):
                The maximum number of iterations that is allowed by the k-means algorithm.
            random_state (int):
                Random seed that determines a random state of k-means initialization.

        Returns:
            selected_cluster_indices (Tensor):
                The assigned cluster labels from the k-means clustering.
        """
        # Convert tensor type to float
        X = X.float().to(device)
        input_size = X.shape[0]

        # Initialize the cluster centers with kmeans_plusplus algorithm.
        plusplus_init_states = kmeans_plusplus_torch(
            X, n_clusters=num_clusters, random_state=random_state, device=device
        )
        centers = plusplus_init_states[0]

        selected_cluster_indices = torch.zeros(input_size).long()

        for _ in range(iter_limit):
            euc_dist = get_euclidean_distance(X, centers, device=device)

            if len(euc_dist.shape) <= 1:
                break
            else:
                selected_cluster_indices = torch.argmin(euc_dist, dim=1)

            center_inits = centers.clone()

            for index in range(num_clusters):
                selected_cluster = (
                    torch.nonzero(selected_cluster_indices == index)
                    .squeeze()
                    .to(device)
                )
                chosen_indices = torch.index_select(X, 0, selected_cluster)

                if chosen_indices.shape[0] == 0:
                    chosen_indices = X[torch.randint(len(X), (1,))]

                centers[index] = chosen_indices.mean(dim=0)

            # Calculate the delta from center_inits to centers
            center_delta_pow = torch.pow((centers - center_inits), 2)
            center_shift_pow = torch.pow(
                torch.sum(torch.sqrt(torch.sum(center_delta_pow, dim=1))), 2
            )

            # If the cluster centers are not changing significantly, stop the loop.
            if center_shift_pow < threshold:
                break

        return selected_cluster_indices


class SpeakerClustering(torch.nn.Module):
    """Clustering module for speaker diarization."""

    def __init__(
        self,
        device: str,
        maj_vote_spk_count: bool = False,
        min_samples_for_nmesc: int = 6,
        nme_mat_size: int = 512,
        parallelism: bool = False,
        sparse_search: bool = True,
        longform_clustering: bool = False,
        chunk_cluster_count: int = 50,
        embeddings_per_chunk: int = 10000,
    ):
        """
        Clustering method for speaker diarization based on cosine similarity.

        Args:
            device (str): Device to use for inference. Can be "cpu" or "cuda".
            maj_vote_spk_count (bool):
                If True, take a majority vote on all p-values in the given range to estimate the number of speakers.
                The majority voting may contribute to surpress overcounting of the speakers and improve speaker
                counting accuracy.
            min_samples_for_nmesc (int):
                The minimum number of samples required for NME clustering. This avoids
                zero p_neighbour_lists. If the input has fewer segments than min_samples,
                it is directed to the enhanced speaker counting mode.
            nme_mat_size (int):
                The size of the NME matrix. The NME matrix is a matrix that contains the NME values for all possible
                number of speakers. The NME matrix is used to estimate the number of speakers in the given audio.
            parallelism (bool):
                If True, use parallel processing for speaker clustering.
            sparse_search (bool):
                Toggle sparse search mode. If True, limit the size of p_value_list to sparse_search_volume.
        """
        super().__init__()

        self.device = device
        self.maj_vote_spk_count: bool = maj_vote_spk_count
        self.min_samples_for_nmesc: int = min_samples_for_nmesc
        self.nme_mat_size: int = nme_mat_size
        self.parallelism: bool = parallelism
        self.sparse_search: bool = sparse_search
        self.longform_clustering: bool = longform_clustering
        self.chunk_cluster_count: int = chunk_cluster_count
        self.embeddings_per_chunk: int = embeddings_per_chunk

    def forward(
        self,
        embeddings_in_scales: List[torch.Tensor],
        timestamps_in_scales: List[torch.Tensor],
        multiscale_weights: torch.Tensor,
        enhanced_count_thres: int = 40,
        fixed_thres: float = -1.0,
        kmeans_random_trials: int = 1,
        oracle_num_speakers: int = -1,
        max_num_speakers: int = 8,
        max_rp_threshold: float = 0.15,
        sparse_search_volume: int = 30,
    ) -> torch.LongTensor:
        """Forward pass of the speaker clustering module.

        Calculate affinity matrix using timestamps and speaker embeddings, run NME analysis to estimate the best
        p-value and perform spectral clustering based on the estimated p-value and the calculated affinity matrix.

        Args:
            embeddings_in_scales (torch.Tensor):
                Concatenated Torch tensor containing embeddings in multiple scales
                This tensor has dimensions of (Number of base segments) x (Embedding Dimension)
            timestamps_in_scales (torch.Tensor):
                Concatenated Torch tensor containing timestamps in multiple scales.
                This tensor has dimensions of (Total number of segments all scales) x 2
            multiscale_weights (torch.Tensor):
                Multi-scale weights that are used when affinity scores are merged.
            enhanced_count_thres (int):
                For the short audio recordings, clustering algorithm cannot
                accumulate enough amount of speaker profile for each cluster.
                Thus, function `getEnhancedSpeakerCount` employs anchor embeddings
                (dummy representations) to mitigate the effect of cluster sparsity.
                enhanced_count_thres = 80 is recommended.
            fixed_thres (float):
                If fixed_thres value is provided, NME-analysis process will be skipped.
                This value should be optimized on a development set to obtain a quality result.
                Default is None and performs NME-analysis to estimate the threshold.
            kmeans_random_trials (int):
                Number of random trials for initializing k-means clustering. More trials
                will result in a more stable clustering result. Default is 1.
            max_num_speakers (int):
                The upper bound for the number of speakers in each session
            max_rp_threshold (float):
                Limits the range of parameter search. Clustering performance can vary depending on this range.
                Default is 0.15.
            oracle_num_speakers (int):
                The number of speakers in a session from the reference transcript
            sparse_search_volume (int):
                Number of p_values we search during NME analysis.
                Default is 30. The lower the value, the faster NME-analysis becomes.
                Lower than 20 might cause a poor parameter estimation.

        Returns:
            Y (torch.LongTensor):
                Speaker labels for the segments in the given input embeddings.
        """
        emb = embeddings_in_scales[-1]

        if emb.shape[0] == 1:
            return torch.zeros((1,), dtype=torch.int64)
        elif (
            emb.shape[0] <= max(enhanced_count_thres, self.min_samples_for_nmesc)
            and oracle_num_speakers < 0
        ):
            estimation_number_of_speaker_enhanced = get_enhanced_speaker_count(
                emb=emb, device=self.device
            )
        else:
            estimation_number_of_speaker_enhanced = torch.tensor(-1)

        if oracle_num_speakers > 0:
            max_num_speakers = oracle_num_speakers

        (
            multiscale_cosine_affinity_matrix,
            session_scale_mapping_list,
        ) = self.get_multiscale_cosine_affinity_matrix(
            embeddings_in_scales,
            timestamps_in_scales,
            multiscale_weights,
        )

        if self.longform_clustering:
            context_embeddings = get_context_embeddings(
                multiscale_weights,
                embeddings_in_scales,
                session_scale_mapping_list,
                device=self.device,
            )

            (
                total_embeddings,
                window_range_list,
                absolute_merge_mapping,
            ) = self.process_context_embeddings(
                context_embeddings=context_embeddings,
                embeddings_per_chunk=self.embeddings_per_chunk,
                chunk_cluster_count=self.chunk_cluster_count,
                max_rp_threshold=max_rp_threshold,
                sparse_search_volume=sparse_search_volume,
            )

            reduce_embeddings = torch.cat(total_embeddings)
            reduced_matrix = get_cosine_affinity_matrix(reduce_embeddings)

            Y_aggregate = self.forward_unit(
                matrix=reduced_matrix,
                max_num_speakers=max_num_speakers,
                max_rp_threshold=max_rp_threshold,
                sparse_search_volume=sparse_search_volume,
                fixed_thres=fixed_thres,
                oracle_num_speakers=oracle_num_speakers,
                estimation_number_of_speaker_enhanced=estimation_number_of_speaker_enhanced,
                kmeans_random_trials=kmeans_random_trials,
            )

            if reduce_embeddings.shape[0] != Y_aggregate.shape[0]:
                raise ValueError(
                    f"The number of embeddings ({reduce_embeddings.shape[0]}) and the"
                    f" number of clustered labels ({Y_aggregate.shape[0]}) do not"
                    " match."
                )

            # Reassign the labels to the original embeddings
            Y_unpack = self.unpack_labels(
                Y_aggr=Y_aggregate,
                window_range_list=window_range_list,
                absolute_merge_mapping=absolute_merge_mapping,
                org_len=context_embeddings.shape[0],
            )
            if Y_unpack.shape[0] != context_embeddings.shape[0]:
                raise ValueError(
                    "The number of raw input embeddings"
                    f" ({context_embeddings.shape[0]}) and the number of clustered"
                    f" labels ({Y_unpack.shape[0]}) do not match."
                )
            return Y_unpack
        else:
            return self.forward_unit(
                matrix=multiscale_cosine_affinity_matrix,
                max_num_speakers=max_num_speakers,
                max_rp_threshold=max_rp_threshold,
                sparse_search_volume=sparse_search_volume,
                fixed_thres=fixed_thres,
                oracle_num_speakers=oracle_num_speakers,
                estimation_number_of_speaker_enhanced=estimation_number_of_speaker_enhanced,
                kmeans_random_trials=kmeans_random_trials,
            )

    def forward_unit(
        self,
        matrix: torch.Tensor,
        max_num_speakers: int,
        max_rp_threshold: float,
        sparse_search_volume: int,
        fixed_thres: float,
        oracle_num_speakers: int,
        estimation_number_of_speaker_enhanced: torch.Tensor,
        kmeans_random_trials: int,
    ) -> torch.LongTensor:
        """
        Performs the forward pass of the unit processing, involving speaker clustering using a multiscale cosine
        affinity matrix and various clustering parameters. This function applies the NMESC algorithm to estimate
        the number of speakers and uses spectral clustering for final speaker segmentation.

        Args:
            matrix (torch.Tensor): The required matrix.
            matrix_shape (int): The shape of the affinity matrix, used for determining the number of segments.
            max_num_speakers (int): The maximum number of speakers to consider in the clustering process.
            max_rp_threshold (float): The maximum RP threshold, used in the NMESC algorithm for estimating the number
                of speakers.
            sparse_search_volume (int): The number of p-values to consider during the NMESC analysis.
            fixed_thres (float): A fixed threshold for the NMESC analysis. If set, NMESC will not estimate the p-value
                but use this fixed value instead.
            oracle_num_speakers (int): The actual number of speakers present, known from a reference transcript.
                If set, this overrides the estimated number of speakers.
            estimation_number_of_speaker_enhanced (torch.Tensor): The enhanced estimation of the number of speakers,
                used for short audio recordings where traditional clustering might not be effective.
            kmeans_random_trials (int): The number of random trials for initializing K-means clustering in the spectral
                clustering process.

        Returns:
            torch.LongTensor: The final speaker labels for each segment, determined by the spectral clustering algorithm.

        Notes:
            This function integrates multiple stages of the speaker clustering pipeline, including NMESC for speaker
            estimation and spectral clustering for final segmentation. It adjusts the clustering approach based on the
            size of the input and the provided parameters, ensuring robustness across different types of audio recordings.
        """
        matrix_shape = matrix.shape

        nmesc = NMESC(
            matrix,
            max_num_speakers=max_num_speakers,
            max_rp_threshold=max_rp_threshold,
            sparse_search=self.sparse_search,
            sparse_search_volume=sparse_search_volume,
            fixed_thres=fixed_thres,
            nme_mat_size=self.nme_mat_size,
            maj_vote_spk_count=self.maj_vote_spk_count,
            parallelism=self.parallelism,
            device=self.device,
        )

        # If there are less than `min_samples_for_nmesc` segments, estimation_number_of_speakers is 1.
        if matrix_shape[0] > self.min_samples_for_nmesc:
            estimation_number_of_speakers, p_hat_value = nmesc.forward()
            affinity_matrix = get_affinity_graph_matrix(matrix, p_hat_value)
        else:
            nmesc.fixed_thres = max_rp_threshold
            estimation_number_of_speakers, p_hat_value = nmesc.forward()
            affinity_matrix = matrix

        # n_clusters is number of speakers estimated from spectral clustering.
        if oracle_num_speakers > 0:
            n_clusters = int(oracle_num_speakers)
        elif estimation_number_of_speaker_enhanced > 0:
            n_clusters = int(estimation_number_of_speaker_enhanced.item())
        else:
            n_clusters = int(estimation_number_of_speakers.item())

        spectral_model = SpectralClustering(
            n_clusters=n_clusters,
            n_random_trials=kmeans_random_trials,
            device=self.device,
        )
        Y = spectral_model.forward(affinity_matrix)

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
            embeddings_in_scales,
            multiscale_weights,
            session_scale_mapping_list,
        ):
            cosine_affinity_matrix = get_cosine_affinity_matrix(
                embeddings.to(self.device)
            )

            repeat_list = self.get_repeated_list(
                map_argmin, torch.tensor(cosine_affinity_matrix.shape[0])
            ).to(self.device)

            repeated_tensor_0 = torch.repeat_interleave(
                cosine_affinity_matrix,
                repeats=repeat_list,
                dim=0,
            ).to(self.device)
            repeated_tensor_1 = torch.repeat_interleave(
                repeated_tensor_0,
                repeats=repeat_list,
                dim=1,
            ).to(self.device)

            fused_sim_d += weight * repeated_tensor_1

        return fused_sim_d, session_scale_mapping_list

    def process_context_embeddings(
        self,
        context_embeddings: torch.Tensor,
        embeddings_per_chunk: int,
        chunk_cluster_count: int,
        max_rp_threshold: float,
        sparse_search_volume: int,
    ) -> Tuple:
        """
        Processes the given embeddings by splitting them into smaller chunks,
        performing overclustering, and merging the clusters.

        Args:
            context_embeddings (torch.Tensor): The scale interpolated embeddings tensor.
            embeddings_per_chunk (int): Number of embeddings per chunk.
            chunk_cluster_count (int): Number of clusters per chunk.
            max_rp_threshold (float): Maximum RP threshold for clustering.
            sparse_search_volume (int): Sparse search volume for clustering.

        Returns:
            List[torch.Tensor]: A list of merged embeddings.
        """
        offset_index = 0
        window_offset = 0
        total_emb = []
        window_range_list = []
        absolute_merge_mapping = []
        total_window_count = int(
            torch.ceil(
                torch.tensor(context_embeddings.shape[0] / embeddings_per_chunk)
            ).item()
        )

        for win_index in range(total_window_count):
            # Split the embeddings into smaller chunks
            emb_part, offset_index = split_embs_to_windows(
                index=win_index,
                emb=context_embeddings,
                embeddings_per_chunk=embeddings_per_chunk,
            )

            # Perform overclustering on the chunks
            if emb_part.shape[0] == 1:
                Y_part = torch.zeros((1,), dtype=torch.int64)
            else:
                matrix = get_cosine_affinity_matrix(emb_part)
                overcluster_count = min(chunk_cluster_count, matrix.shape[0])
                Y_part = self.speaker_clustering.forward_unit_infer(
                    mat=matrix,
                    oracle_num_speakers=overcluster_count,
                    max_rp_threshold=max_rp_threshold,
                    max_num_speakers=chunk_cluster_count,
                    sparse_search_volume=sparse_search_volume,
                )

            # Merge the clusters
            num_to_be_merged = int(
                min(embeddings_per_chunk, emb_part.shape[0]) - chunk_cluster_count
            )
            min_count_per_cluster = int(
                torch.ceil(
                    torch.tensor(chunk_cluster_count / len(torch.unique(Y_part)))
                ).item()
            )

            class_target_vol = get_merge_quantity(
                num_to_be_removed=num_to_be_merged,
                pre_clus_labels=Y_part,
                min_count_per_cluster=min_count_per_cluster,
            )

            # Process each cluster
            for spk_idx, merge_quantity in enumerate(list(class_target_vol)):
                merged_embs, merged_clus_labels, index_mapping = run_reducer(
                    pre_embs=emb_part,
                    target_spk_idx=spk_idx,
                    merge_quantity=merge_quantity,
                    pre_clus_labels=Y_part,
                )
                total_emb.append(merged_embs)
                absolute_index_mapping = [x + offset_index for x in index_mapping]
                absolute_merge_mapping.append(absolute_index_mapping)
                window_range_list.append(
                    [window_offset, window_offset + merged_embs.shape[0]]
                )
                window_offset += merged_embs.shape[0]

        return total_emb, window_range_list, absolute_merge_mapping

    def unpack_labels(
        self,
        Y_aggr: torch.Tensor,
        window_range_list: List[List[int]],
        absolute_merge_mapping: List[List[torch.Tensor]],
        org_len: int,
    ) -> torch.LongTensor:
        """
        Unpack the labels from the aggregated labels to the original labels.

        Args:
            Y_aggr (Tensor):
                Aggregated label vector from the merged segments.
            window_range_list (List[List[int]]):
                List of window ranges for each of the merged segments.
            absolute_merge_mapping (List[List[torch.Tensor]]):
                List of absolute mappings for each of the merged segments. Each list element contains two tensors:
                    - The first tensor represents the absolute index of the bypassed segment (segments that remain unchanged).
                    - The second tensor represents the absolute index of the merged segment (segments that have had their indexes changed).
            org_len (int):
                Original length of the labels. In most cases, this is a fairly large number (on the order of 10^5).

        Returns:
            Y_unpack (Tensor):
                Unpacked labels derived from the aggregated labels.
        """
        Y_unpack = torch.zeros((org_len,)).long().to(Y_aggr.device)
        for win_rng, abs_mapping in zip(window_range_list, absolute_merge_mapping):
            inferred_merged_embs = Y_aggr[win_rng[0] : win_rng[1]]
            if len(abs_mapping[1]) > 0:
                Y_unpack[abs_mapping[1]] = inferred_merged_embs[-1].clone()  # Merged
                if len(abs_mapping[0]) > 0:
                    Y_unpack[abs_mapping[0]] = inferred_merged_embs[
                        :-1
                    ].clone()  # Bypass
            else:
                if len(abs_mapping[0]) > 0:
                    Y_unpack[abs_mapping[0]] = inferred_merged_embs.clone()
        return Y_unpack

    @staticmethod
    def get_repeated_list(
        map_argmin: torch.Tensor, matrix_size: torch.Tensor
    ) -> torch.Tensor:
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

    def __init__(
        self, device: str, max_num_speakers: int = 8, longform_clustering: bool = False
    ) -> None:
        """Initialize the clustering module."""
        self.params = {
            "max_num_speakers": max_num_speakers,
            "enhanced_count_thres": 80,
            "max_rp_threshold": 0.25,
            "sparse_search_volume": 30,
            "maj_vote_spk_count": False,
        }
        self.clustering_model = SpeakerClustering(
            device=device, parallelism=False, longform_clustering=longform_clustering
        )

    def __call__(
        self,
        ms_emb_ts: MultiscaleEmbeddingsAndTimestamps,
        oracle_num_speakers: int,
    ) -> List[Tuple[float, float, int]]:
        """
        Run the clustering module and return the speaker segments.

        Args:
            ms_emb_ts (MultiscaleEmbeddingsAndTimestamps): Embeddings and timestamps of the audio file in multiscale.
                The multiscale embeddings and timestamps are from the SegmentationModule.
            oracle_num_speakers (int): Number of speakers in the audio file.

        Returns:
            List[Tuple[float, float, int]]: List of segments with the following keys: "start", "end", "speaker".
        """
        timestamps = ms_emb_ts.timestamps[ms_emb_ts.base_scale_index]
        cluster_labels = self.clustering_model.forward(
            embeddings_in_scales=ms_emb_ts.embeddings,
            timestamps_in_scales=ms_emb_ts.timestamps,
            multiscale_weights=ms_emb_ts.multiscale_weights,
            enhanced_count_thres=self.params["enhanced_count_thres"],
            oracle_num_speakers=oracle_num_speakers,
            max_num_speakers=self.params["max_num_speakers"],
            max_rp_threshold=self.params["max_rp_threshold"],
            sparse_search_volume=self.params["sparse_search_volume"],
        )

        del ms_emb_ts
        torch.cuda.empty_cache()

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
