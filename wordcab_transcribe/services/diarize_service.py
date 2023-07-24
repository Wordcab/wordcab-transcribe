# Copyright 2023 The Wordcab Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diarization Service for audio files."""

from pathlib import Path
from typing import List, NamedTuple, Union

import librosa
import soundfile as sf
import torch
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from wordcab_transcribe.logging import time_and_tell
from wordcab_transcribe.utils import load_nemo_config


class NemoModel(NamedTuple):
    """NeMo Model."""

    model: NeuralDiarizer
    output_path: str
    tmp_audio_path: str
    device: str


class DiarizeService:
    """Diarize Service for audio files."""

    def __init__(
        self,
        domain_type: str,
        storage_path: str,
        output_path: str,
        device: str,
        device_index: List[int],
    ) -> None:
        """Initialize the Diarize Service.

        This service uses the NeuralDiarizer from NeMo to diarize audio files.

        Args:
            domain_type (str): Domain type to use for diarization. Can be "general", "telephonic" or "meeting".
            storage_path (str): Path where the diarization pipeline will save temporary files.
            output_path (str): Path where the diarization pipeline will save the final output files.
            device (str): Device to use for inference. Can be "cpu" or "cuda".
            device_index (Union[int, List[int]]): Index of the device to use for inference.
        """
        self.device = device
        self.models = {}

        for idx in device_index:
            _output_path = Path(output_path) / f"output_{idx}"

            _device = f"cuda:{idx}" if self.device == "cuda" else "cpu"
            cfg, tmp_audio_path = load_nemo_config(
                domain_type=domain_type,
                storage_path=storage_path,
                output_path=_output_path,
                device=_device,
                index=idx,
            )
            model = NeuralDiarizer(cfg=cfg).to(_device)
            self.models[idx] = NemoModel(
                model=model,
                output_path=_output_path,
                tmp_audio_path=tmp_audio_path,
                device=_device,
            )

    @time_and_tell
    def __call__(
        self, filepath: Union[str, torch.Tensor], model_index: int
    ) -> List[dict]:
        """
        Run inference with the diarization model.

        Args:
            filepath (Union[str, torch.Tensor]): Path to the audio file or waveform.
            model_index (int): Index of the model to use for inference.

        Returns:
            List[dict]: List of segments with the following keys: "start", "end", "speaker".
        """
        if isinstance(filepath, str):
            waveform, sample_rate = librosa.load(filepath, sr=None)
        else:
            waveform = filepath
            sample_rate = 16000

        sf.write(
            self.models[model_index].tmp_audio_path, waveform, sample_rate, "PCM_16"
        )

        self.models[model_index].model.diarize()

        outputs = self._format_timestamps(self.models[model_index].output_path)

        return outputs

    @staticmethod
    def _format_timestamps(output_path: str) -> List[dict]:
        """
        Format timestamps from the diarization pipeline.

        Args:
            output_path (str): Path where the diarization pipeline saved the final output files.

        Returns:
            List[dict]: List of segments with the following keys: "start", "end", "speaker".
        """
        speaker_timestamps = []

        with open(f"{output_path}/pred_rttms/mono_file.rttm") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_timestamps.append([s, e, int(line_list[11].split("_")[-1])])

        return speaker_timestamps


class ClusterEmbedding(torch.nn.Module):
    """"""
    def __init__(self) -> None:
        super().__init__()
        self._speaker_model = speaker_model
        self.scale_window_length_list = list(
            self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.window_length_in_sec
        )
        self.scale_n = len(self.scale_window_length_list)
        self.base_scale_index = len(self.scale_window_length_list) - 1
        self.clus_diar_model = ClusteringDiarizer(cfg=self.cfg_diar_infer, speaker_model=self._speaker_model)
        self.max_num_speakers = 8

    def prepare_cluster_embs_infer(self):
        """
        Launch clustering diarizer to prepare embedding vectors and clustering results.
        """
        self.emb_sess_test_dict, self.emb_seq_test, self.clus_test_label_dict, _ = self.run_clustering_diarizer(
            "to be removed", "To be removed"
        )

    def run_clustering_diarizer(self, manifest_filepath: str, emb_dir: str):
        """
        If no pre-existing data is provided, run clustering diarizer from scratch. This will create scale-wise speaker embedding
        sequence, cluster-average embeddings, scale mapping and base scale clustering labels. Note that speaker embedding `state_dict`
        is loaded from the `state_dict` in the provided MSDD checkpoint.

        Args:
            manifest_filepath (str):
                Input manifest file for creating audio-to-RTTM mapping.
            emb_dir (str):
                Output directory where embedding files and timestamp files are saved.

        Returns:
            emb_sess_avg_dict (dict):
                Dictionary containing cluster-average embeddings for each session.
            emb_scale_seq_dict (dict):
                Dictionary containing embedding tensors which are indexed by scale numbers.
            base_clus_label_dict (dict):
                Dictionary containing clustering results. Clustering results are cluster labels for the base scale segments.
        """
        scores = self.clus_diar_model.diarize(batch_size=self.cfg_diar_infer.batch_size)

        # If RTTM (ground-truth diarization annotation) files do not exist, scores is None.
        if scores is not None:
            metric, speaker_mapping_dict, _ = scores
        else:
            metric, speaker_mapping_dict = None, None

        # Get the mapping between segments in different scales.
        self._embs_and_timestamps = get_embs_and_timestamps(
            self.clus_diar_model.multiscale_embeddings_and_timestamps, self.clus_diar_model.multiscale_args_dict
        )
        session_scale_mapping_dict = self.get_scale_map(self._embs_and_timestamps)
        emb_scale_seq_dict = self.load_emb_scale_seq_dict(emb_dir)
        clus_labels = self.load_clustering_labels(emb_dir)
        emb_sess_avg_dict, base_clus_label_dict = self.get_cluster_avg_embs(
            emb_scale_seq_dict, clus_labels, speaker_mapping_dict, session_scale_mapping_dict
        )
        emb_scale_seq_dict['session_scale_mapping'] = session_scale_mapping_dict
        return emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict, metric


class NeuralDiarizer(LightningModule):
    """
    Class for inference based on multiscale diarization decoder (MSDD). MSDD requires initializing clustering results from
    clustering diarizer. Overlap-aware diarizer requires separate RTTM generation and evaluation modules to check the effect of
    overlap detection in speaker diarization.
    """

    def __init__(self, cfg: Union[DictConfig, NeuralDiarizerInferenceConfig]):
        super().__init__()
        self._cfg = cfg

        # Parameter settings for MSDD model
        self.use_speaker_model_from_ckpt = cfg.diarizer.msdd_model.parameters.get('use_speaker_model_from_ckpt', True)
        self.use_clus_as_main = cfg.diarizer.msdd_model.parameters.get('use_clus_as_main', False)
        self.max_overlap_spks = cfg.diarizer.msdd_model.parameters.get('max_overlap_spks', 2)
        self.num_spks_per_model = cfg.diarizer.msdd_model.parameters.get('num_spks_per_model', 2)
        self.use_adaptive_thres = cfg.diarizer.msdd_model.parameters.get('use_adaptive_thres', True)
        self.max_pred_length = cfg.diarizer.msdd_model.parameters.get('max_pred_length', 0)
        self.diar_eval_settings = cfg.diarizer.msdd_model.parameters.get(
            'diar_eval_settings', [(0.25, True), (0.25, False), (0.0, False)]
        )

        self._init_msdd_model(cfg)
        self.diar_window_length = cfg.diarizer.msdd_model.parameters.diar_window_length
        self.msdd_model.cfg = self.transfer_diar_params_to_model_params(self.msdd_model, cfg)

        # Initialize clustering and embedding preparation instance (as a diarization encoder).
        self.clustering_embedding = ClusterEmbedding(
            cfg_diar_infer=cfg, cfg_msdd_model=self.msdd_model.cfg, speaker_model=self._speaker_model
        )

        # Parameters for creating diarization results from MSDD outputs.
        self.clustering_max_spks = self.msdd_model._cfg.max_num_of_spks
        self.overlap_infer_spk_limit = cfg.diarizer.msdd_model.parameters.get(
            'overlap_infer_spk_limit', self.clustering_max_spks
        )

    def transfer_diar_params_to_model_params(self, msdd_model, cfg):
        """
        Transfer the parameters that are needed for MSDD inference from the diarization inference config files
        to MSDD model config `msdd_model.cfg`.
        """
        msdd_model.cfg.diarizer.out_dir = cfg.diarizer.out_dir
        msdd_model.cfg.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        msdd_model.cfg.test_ds.emb_dir = cfg.diarizer.out_dir
        msdd_model.cfg.test_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size
        msdd_model.cfg.test_ds.seq_eval_mode = cfg.diarizer.msdd_model.parameters.seq_eval_mode
        msdd_model._cfg.max_num_of_spks = cfg.diarizer.clustering.parameters.max_num_speakers
        return msdd_model.cfg

    @rank_zero_only
    def save_to(self, save_path: str):
        """
        Saves model instances (weights and configuration) into EFF archive.
        You can use "restore_from" method to fully restore instance from .nemo file.

        .nemo file is an archive (tar.gz) with the following:
            model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
            model_wights.chpt - model checkpoint

        Args:
            save_path: Path to .nemo file where model instance should be saved
        """
        self.clus_diar = self.clustering_embedding.clus_diar_model
        _NEURAL_DIAR_MODEL = "msdd_model.nemo"

        with tempfile.TemporaryDirectory() as tmpdir:
            config_yaml = os.path.join(tmpdir, _MODEL_CONFIG_YAML)
            spkr_model = os.path.join(tmpdir, _SPEAKER_MODEL)
            neural_diar_model = os.path.join(tmpdir, _NEURAL_DIAR_MODEL)

            self.clus_diar.to_config_file(path2yaml_file=config_yaml)
            if self.clus_diar.has_vad_model:
                vad_model = os.path.join(tmpdir, _VAD_MODEL)
                self.clus_diar._vad_model.save_to(vad_model)
            self.clus_diar._speaker_model.save_to(spkr_model)
            self.msdd_model.save_to(neural_diar_model)
            self.clus_diar.__make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)

    def extract_standalone_speaker_model(self, prefix: str = 'msdd._speaker_model.') -> EncDecSpeakerLabelModel:
        """
        MSDD model file contains speaker embedding model and MSDD model. This function extracts standalone speaker model and save it to
        `self.spk_emb_state_dict` to be loaded separately for clustering diarizer.

        Args:
            ext (str):
                File-name extension of the provided model path.
        Returns:
            standalone_model_path (str):
                Path to the extracted standalone model without speaker embedding extractor model.
        """
        model_state_dict = self.msdd_model.state_dict()
        spk_emb_module_names = []
        for name in model_state_dict.keys():
            if prefix in name:
                spk_emb_module_names.append(name)

        spk_emb_state_dict = {}
        for name in spk_emb_module_names:
            org_name = name.replace(prefix, '')
            spk_emb_state_dict[org_name] = model_state_dict[name]

        _speaker_model = EncDecSpeakerLabelModel.from_config_dict(self.msdd_model.cfg.speaker_model_cfg)
        _speaker_model.load_state_dict(spk_emb_state_dict)
        return _speaker_model

    def _init_msdd_model(self, cfg: Union[DictConfig, NeuralDiarizerInferenceConfig]):

        """
        Initialized MSDD model with the provided config. Load either from `.nemo` file or `.ckpt` checkpoint files.
        """
        model_path = cfg.diarizer.msdd_model.model_path
        if model_path.endswith('.nemo'):
            logging.info(f"Using local nemo file from {model_path}")
            self.msdd_model = EncDecDiarLabelModel.restore_from(restore_path=model_path, map_location=cfg.device)
        elif model_path.endswith('.ckpt'):
            logging.info(f"Using local checkpoint from {model_path}")
            self.msdd_model = EncDecDiarLabelModel.load_from_checkpoint(
                checkpoint_path=model_path, map_location=cfg.device
            )
        else:
            if model_path not in get_available_model_names(EncDecDiarLabelModel):
                logging.warning(f"requested {model_path} model name not available in pretrained models, instead")
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self.msdd_model = EncDecDiarLabelModel.from_pretrained(model_name=model_path, map_location=cfg.device)
        # Load speaker embedding model state_dict which is loaded from the MSDD checkpoint.
        if self.use_speaker_model_from_ckpt:
            self._speaker_model = self.extract_standalone_speaker_model()
        else:
            self._speaker_model = None

    def get_pred_mat(self, data_list: List[Union[Tuple[int], List[torch.Tensor]]]) -> torch.Tensor:
        """
        This module puts together the pairwise, two-speaker, predicted results to form a finalized matrix that has dimension of
        `(total_len, n_est_spks)`. The pairwise results are evenutally averaged. For example, in 4 speaker case (speaker 1, 2, 3, 4),
        the sum of the pairwise results (1, 2), (1, 3), (1, 4) are then divided by 3 to take average of the sigmoid values.

        Args:
            data_list (list):
                List containing data points from `test_data_collection` variable. `data_list` has sublists `data` as follows:
                data[0]: `target_spks` tuple
                    Examples: (0, 1, 2)
                data[1]: Tensor containing estimaged sigmoid values.
                   [[0.0264, 0.9995],
                    [0.0112, 1.0000],
                    ...,
                    [1.0000, 0.0512]]

        Returns:
            sum_pred (Tensor):
                Tensor containing the averaged sigmoid values for each speaker.
        """
        all_tups = tuple()
        for data in data_list:
            all_tups += data[0]
        n_est_spks = len(set(all_tups))
        digit_map = dict(zip(sorted(set(all_tups)), range(n_est_spks)))
        total_len = max([sess[1].shape[1] for sess in data_list])
        sum_pred = torch.zeros(total_len, n_est_spks)
        for (_dim_tup, pred_mat) in data_list:
            dim_tup = [digit_map[x] for x in _dim_tup]
            if len(pred_mat.shape) == 3:
                pred_mat = pred_mat.squeeze(0)
            if n_est_spks <= self.num_spks_per_model:
                sum_pred = pred_mat
            else:
                _end = pred_mat.shape[0]
                sum_pred[:_end, dim_tup] += pred_mat.cpu().float()
        sum_pred = sum_pred / (n_est_spks - 1)
        return sum_pred

    def get_integrated_preds_list(
        self, uniq_id_list: List[str], test_data_collection: List[Any], preds_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Merge multiple sequence inference outputs into a session level result.

        Args:
            uniq_id_list (list):
                List containing `uniq_id` values.
            test_data_collection (collections.DiarizationLabelEntity):
                Class instance that is containing session information such as targeted speaker indices, audio filepaths and RTTM filepaths.
            preds_list (list):
                List containing tensors filled with sigmoid values.

        Returns:
            output_list (list):
                List containing session-level estimated prediction matrix.
        """
        session_dict = get_id_tup_dict(uniq_id_list, test_data_collection, preds_list)
        output_dict = {uniq_id: [] for uniq_id in uniq_id_list}
        for uniq_id, data_list in session_dict.items():
            sum_pred = self.get_pred_mat(data_list)
            output_dict[uniq_id] = sum_pred.unsqueeze(0)
        output_list = [output_dict[uniq_id] for uniq_id in uniq_id_list]
        return output_list

    def get_emb_clus_infer(self, cluster_embeddings):
        """Assign dictionaries containing the clustering results from the class instance `cluster_embeddings`.
        """
        self.msdd_model.emb_sess_test_dict = cluster_embeddings.emb_sess_test_dict
        self.msdd_model.clus_test_label_dict = cluster_embeddings.clus_test_label_dict
        self.msdd_model.emb_seq_test = cluster_embeddings.emb_seq_test

    @torch.no_grad()
    def diarize(self) -> Optional[List[Optional[List[Tuple[DiarizationErrorRate, Dict]]]]]:
        """
        Launch diarization pipeline which starts from VAD (or a oracle VAD stamp generation), initialization clustering and multiscale diarization decoder (MSDD).
        Note that the result of MSDD can include multiple speakers at the same time. Therefore, RTTM output of MSDD needs to be based on `make_rttm_with_overlap()`
        function that can generate overlapping timestamps. `self.run_overlap_aware_eval()` function performs DER evaluation.
        """
        self.clustering_embedding.prepare_cluster_embs_infer()
        self.msdd_model.pairwise_infer = True
        self.get_emb_clus_infer(self.clustering_embedding)
        preds_list, targets_list, signal_lengths_list = self.run_pairwise_diarization()
        thresholds = list(self._cfg.diarizer.msdd_model.parameters.sigmoid_threshold)
        return [self.run_overlap_aware_eval(preds_list, threshold) for threshold in thresholds]
