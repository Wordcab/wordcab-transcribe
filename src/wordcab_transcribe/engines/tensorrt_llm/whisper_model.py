from abc import ABC, abstractmethod

import torch

from wordcab_transcribe.engines.tensorrt_llm.audio import LogMelSpectogram
from wordcab_transcribe.engines.tensorrt_llm.data import WhisperTRTDataLoader
from wordcab_transcribe.engines.tensorrt_llm.segmenter import SpeechSegmenter


class NoneTokenizer:
    """A dummy tokenizer that does nothing."""

    def __init__(self):
        self.sot_prev = 0
        self.silent_token = 0
        self.no_timestamps = 0
        self.timestamp_begin = 0

    def sot_sequence(self, task=None, lang=None):
        return [task, lang]

    @staticmethod
    def encode(self):
        return [0]


def fix_batch_param(param, default_value, N):
    if param is None:
        param = N * [default_value]
    elif type(param) == type(default_value):
        param = N * [param]
    elif len(param) != N:
        param = N * [param[0]]

    return param


class WhisperModel(ABC):
    """
    A Whisper model class for the WhisperTRT engine.

    Args:
        tokenizer: A tokenizer object.
        vad_model: A VAD model object.
        n_mels: Number of mel filters.
        device: Device to run the model on.
        device_index: Index of the device.
        compute_type: Type of computation.
        merge_chunks: Merge chunks.
        dta_padding: DTA padding.
        use_dynamic_time_axis: Use dynamic time axis.
        max_speech_len: Maximum speech length.
        max_text_token_len: Maximum text token length.
        without_timestamps: Without timestamps.
        speech_segmenter_options: Speech segmenter options.
    """

    def __init__(
        self,
        speech_segmenter_options: dict,
        tokenizer=None,
        vad_model=None,
        n_mels=80,
        device="cuda",
        device_index=0,
        compute_type="float16",
        merge_chunks=True,
        dta_padding=3.0,
        use_dynamic_time_axis=False,
        max_speech_len=29.0,
        max_text_token_len=448,
        without_timestamps=True,
    ):
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type

        self.n_mels = n_mels
        self.merge_chunks = merge_chunks
        self.max_speech_len = max_speech_len

        self.dta_padding = dta_padding
        self.use_dynamic_time_axis = use_dynamic_time_axis

        self.without_timestamps = without_timestamps
        self.max_text_token_len = max_text_token_len

        self.vad_model = vad_model
        self.speech_segmenter_options = speech_segmenter_options
        self.speech_segmenter_options["max_seg_len"] = self.max_speech_len

        if tokenizer is None:
            tokenizer = NoneTokenizer()

        self.tokenizer = tokenizer
        self._init_dependables()

    def _init_dependables(self):
        self.dta_padding = int(self.dta_padding * 16000)
        self.max_initial_prompt_len = self.max_text_token_len // 2 - 1
        self.preprocessor = LogMelSpectogram(n_mels=self.n_mels).to(self.device)
        self.speech_segmenter = SpeechSegmenter(
            self.vad_model, device=self.device, **self.speech_segmenter_options
        )

        self.data_loader = WhisperTRTDataLoader(
            self.device,
            self.tokenizer,
            self.speech_segmenter,
            dta_padding=self.dta_padding,
            without_timestamps=self.without_timestamps,
            max_speech_len=self.max_speech_len,
            max_initial_prompt_len=self.max_initial_prompt_len,
            use_dynamic_time_axis=self.use_dynamic_time_axis,
            merge_chunks=self.merge_chunks,
        )

    def update_params(self, params: dict):
        for key, value in params.items():
            setattr(self, key, value)

        self._init_dependables()

    @abstractmethod
    def generate_segment_batched(self, features, prompts):
        pass

    @torch.no_grad()
    def transcribe(
        self,
        audio_data,
        lang_codes=None,
        tasks=None,
        initial_prompts=None,
        batch_size=8,
        use_vad=True,
    ):
        lang_codes = fix_batch_param(lang_codes, "en", len(audio_data))
        tasks = fix_batch_param(tasks, "transcribe", len(audio_data))
        initial_prompts = fix_batch_param(initial_prompts, None, len(audio_data))
        responses = [[] for _ in audio_data]
        for signals, prompts, seq_len, seg_metadata in self.data_loader(
            audio_data,
            lang_codes,
            tasks,
            initial_prompts,
            batch_size=batch_size,
            use_vad=use_vad,
        ):
            mels, seq_len = self.preprocessor(signals, seq_len)
            res = self.generate_segment_batched(
                mels.to(self.device), prompts, seq_len, seg_metadata
            )
            for res_idx, _seg_metadata in enumerate(seg_metadata):
                responses[_seg_metadata["file_id"]].append(
                    {
                        **res[res_idx],
                        "start_time": round(_seg_metadata["start_time"], 3),
                        "end_time": round(_seg_metadata["end_time"], 3),
                    }
                )
        return responses
