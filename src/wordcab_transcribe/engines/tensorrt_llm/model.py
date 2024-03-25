import os

import ctranslate2
import numpy as np

from wordcab_transcribe.engines.tensorrt_llm.engine_builder.create_trt_model import (
    build_whisper_trt_model,
)
from wordcab_transcribe.engines.tensorrt_llm.hf_utils import download_model
from wordcab_transcribe.engines.tensorrt_llm.tokenizers import Tokenizer
from wordcab_transcribe.engines.tensorrt_llm.trt_model import WhisperTRT
from wordcab_transcribe.engines.tensorrt_llm.whisper_model import WhisperModel


def exact_div(x, y):
    assert x % y == 0
    return x // y


N_FFT = 400
INPUT_STRIDE = 2
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
SAMPLE_RATE = 16000
MAX_TEXT_TOKEN_LENGTH = 448
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * INPUT_STRIDE  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token
TIME_PRECISION = 1 / TOKENS_PER_SECOND


FAST_ASR_OPTIONS = {
    "beam_size": 1,
    "best_of": 1,
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1.01,
    "no_repeat_ngram_size": 0,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.5,
    "prefix": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 1.0,
    "word_timestamps": False,
    "sampling_temperature": 1.0,
    "return_scores": True,
    "return_no_speech_prob": True,
    "word_aligner_model": "tiny",
}

BEST_ASR_CONFIG = {
    "beam_size": 5,
    "best_of": 1,
    "patience": 2,
    "length_penalty": 1,
    "repetition_penalty": 1.01,
    "no_repeat_ngram_size": 0,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.5,
    "prefix": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 1.0,
    "word_timestamps": True,
    "sampling_temperature": 1.0,
    "return_scores": True,
    "return_no_speech_prob": True,
    "word_aligner_model": "tiny",
}


class WhisperModelTRT(WhisperModel):
    """TensorRT-LLM implementation of the Whisper model."""

    def __init__(
        self,
        model_name: str,
        asr_options: dict,
        cpu_threads=4,
        num_workers=1,
        device="cuda",
        device_index=0,
        compute_type="float16",
        max_text_token_len=15,
        **model_kwargs
    ):
        self.asr_options = FAST_ASR_OPTIONS
        self.asr_options.update(asr_options)
        self.model_name = model_name
        self.model_path = os.path.join("models", self.model_name)

        if not os.path.exists(self.model_path):
            self.model_path = build_whisper_trt_model(
                self.model_path, model_name=self.model_name
            )
        self.model = WhisperTRT(self.model_path)

        tokenizer_file = os.path.join(self.model_path, "tokenizer.json")
        tokenizer = Tokenizer(
            Tokenizer.from_file(tokenizer_file), self.model.is_multilingual
        )

        if self.asr_options["word_timestamps"]:
            # TODO: Option to load word aligner model from file
            self.aligner_model_path = download_model(
                self.asr_options["word_aligner_model"]
            )
            self.aligner_model = ctranslate2.models.Whisper(
                self.aligner_model_path,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                intra_threads=cpu_threads,
                inter_threads=num_workers,
            )

        self.generate_kwargs = {
            "end_id": tokenizer.eot,
            "pad_id": tokenizer.eot,
            "max_new_tokens": max_text_token_len,
            "length_penalty": self.asr_options["length_penalty"],
            "repetition_penalty": self.asr_options["repetition_penalty"],
            "num_beams": self.asr_options["beam_size"],
            "stop_words_list": self.asr_options["suppress_blank"],
            "bad_words_list": self.asr_options["suppress_tokens"],
            "temperature": self.asr_options["sampling_temperature"],
        }

        super().__init__(
            tokenizer=tokenizer,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            max_text_token_len=max_text_token_len,
            **model_kwargs
        )

    def update_generation_kwargs(self, params: dict):
        self.generate_kwargs.update(params)

        if "max_text_token_len" in params:
            self.update_params(
                params={"max_text_token_len": params["max_text_token_len"]}
            )

    def encode(self, features):
        """
        [Not Used]
        """
        return self.model.encode(features)

    def assign_word_timings(self, alignments, text_token_probs, words, word_tokens):
        text_indices = np.array([pair[0] for pair in alignments])
        time_indices = np.array([pair[1] for pair in alignments])

        if len(word_tokens) <= 1:
            return []

        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        if len(word_boundaries) <= 1:
            return []

        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] * TIME_PRECISION
        start_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:]]
        word_probs = [
            np.mean(text_token_probs[i:j])
            for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
        ]

        return [
            {
                "word": word,
                "start": round(start, 2),
                "end": round(end, 2),
                "prob": round(prob, 2),
            }
            for word, start, end, prob in zip(words, start_times, end_times, word_probs)
        ]

    def align_words(
        self, features, texts, text_tokens, sot_seqs, seq_lens, seg_metadata
    ):
        lang_codes = [_["lang_code"] for _ in seg_metadata]
        word_tokens = self.tokenizer.split_to_word_tokens_batch(
            texts, text_tokens, lang_codes
        )

        start_seq_wise_req = {}
        for _idx, _sot_seq in enumerate(sot_seqs):
            try:
                # print(_sot_seq)
                start_seq_wise_req[_sot_seq].append(_idx)
            except Exception:
                start_seq_wise_req[_sot_seq] = [_idx]

        token_alignments = [[] for _ in seg_metadata]
        for start_seq, req_idx in start_seq_wise_req.items():
            res = self.aligner_model.align(
                ctranslate2.StorageView.from_array(features[req_idx]),
                start_sequence=list(start_seq),
                text_tokens=[text_tokens[_] for _ in req_idx],
                num_frames=list(seq_lens[req_idx].detach().cpu().numpy()),
                median_filter_width=7,
            )

            for _res, _req_idx in zip(res, req_idx):
                token_alignments[_req_idx] = _res

        word_timings = []
        for _idx, _seg_metadata in enumerate(seg_metadata):
            _word_timings = self.assign_word_timings(
                token_alignments[_idx].alignments,
                token_alignments[_idx].text_token_probs,
                word_tokens[_idx][0],
                word_tokens[_idx][1],
            )

            stitched_seg = _seg_metadata["stitched_seg"]

            current_seg_idx = 0
            current_offset = _seg_metadata["start_time"]

            for w in _word_timings:
                while (w["start"] + current_offset) >= stitched_seg[current_seg_idx][1]:
                    current_seg_idx += 1
                    current_offset += (
                        stitched_seg[current_seg_idx][0]
                        - stitched_seg[current_seg_idx - 1][1]
                    )

                w["start"] += current_offset
                w["end"] += current_offset

            word_timings.append(_word_timings)

        return word_timings

    def generate_segment_batched(self, features, prompts, seq_lens, seg_metadata):
        result = self.model.generate(features, prompts, **self.generate_kwargs)

        texts = self.tokenizer.decode_batch([x[0] for x in result])

        response = []
        for idx, _r in enumerate(result):
            response.append({"text": texts[idx].strip()})

        if self.asr_options["word_timestamps"]:
            text_tokens = [
                [_t for _t in x[0] if _t < self.tokenizer.eot] + [self.tokenizer.eot]
                for x in result
            ]
            sot_seqs = [tuple(_[-4:]) for _ in prompts]
            word_timings = self.align_words(
                features, texts, text_tokens, sot_seqs, seq_lens, seg_metadata
            )

            for _response, _word_timings in zip(response, word_timings):
                _response["word_timestamps"] = _word_timings

        return response
