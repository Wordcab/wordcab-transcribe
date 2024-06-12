import os
import subprocess
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "distil-large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "distil-large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}

_TOKENIZERS = {
    "tiny.en": (
        "https://huggingface.co/Systran/faster-whisper-tiny.en/raw/main/tokenizer.json"
    ),
    "tiny": (
        "https://huggingface.co/Systran/faster-whisper-tiny/raw/main/tokenizer.json"
    ),
    "small.en": (
        "https://huggingface.co/Systran/faster-whisper-small.en/raw/main/tokenizer.json"
    ),
    "small": (
        "https://huggingface.co/Systran/faster-whisper-small/raw/main/tokenizer.json"
    ),
    "base.en": (
        "https://huggingface.co/Systran/faster-whisper-base.en/raw/main/tokenizer.json"
    ),
    "base": (
        "https://huggingface.co/Systran/faster-whisper-base/raw/main/tokenizer.json"
    ),
    "medium.en": "https://huggingface.co/Systran/faster-whisper-medium.en/raw/main/tokenizer.json",
    "medium": (
        "https://huggingface.co/Systran/faster-whisper-medium/raw/main/tokenizer.json"
    ),
    "large-v1": (
        "https://huggingface.co/Systran/faster-whisper-large-v1/raw/main/tokenizer.json"
    ),
    "large-v2": (
        "https://huggingface.co/Systran/faster-whisper-large-v2/raw/main/tokenizer.json"
    ),
    "large-v3": (
        "https://huggingface.co/Systran/faster-whisper-large-v3/raw/main/tokenizer.json"
    ),
    "distil-large-v2": "https://huggingface.co/Systran/faster-distil-whisper-large-v2/raw/main/tokenizer.json",
    "distil-large-v3": "https://huggingface.co/Systran/faster-distil-whisper-large-v3/raw/main/tokenizer.json",
    "large": (
        "https://huggingface.co/Systran/faster-whisper-large-v3/raw/main/tokenizer.json"
    ),
}


TRT_BUILD_MAX_OUTPUT_LEN = os.getenv("TRT_BUILD_MAX_OUTPUT_LEN", None)
TRT_BUILD_MAX_BEAM_WIDTH = os.getenv("TRT_BUILD_MAX_BEAM_WIDTH", None)
if not TRT_BUILD_MAX_OUTPUT_LEN:
    TRT_BUILD_MAX_OUTPUT_LEN = 448
else:
    TRT_BUILD_MAX_OUTPUT_LEN = int(TRT_BUILD_MAX_OUTPUT_LEN)
logger.info(f"TRT_BUILD_MAX_OUTPUT_LEN: {TRT_BUILD_MAX_OUTPUT_LEN}")

if not TRT_BUILD_MAX_BEAM_WIDTH:
    TRT_BUILD_MAX_BEAM_WIDTH = 1
else:
    TRT_BUILD_MAX_BEAM_WIDTH = int(TRT_BUILD_MAX_BEAM_WIDTH)
logger.info(f"TRT_BUILD_MAX_BEAM_WIDTH: {TRT_BUILD_MAX_BEAM_WIDTH}")


def build_whisper_trt_model(
    output_dir,
    use_gpt_attention_plugin=True,
    use_gemm_plugin=True,
    use_bert_attention_plugin=True,
    enable_context_fmha=True,
    use_weight_only=False,
    max_output_len=TRT_BUILD_MAX_OUTPUT_LEN,
    max_beam_width=TRT_BUILD_MAX_BEAM_WIDTH,
    model_name="large-v3",
):
    """
    Build a Whisper model using the specified configuration.

    Args:
        output_dir (str): The output directory where the model will be saved.
        use_gpt_attention_plugin (bool): Whether to use the GPT attention plugin.
        use_gemm_plugin (bool): Whether to use the GEMM plugin.
        use_bert_attention_plugin (bool): Whether to use the BERT attention plugin.
        enable_context_fmha (bool): Whether to enable context FMHA.
        use_weight_only (bool): Whether to use int8 weight-only quantization.
        model_name (str): The name of the model to build (default: "large-v3").

    Returns:
        None
    """
    model_url = _MODELS[model_name]
    model_url.split("/")[-2]
    model_ckpt_dir = Path(__file__).parent.parent / "assets"
    model_ckpt_path = model_ckpt_dir / f"{model_name}.pt"
    tokenizer_path = output_dir / "tokenizer.json"

    if not output_dir.exists() and not model_ckpt_path.exists():
        if "distil" in model_name:
            logger.info(f"Downloading and converting distilled model '{model_name}'...")
            convert_file = Path(__file__).parent / "convert_from_distil_whisper.py"
            command = [
                "python",
                str(convert_file),
                "--output_dir",
                str(model_ckpt_dir),
                "--model_name",
                f"distil-whisper/{model_name}",
                "--output_name",
                model_name,
            ]

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Error occurred while converting the distilled model: {e}"
                )
                raise
        else:
            logger.info(f"Downloading model '{model_name}' from {model_url}...")
            response = requests.get(model_url, stream=True)
            total_size = int(response.headers.get("Content-Length", 0))

            with open(model_ckpt_path, "wb") as output:
                with tqdm(
                    total=total_size,
                    ncols=80,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for data in response.iter_content(chunk_size=8192):
                        size = output.write(data)
                        pbar.update(size)
        logger.info(f"Model '{model_name}' has been downloaded successfully.")

    logger.info(f"output_dir: {output_dir} | Exists: {os.path.exists(output_dir)}")
    if not output_dir.exists():
        logger.info("Building the model...")
        build_file = Path(__file__).parent / "build.py"
        command = [
            "python",
            str(build_file),
            "--output_dir",
            str(output_dir),
            "--model_name",
            model_name,
            "--model_dir",
            model_ckpt_dir,
        ]

        if use_gpt_attention_plugin:
            command.append("--use_gpt_attention_plugin")
        if use_gemm_plugin:
            command.append("--use_gemm_plugin")
        if use_bert_attention_plugin:
            command.append("--use_bert_attention_plugin")
        if enable_context_fmha:
            command.append("--enable_context_fmha")
        if use_weight_only:
            command.append("--use_weight_only")
        if max_output_len:
            command.extend(["--max_output_len", str(max_output_len)])
        if max_beam_width:
            command.extend(["--max_beam_width", str(max_beam_width)])

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error occurred while building the model: {e}")
            raise
        logger.info("Model has been built successfully.")

    if not tokenizer_path.exists():
        logger.info(f"Downloading tokenizer for model '{model_name}'...")
        response = requests.get(_TOKENIZERS[model_name], stream=True)
        total_size = int(response.headers.get("Content-Length", 0))

        with open(tokenizer_path, "wb") as output:
            with tqdm(
                total=total_size,
                ncols=80,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=8192):
                    size = output.write(data)
                    pbar.update(size)
        logger.info("Tokenizer has been downloaded successfully.")

    for filename in os.listdir(output_dir):
        if "encoder" in filename and filename.endswith(".engine"):
            new_filename = "encoder.engine"
            old_path = os.path.join(output_dir, filename)
            new_path = os.path.join(output_dir, new_filename)
            os.rename(old_path, new_path)
            logger.info(f"Renamed '{filename}' to '{new_filename}'")
        elif "decoder" in filename and filename.endswith(".engine"):
            new_filename = "decoder.engine"
            old_path = os.path.join(output_dir, filename)
            new_path = os.path.join(output_dir, new_filename)
            os.rename(old_path, new_path)
            logger.info(f"Renamed '{filename}' to '{new_filename}'")

    return output_dir
