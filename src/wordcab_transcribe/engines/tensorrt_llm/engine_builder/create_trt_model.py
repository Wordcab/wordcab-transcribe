import os
import subprocess

import requests

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
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}


def build_whisper_model(
    output_dir,
    use_gpt_attention_plugin=True,
    use_gemm_plugin=True,
    use_bert_attention_plugin=True,
    enable_context_fmha=True,
    use_weight_only=False,
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
    model_path = f"assets/{model_name}.pt"

    # Download the model if it doesn't exist
    if not os.path.exists(model_path):
        os.makedirs("assets", exist_ok=True)

        print(f"Downloading model '{model_name}' from {model_url}...")
        response = requests.get(model_url)

        if response.status_code == 200:
            with open(model_path, "wb") as file:
                file.write(response.content)
            print(f"Model '{model_name}' downloaded successfully.")
        else:
            print(
                f"Failed to download model '{model_name}'. Status code:"
                f" {response.status_code}"
            )
            return

    command = ["python3", "build.py", "--output_dir", output_dir]

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

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while building the model: {e}")
        raise
