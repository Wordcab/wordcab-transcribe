# This script converts a whisper model to a ctranslate2 model.
# Use this script to convert a model from the HuggingFace Hub to a ctranslate2 model.
# E.g. `python convert_whisper_model.py -p openai/whisper-large-v2 -o whisper-large-v2 -q int8_float16`
#
# The script requires the following dependencies:
# - ctranslate2
# - transformers[torch]
"""Convert a whisper model to ctranslate2."""

import argparse
import importlib
import subprocess  # noqa: S404


def check_dependency(module_name: str) -> None:
    """
    Check if a module is installed. If not, exit with an error message.

    Args:
        module_name (str): Name of the module to check.
    """
    try:
        importlib.import_module(module_name)

    except ImportError:
        print(
            f"Error: {module_name} module not found. Please make sure it is installed and try again."
        )
        exit(1)


if __name__ == "__main__":
    check_dependency("ctranslate2")
    check_dependency("transformers")

    parser = argparse.ArgumentParser(
        description="Convert a whisper model to ctranslate2"
    )

    parser.add_argument(
        "-p",
        "--model_path",
        type=str,
        help="Path to the whisper model stored on the HuggingFace Hub.",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, help="Path to the output directory."
    )
    parser.add_argument(
        "-q",
        "--quantization",
        default=None,
        type=str,
        help="Quantization type. Can be 'int8', 'int8_float16', 'int16' or 'float16'.",
    )

    args = parser.parse_args()

    # Check if the quantization type is valid.
    if args.quantization and args.quantization not in [
        "int8",
        "int8_float16",
        "int16",
        "float16",
    ]:
        print(
            f"Error: {args.quantization} is not a valid quantization type."
            "Please choose between 'int8', 'int8_float16', 'int16' or 'float16'."
            "If you don't want to quantize the model, don't use the '-q' option."
        )
        exit(1)

    command = [
        "ct2-transformers-converter",
        "--model",
        args.model_path,
        "--output_dir",
        args.output_path,
    ]
    if args.quantization:
        command.extend(["--quantization", args.quantization])

    process = subprocess.Popen(  # noqa: S603,S607
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error: {stderr.decode('utf-8')}")
        exit(1)
