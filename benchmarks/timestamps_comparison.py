import json
from typing import Union

import numpy as np
from loguru import logger


def compare_starting_timestamps(
    api_output: Union[str, dict], reference_timestamp: float, tolerance: float = 1e-2
) -> bool:
    """
    Compare the starting timestamps in the API output with the reference timestamp.

    Args:
        api_output: API output.
        reference_timestamp: Reference timestamp.
        tolerance: Tolerance in seconds.

    Returns:
        True if the starting timestamps are equal, False otherwise.

    Raises:
        ValueError: If no utterances are found in the API output.
    """
    if isinstance(api_output, str):
        with open(api_output, "r") as f:
            api_output = json.load(f)

    utterances = api_output.get("utterances", [])
    if not utterances:
        raise ValueError("No utterances found in the API output. Cannot compare timestamps.")

    first_timestamp = utterances[0]["start"]
    if np.allclose(first_timestamp, reference_timestamp, atol=tolerance):
        logger.info(f"First timestamp {first_timestamp} matches the reference timestamp {reference_timestamp} within a tolerance of {tolerance} seconds.")
        return True
    else:
        logger.error(f"First timestamp {first_timestamp} does not match the reference timestamp {reference_timestamp}.")
        return False
    
compare_starting_timestamps("data/nissan_sample_5.json", 11.8)
