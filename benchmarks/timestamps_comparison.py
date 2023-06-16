import json
from typing import Union

import numpy as np


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
        api_output = json.loads(api_output)

    utterances = api_output.get("utterances", [])
    if not utterances:
        raise ValueError("No utterances found in the API output. Cannot compare timestamps.")

    first_timestamp = utterances[0]["start"]
    if np.allclose(first_timestamp, reference_timestamp, atol=tolerance):
        return True
    else:
        return False
