import json
from typing import Union

import numpy as np
from loguru import logger


def compare_starting_timestamps(
    api_output: Union[str, dict], reference_timestamp: float, tolerance: float = 5e-1
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


def launch_timestamps_comparison(args):
    """Launch the timestamps comparison subcommand."""
    return TimestampsComparison(args.json, args.ref, args.tolerance)


class TimestampsComparison:
    @staticmethod
    def register_subcommand(parser) -> None:
        """Register the subcommand."""
        subparser = parser.add_parser("timestamps", help="Compare the timestamps of the API output with the reference timestamps.")
        subparser.add_argument("-j", "--json", type=str, required=True, help="Path to the API output json file.")
        subparser.add_argument("-r", "--ref", type=float, required=True, help="Reference timestamp of the audio starting point in seconds.")
        subparser.add_argument("-t", "--tolerance", type=float, default=5e-1, help="Tolerance in seconds. Defaults to 0.5 seconds.")
        subparser.set_defaults(func=launch_timestamps_comparison)

    def __init__(
        self, api_output: Union[str, dict], reference_timestamp: float, tolerance: float = 5e-1
    ) -> None:
        """Initialize the subcommand."""
        self.api_output = api_output
        self.reference_timestamp = reference_timestamp
        self.tolerance = tolerance

    def run(self) -> bool:
        """Run the subcommand."""
        return compare_starting_timestamps(
            self.api_output, self.reference_timestamp, self.tolerance
        )
