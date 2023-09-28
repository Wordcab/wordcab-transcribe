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
"""Custom Pydantic annotations for the Wordcab Transcribe API."""

from typing import Any

import torch
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema


class TorchTensorPydanticAnnotation:
    """Pydantic annotation for torch.Tensor."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Custom validation and serialization for torch.Tensor."""

        def validate_tensor(value) -> torch.Tensor:
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Expected a torch.Tensor but got {type(value)}")
            return value

        return core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(validate_tensor),
            ]
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # This is a custom representation for the tensor in JSON Schema.
        # Here, it represents a tensor as an object with metadata.
        return {
            "type": "object",
            "properties": {
                "dtype": {
                    "type": "string",
                    "description": "Data type of the tensor",
                },
                "shape": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Shape of the tensor",
                },
            },
            "required": ["dtype", "shape"],
        }
