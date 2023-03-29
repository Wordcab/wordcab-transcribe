# Copyright (c) 2023, The Wordcab team. All rights reserved.
"""Models module of the Wordcab ASR API."""

from pydantic import BaseModel
from typing import List, Optional


class ASRRequest(BaseModel):
    """Request model for the ASR API."""
    url: Optional[str] = None
    num_speakers: Optional[int] = None


class ASRResponse(BaseModel):
    """Response model for the ASR API."""
    text: List[dict]
