# Copyright (c) 2023, The Wordcab team. All rights reserved.
"""Models module of the Wordcab ASR API."""

from pydantic import BaseModel


class ASRRequest(BaseModel):
    """Request model for the ASR API."""
    audio: bytes


class ASRResponse(BaseModel):
    """Response model for the ASR API."""
    text: str
