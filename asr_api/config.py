# Copyright (c) 2023, The Wordcab team. All rights reserved.
"""Configuration module of the Wordcab ASR API."""

from os import getenv
from dotenv import load_dotenv

from pydantic import BaseSettings


class Settings(BaseSettings):
    project_name: str
    version: str
    description: str
    api_prefix: str
    debug: bool
    


load_dotenv()

settings = Settings(
    project_name=getenv("PROJECT_NAME"),
    version=getenv("VERSION"),
    description=getenv("DESCRIPTION"),
    api_prefix=getenv("API_PREFIX"),
    debug=getenv("DEBUG"),
)