import os
from dataclasses import dataclass
from typing import Set

def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    return api_key

def get_base_url() -> str:
    api_key = os.getenv("SEARXNG_BASE_URL")
    if not api_key:
        raise ValueError("URL for SearXNG Server not found. Please set the SEARXNG_BASE_URL environment variable.")
    return api_key

def get_sear_key() -> str:
    api_key = os.getenv("SEARXNG_API_KEY")
    if not api_key:
        raise ValueError("SearXNG API key not found. Please set the SEARXNG_API_KEY environment variable.")
    return api_key

@dataclass
class ChatConfig:
    """Configuration for the chat application"""

    api_key: str = get_api_key()  # This becomes a class variable
    model: str = "gpt-5-mini"
    reasoning_effort: str = "low"
    exit_commands: Set[str] = frozenset({"/exit", "/quit"})
    searxng_base_url: str = get_base_url()
    searxng_api_key: str = get_sear_key()

    def __init__(self):
        # Prevent instantiation
        raise TypeError("ChatConfig is not meant to be instantiated")