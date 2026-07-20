import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

models = {
    "claude": "claude-sonnet-4.6",
    "gemini3.1pro": "gemini-3.1-pro-preview",
    "gpt5mini": "gpt-5-mini",
    "gemini3.1flash": "gemini-3.1-flash-lite-preview",
    "glm5": "glm-5",
    "flash": "gemini-3-flash-preview",
    "gpt5.1": "gpt-5.1",
}


class Config(BaseSettings):
    MODEL_TYPE: Literal["ollama", "ai_tunnel"]
    MODEL_NAME: str
    MODEL_SUMMARY_NAME: str
    USE_LANGFUSE: bool
    TEMPERATURE: float | None
    CHECKPOINTER: Literal["InMemorySaver", "SqliteSaver"]

    AI_TUNNEL_API_KEY: str
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


cfg = Config(
    MODEL_TYPE="ai_tunnel",
    MODEL_NAME=models["gemini3.1pro"],
    MODEL_SUMMARY_NAME=models["gemini3.1flash"],
    USE_LANGFUSE=True,
    TEMPERATURE=None,
    CHECKPOINTER="SqliteSaver",
)


if __name__ == "__main__":
    print(cfg)
