"""Configuration, loaded from .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: str = ""
    # NOT gemini-3.5-flash: its free tier is 20 requests PER DAY, and one agent
    # turn costs 2-3. Measured, not guessed -- the 429 names the quota
    # explicitly. The -lite tier is far more generous and still calls tools.
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_rpm_guess: int = 10
    # Deliberately conservative. Google no longer publishes free-tier limits, so
    # the quota store LEARNS the real ceiling from 429s rather than trusting this.
    gemini_rpd_guess: int = 200

    groq_api_key: str = ""
    # NOT llama-3.3-70b-versatile: Groq rejects its tool calls server-side
    # ("Failed to call a function"), and the model leaks `<function=...>` as
    # plain text into the answer. Verified working alternatives with real tool
    # schemas: openai/gpt-oss-20b and llama-3.1-8b-instant.
    groq_model: str = "openai/gpt-oss-20b"

    ollama_host: str = "http://localhost:11434"
    ollama_voice_model: str = "mehul-voice"
    ollama_base_model: str = "qwen3:1.7b"

    mehullm_api_token: str = ""
    mehullm_host: str = "127.0.0.1"
    mehullm_port: int = 8000
    mehullm_cors_origins: str = "http://localhost:3000"

    mehullm_db: str = str(ROOT / "data" / "derived" / "mehullm.db")
    memory_db: str = str(ROOT / "data" / "derived" / "memory.db")
    memory_facts_k: int = 8
    mehullm_derived_dir: str = str(ROOT / "data" / "derived")

    mehullm_max_tool_calls_per_turn: int = 25
    mehullm_max_turn_seconds: int = 180
    mehullm_confirm_timeout_s: int = 120
    mehullm_max_daily_requests: int = 200

    servers_config: str = str(ROOT / "config" / "servers.yaml")
    policy_config: str = str(ROOT / "config" / "policy.yaml")

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.mehullm_cors_origins.split(",") if o.strip()]


settings = Settings()
