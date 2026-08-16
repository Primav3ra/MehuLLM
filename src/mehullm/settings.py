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
    gemini_model: str = "gemini-3.5-flash"
    gemini_rpm_guess: int = 10
    gemini_rpd_guess: int = 250

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    ollama_host: str = "http://localhost:11434"
    ollama_voice_model: str = "mehul-voice"
    ollama_base_model: str = "qwen3:1.7b"

    mehullm_api_token: str = ""
    mehullm_host: str = "127.0.0.1"
    mehullm_port: int = 8000
    mehullm_cors_origins: str = "http://localhost:3000"

    mehullm_db: str = str(ROOT / "data" / "derived" / "mehullm.db")
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
