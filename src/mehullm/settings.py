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
    # NOT gemini-3.5-flash: its free tier is 20 requests PER DAY, and one agent.
    gemini_model: str = "gemini-3.5-flash-lite"
    # A LADDER, because free-tier quota is per-model.
    gemini_models: str = (
        "gemini-3.5-flash-lite,gemini-3.6-flash,gemini-3.7-flash,"
        "gemini-3.1-flash-lite,gemini-3.1-flash-lite-preview,"
        "gemini-3-flash-preview,gemini-3.5-flash"
    )
    gemini_rpm_guess: int = 5  # measured from the 429 quotaValue, not guessed
    # Deliberately conservative. Google no longer publishes free-tier limits, so
    # the quota store LEARNS the real ceiling from 429s rather than trusting this.
    gemini_rpd_guess: int = 200

    ollama_host: str = "http://localhost:11434"
    ollama_voice_model: str = "mehul-voice"
    ollama_base_model: str = "qwen3:1.7b"

    log_level: str = "INFO"
    log_file: str = str(ROOT / "data" / "derived" / "mehullm.log")

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

    @staticmethod
    def _ladder(raw: str, primary: str) -> list[str]:
        """Primary first, then the rest, de-duplicated, order preserved."""
        out: list[str] = []
        for m in [primary, *raw.split(",")]:
            m = m.strip()
            if m and m not in out:
                out.append(m)
        return out

    @property
    def gemini_model_list(self) -> list[str]:
        return self._ladder(self.gemini_models, self.gemini_model)


settings = Settings()
