"""Build the Gemini model ladder."""

from __future__ import annotations

from mehullm.llm.quota import Limits
from mehullm.llm.router import Provider
from mehullm.settings import settings


def build_providers() -> list[Provider]:
    if not settings.gemini_api_key:
        return []
    from mehullm.llm.gemini.client import GeminiClient

    return [
        Provider(
            client=GeminiClient(settings.gemini_api_key, model),
            # Identical per rung: the real ceiling is unpublished and differs by
            # model, so the quota store learns it from each rung's own 429.
            limits=Limits(settings.gemini_rpm_guess, settings.gemini_rpd_guess),
            priority=i,
        )
        for i, model in enumerate(settings.gemini_model_list)
    ]
