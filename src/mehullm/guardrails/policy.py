"""Risk tiers and the allow / deny / confirm decision.

FAIL CLOSED. `default_action: confirm` means a newly added server's tools
*prompt* rather than fire. A denylist would silently auto-allow them, which is
the wrong default for something that can send email.

Server-supplied `annotations` (readOnlyHint, destructiveHint) are UNTRUSTED.
They only auto-classify tools the policy file does not mention, only in the
permissive direction, and never override an explicit tier. A compromised server
claiming readOnlyHint on its delete_everything tool must not be able to escalate.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Action = Literal["allow", "deny", "confirm"]
Tier = Literal["T0", "T1", "T2", "T3"]

TIER_RISK = {"T0": "read", "T1": "write", "T2": "irreversible", "T3": "denied"}


@dataclass
class Rule:
    id: str
    action: Action
    tools: list[str] = field(default_factory=list)
    tiers: list[str] = field(default_factory=list)
    provenance_contains: str | None = None
    reason: str = ""

    def matches(self, tool: str, tier: str, provenance: set[str]) -> bool:
        if self.tools and not any(fnmatch.fnmatch(tool, p) for p in self.tools):
            return False
        if self.tiers and tier not in self.tiers:
            return False
        if self.provenance_contains and self.provenance_contains not in provenance:
            return False
        return bool(self.tools or self.tiers or self.provenance_contains)


@dataclass
class Verdict:
    action: Action
    tier: Tier
    rule_id: str
    reason: str = ""

    @property
    def risk(self) -> str:
        return TIER_RISK.get(self.tier, "write")


@dataclass
class Limits:
    max_tool_calls_per_turn: int = 25
    max_turn_seconds: int = 180
    max_daily_requests: int = 200
    max_steps: int = 12


@dataclass
class Policy:
    default_action: Action = "confirm"
    tiers: dict[str, str] = field(default_factory=dict)  # glob -> tier
    rules: list[Rule] = field(default_factory=list)
    limits: Limits = field(default_factory=Limits)

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            default_action=raw.get("default_action", "confirm"),
            tiers={k: v for k, v in (raw.get("tiers") or {}).items()},
            limits=Limits(**(raw.get("limits") or {})),
            rules=[
                Rule(
                    id=r.get("id", f"rule-{i}"),
                    action=r["action"],
                    tools=list(r.get("match", {}).get("tool", []) or []),
                    tiers=list(r.get("match", {}).get("tier", []) or []),
                    provenance_contains=r.get("match", {}).get("provenance_contains"),
                    reason=r.get("reason", ""),
                )
                for i, r in enumerate(raw.get("rules") or [])
            ],
        )

    def tier_for(self, tool: str, read_only_hint: bool = False) -> Tier:
        for pattern, tier in self.tiers.items():
            if fnmatch.fnmatch(tool, pattern):
                return tier  # type: ignore[return-value]
        # Unclassified. A server's readOnlyHint may relax it to T0, but only
        # here -- never over an explicit mapping above.
        return "T0" if read_only_hint else "T2"

    def evaluate(
        self, tool: str, *, read_only_hint: bool = False, provenance: set[str] | None = None
    ) -> Verdict:
        tier = self.tier_for(tool, read_only_hint)
        provenance = provenance or set()

        if tier == "T3":
            return Verdict("deny", tier, "tier-T3", "Tier T3 tools are disabled in this build.")

        for rule in self.rules:
            if rule.matches(tool, tier, provenance):
                return Verdict(rule.action, tier, rule.id, rule.reason)

        if tier == "T0":
            return Verdict("allow", tier, "tier-default")
        if tier == "T1":
            return Verdict("allow", tier, "tier-default")
        return Verdict(self.default_action, tier, "default")
