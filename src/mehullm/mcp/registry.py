"""Tool registry: namespacing across servers + MCP -> neutral schema."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from mehullm.llm.types import ToolDef

_SAFE = re.compile(r"[^a-zA-Z0-9_-]")
MAX_NAME = 64
SEP = "__"


@dataclass(frozen=True, slots=True)
class ToolRef:
    server_id: str
    tool_name: str  # the name as the server knows it
    namespaced: str  # the name the model sees
    annotations: dict[str, Any] = field(default_factory=dict)

    @property
    def read_only_hint(self) -> bool:
        """Server-supplied and therefore UNTRUSTED."""
        return bool(self.annotations.get("readOnlyHint"))


def glob_match(pattern: str, name: str) -> bool:
    """`*` only. NOT fnmatch: its ? and [seq] would silently reinterpret configs."""
    if "*" not in pattern:
        return pattern == name
    rx = "^" + ".*".join(re.escape(p) for p in pattern.split("*")) + "$"
    return re.match(rx, name) is not None


def namespace(server_id: str, tool_name: str) -> str:
    base = f"{_SAFE.sub('_', server_id)}{SEP}{_SAFE.sub('_', tool_name)}"
    if len(base) <= MAX_NAME:
        return base
    digest = hashlib.sha1(base.encode()).hexdigest()[:6]
    return f"{base[: MAX_NAME - 7]}_{digest}"


class Registry:
    def __init__(self) -> None:
        self._refs: dict[str, ToolRef] = {}
        self._defs: dict[str, ToolDef] = {}

    def ingest(
        self,
        server_id: str,
        tools: list[Any],
        allow: set[str] | None = None,
    ) -> int:
        """Register a server's tools."""
        added = 0
        for t in sorted(tools, key=lambda x: getattr(x, "name", "")):
            name = getattr(t, "name", None)
            if not name or (allow is not None and name not in allow):
                continue
            ns = namespace(server_id, name)
            desc = (getattr(t, "description", "") or "").strip()
            self._refs[ns] = ToolRef(
                server_id=server_id,
                tool_name=name,
                namespaced=ns,
                annotations=dict(getattr(t, "annotations", None) or {}),
            )
            self._defs[ns] = ToolDef(
                name=ns,
                # Prefix the server identity: without it, `search` from three
                # servers is indistinguishable to the model.
                description=f"[{server_id}] {desc}"[:1000],
                parameters=getattr(t, "inputSchema", None)
                or getattr(t, "input_schema", None)
                or {"type": "object", "properties": {}},
            )
            added += 1
        return added

    def drop_server(self, server_id: str) -> None:
        """Called when a server dies."""
        for ns in [k for k, v in self._refs.items() if v.server_id == server_id]:
            self._refs.pop(ns, None)
            self._defs.pop(ns, None)

    def resolve(self, namespaced: str) -> ToolRef | None:
        return self._refs.get(namespaced)

    def tool_defs(self) -> list[ToolDef]:
        return [self._defs[k] for k in sorted(self._defs)]

    def describe(self) -> list[dict]:
        return [
            {
                "name": ns,
                "server": ref.server_id,
                "tool": ref.tool_name,
                "description": self._defs[ns].description,
                "read_only_hint": ref.read_only_hint,
            }
            for ns, ref in sorted(self._refs.items())
        ]

    def __len__(self) -> int:
        return len(self._refs)

    def estimated_schema_tokens(self) -> int:
        """Rough guard for the TPM ceiling. 30 tools is roughly the budget."""
        import json

        return (
            sum(len(json.dumps(d.parameters)) + len(d.description) for d in self._defs.values())
            // 4
        )
