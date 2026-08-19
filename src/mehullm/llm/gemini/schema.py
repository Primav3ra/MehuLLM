"""JSON Schema -> Gemini's OpenAPI subset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["needs_parameters", "sanitize", "sanitize_with_notes"]

_ALLOWED = {"type", "description", "properties", "required", "items", "enum", "nullable"}
_NOISE = {"$schema", "title", "default", "examples", "$comment", "readOnly", "writeOnly"}
_FORMAT_HINTS = {"date-time", "date", "time", "uri", "email", "uuid"}

_MAX_DEPTH = 6  # also breaks $ref cycles


@dataclass
class _Ctx:
    defs: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def note(self, path: str, msg: str) -> None:
        self.notes.append(f"{path}: {msg}")


def sanitize(schema: dict[str, Any]) -> dict[str, Any]:
    return sanitize_with_notes(schema)[0]


def sanitize_with_notes(schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    ctx = _Ctx(defs=schema.get("$defs") or schema.get("definitions") or {})
    return _walk(schema, ctx, 0, "$"), ctx.notes


def needs_parameters(schema: dict[str, Any] | None) -> bool:
    """False for zero-arg tools."""
    return bool(schema and schema.get("properties"))


def _walk(node: Any, ctx: _Ctx, depth: int, path: str) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {"type": "string"}
    if depth > _MAX_DEPTH:
        ctx.note(path, f"depth>{_MAX_DEPTH}, collapsed to string")
        return {"type": "string", "description": "nested object encoded as JSON"}
    if "$ref" in node:
        return _inline_ref(node, ctx, depth, path)
    if "anyOf" in node or "oneOf" in node:
        return _widen_union(node, ctx, depth, path)
    if "allOf" in node:
        return _merge_all_of(node, ctx, depth, path)
    return _rewrite_keys(node, ctx, depth, path)


def _inline_ref(node: dict, ctx: _Ctx, depth: int, path: str) -> dict[str, Any]:
    key = str(node["$ref"]).rsplit("/", 1)[-1]
    target = ctx.defs.get(key)
    if target is None:
        ctx.note(path, f"unresolved $ref {node['$ref']!r}")
        return {"type": "string"}
    merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
    return _walk(merged, ctx, depth + 1, path)


def _widen_union(node: dict, ctx: _Ctx, depth: int, path: str) -> dict[str, Any]:
    """Gemini has no union type, so keep the first real branch and describe the rest."""
    branches = node.get("anyOf") or node.get("oneOf") or []
    real = [b for b in branches if isinstance(b, dict) and b.get("type") != "null"]
    out = _walk(real[0] if real else {"type": "string"}, ctx, depth + 1, path)
    if len(real) < len(branches):
        out["nullable"] = True
    if len(real) > 1:
        kinds = [b.get("type", "?") for b in real]
        ctx.note(path, f"union {kinds} widened to {out.get('type')}")
        desc = out.get("description", "")
        out["description"] = f"{desc} (one of: {', '.join(map(str, kinds))})".strip()
    return out


def _merge_all_of(node: dict, ctx: _Ctx, depth: int, path: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for branch in node["allOf"]:
        merged |= _walk(branch, ctx, depth + 1, path)
    merged.update({k: v for k, v in node.items() if k != "allOf" and k in _ALLOWED})
    return merged


def _rewrite_keys(node: dict, ctx: _Ctx, depth: int, path: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k in ("$defs", "definitions") or k == "additionalProperties" or k in _NOISE:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _walk(pv, ctx, depth + 1, f"{path}.{pk}") for pk, pv in v.items()}
        elif k == "items":
            out[k] = _walk(v, ctx, depth + 1, f"{path}[]")
        elif k == "type" and isinstance(v, list):
            out.update(_scalar_type(v, ctx, path))
        elif k == "enum":
            # Gemini rejects non-string enum members and applies enum only to strings.
            out[k] = [str(x) for x in v] if isinstance(v, list) else v
            out.setdefault("type", "string")
        elif k in _ALLOWED:
            out[k] = v
        elif k == "format" and v in _FORMAT_HINTS:
            desc = out.get("description") or node.get("description") or ""
            out["description"] = f"{desc} ({v} format)".strip()
        elif k != "format":
            ctx.note(path, f"dropped unsupported key {k!r}")

    if out.get("type") == "object" and not out.get("properties"):
        # Gemini rejects a nested empty object; top-level ones go to needs_parameters().
        out["properties"] = {"_": {"type": "string", "description": "unused"}}
        ctx.note(path, "empty object given a placeholder property")
    return out or {"type": "string"}


def _scalar_type(kinds: list, ctx: _Ctx, path: str) -> dict[str, Any]:
    """JSON Schema allows a list of types; Gemini wants one scalar."""
    real = [x for x in kinds if x != "null"]
    out: dict[str, Any] = {"type": real[0] if real else "string"}
    if len(kinds) > len(real):
        out["nullable"] = True
    if len(real) > 1:
        ctx.note(path, f"type {kinds} narrowed to {out['type']}")
    return out
