"""JSON Schema -> Gemini's OpenAPI subset."""

from __future__ import annotations

from typing import Any

__all__ = ["needs_parameters", "sanitize", "sanitize_with_notes"]

# Keys Gemini understands. Anything else is dropped.
_ALLOWED = {"type", "description", "properties", "required", "items", "enum", "nullable"}

# Dropped silently -- they carry no meaning Gemini can use.
_NOISE = {"$schema", "title", "default", "examples", "$comment", "readOnly", "writeOnly"}

_MAX_DEPTH = 6  # also breaks $ref cycles


def sanitize(schema: dict[str, Any]) -> dict[str, Any]:
    return sanitize_with_notes(schema)[0]


def sanitize_with_notes(schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    defs = schema.get("$defs") or schema.get("definitions") or {}
    return _walk(schema, defs, 0, notes, path="$"), notes


def _walk(node: Any, defs: dict, depth: int, notes: list[str], path: str) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {"type": "string"}

    if depth > _MAX_DEPTH:
        notes.append(f"{path}: depth>{_MAX_DEPTH}, collapsed to string")
        return {"type": "string", "description": "nested object encoded as JSON"}

    # --- $ref: inline it. Depth cap handles cycles. ---
    if "$ref" in node:
        key = str(node["$ref"]).rsplit("/", 1)[-1]
        target = defs.get(key)
        if target is None:
            notes.append(f"{path}: unresolved $ref {node['$ref']!r}")
            return {"type": "string"}
        merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
        return _walk(merged, defs, depth + 1, notes, path)

    if "anyOf" in node or "oneOf" in node:
        branches = node.get("anyOf") or node.get("oneOf") or []
        real = [b for b in branches if isinstance(b, dict) and b.get("type") != "null"]
        nullable = len(real) < len(branches)
        out = _walk(real[0] if real else {"type": "string"}, defs, depth + 1, notes, path)
        if nullable:
            out["nullable"] = True
        if len(real) > 1:
            kinds = [b.get("type", "?") for b in real]
            notes.append(f"{path}: union {kinds} widened to {out.get('type')}")
            desc = out.get("description", "")
            out["description"] = f"{desc} (one of: {', '.join(map(str, kinds))})".strip()
        return out

    if "allOf" in node:
        merged: dict[str, Any] = {}
        for b in node["allOf"]:
            merged |= _walk(b, defs, depth + 1, notes, path)
        for k, v in node.items():
            if k != "allOf" and k in _ALLOWED:
                merged[k] = v
        return merged

    out: dict[str, Any] = {}
    for k, v in node.items():
        if k in ("$defs", "definitions"):
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _walk(pv, defs, depth + 1, notes, f"{path}.{pk}") for pk, pv in v.items()}
        elif k == "items":
            out[k] = _walk(v, defs, depth + 1, notes, f"{path}[]")
        elif k in _ALLOWED:
            out[k] = v
        elif k == "format":
            # Not supported, but the hint is useful -- move it into the prose.
            if v in ("date-time", "date", "time", "uri", "email", "uuid"):
                desc = out.get("description") or node.get("description") or ""
                out["description"] = f"{desc} ({v} format)".strip()
        elif k == "additionalProperties":
            pass  # rejected by Gemini; dropping changes nothing semantically
        elif k not in _NOISE:
            notes.append(f"{path}: dropped unsupported key {k!r}")

    if out.get("type") == "object" and not out.get("properties"):
        # A nested empty object is rejected outright. Give it one dummy field.
        # (Top-level empties are handled by needs_parameters() instead.)
        out["properties"] = {"_": {"type": "string", "description": "unused"}}
        notes.append(f"{path}: empty object given a placeholder property")

    return out or {"type": "string"}


def needs_parameters(schema: dict[str, Any] | None) -> bool:
    """False for zero-arg tools."""
    return bool(schema and schema.get("properties"))
