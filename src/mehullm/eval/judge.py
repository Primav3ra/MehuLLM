"""LLM judge — the ~30% that cannot be checked mechanically.

Scope is deliberately narrow: correctness, faithfulness, helpfulness. Everything
else is a deterministic assertion in ``graders.py``.

Two rules that make the number defensible:

1. **Style is never judged.** The judge prompt ends with an explicit instruction
   not to reward or penalise tone, slang, length or language mixing. A judge
   that likes polished English would silently punish the exact Hinglish voice
   this project exists to produce -- and it would do so invisibly, as a
   correctness score. Style gets the objective metric in ``style_score.py``.

2. **Each scenario is judged TWICE with the criteria order swapped**, and
   disagreement flags for manual review instead of being averaged. Averaging two
   contradictory verdicts manufactures a confident number out of a coin flip;
   surfacing the disagreement tells you the scenario is ambiguous, which is
   usually true and worth knowing.
"""

from __future__ import annotations

import json
from typing import Any

from mehullm.eval.bank import Scenario
from mehullm.eval.graders import Transcript
from mehullm.llm.types import Msg

_CRITERIA = [
    ("correct", "Does the answer satisfy the rubric?"),
    ("faithful", "Is every factual claim supported by <memory> or a tool result "
                 "shown in the transcript? Unsupported specifics are unfaithful."),
    ("helpful", "Does it actually address what the user asked?"),
]

_SYS = """You are grading one response from a personal AI assistant.

Return JSON only:
{{"correct": bool, "faithful": bool, "helpful": bool, "reason": "<one sentence>"}}

Criteria, in order:
{criteria}

Judge ONLY the criteria above.

Do not reward or penalize writing style, tone, length, slang, capitalization,
emoji, or language mixing. The assistant is SUPPOSED to write casually and to
mix Hindi and English. Casual or Hinglish phrasing is never a defect.

Saying "I don't know" is CORRECT when the rubric asks for it. An answer that
invents a specific detail is never correct, however fluent."""

_USER = """RUBRIC:
{rubric}

USER ASKED:
{prompt}

MEMORY AVAILABLE:
{memory}

TOOLS CALLED: {tools}

ASSISTANT ANSWERED:
{answer}"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "faithful": {"type": "boolean"},
        "helpful": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "faithful", "helpful", "reason"],
}


def _render_criteria(order: list[tuple[str, str]]) -> str:
    return "\n".join(f"{i}. {k}: {v}" for i, (k, v) in enumerate(order, 1))


class Judge:
    """Judged twice, order swapped. `client` is any LLMClient."""

    def __init__(self, client: Any, *, temperature: float = 0.0):
        self.client = client
        self.temperature = temperature

    async def _once(self, s: Scenario, t: Transcript,
                    order: list[tuple[str, str]]) -> dict[str, Any]:
        memory = "\n".join(f["text"] for f in s.seed_facts) or "(nothing)"
        user = _USER.format(
            rubric=s.rubric.strip() or "(none given)",
            prompt=s.prompt,
            memory=memory,
            tools=", ".join(t.tools_called) or "none",
            answer=t.final_text[:4000] or "(empty)",
        )
        chunks: list[str] = []
        async for ev in self.client.stream(
            system=_SYS.format(criteria=_render_criteria(order)),
            messages=[Msg(role="user", text=user)],
            tools=[],
            max_tokens=300,
        ):
            text = getattr(ev, "text", None)
            if text:
                chunks.append(text)
        return _parse("".join(chunks))

    async def judge(self, s: Scenario, t: Transcript) -> dict[str, Any]:
        a = await self._once(s, t, _CRITERIA)
        b = await self._once(s, t, list(reversed(_CRITERIA)))

        keys = ("correct", "faithful", "helpful")
        disagree = [k for k in keys if a.get(k) != b.get(k)]
        out: dict[str, Any] = {
            "reason": a.get("reason", "") or b.get("reason", ""),
            "disagreement": disagree,
            "needs_review": bool(disagree),
        }
        # On disagreement take the STRICTER verdict and flag it. Optimistic
        # resolution would let every ambiguous scenario drift green over time.
        for k in keys:
            out[k] = bool(a.get(k)) and bool(b.get(k))
        return out


def _parse(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
    # An unparseable judge must not silently pass the scenario.
    return {"correct": False, "faithful": False, "helpful": False,
            "reason": f"judge returned unparseable output: {raw[:120]}"}
