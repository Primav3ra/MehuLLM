"""Talk to a running MehuLLM backend from the terminal.

    uv run --no-sync python scripts/chat.py "what sports do I play?"
    uv run --no-sync python scripts/chat.py            # interactive
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from mehullm.obs import utf8_stdout
from mehullm.settings import settings

API = f"http://{settings.mehullm_host}:{settings.mehullm_port}"
DIM, TOOL, BOT, WARN, OFF = "\033[2m", "\033[36m", "\033[38;5;209m", "\033[33m", "\033[0m"


def _post(path: str, body: dict):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{API}{path}", data=data,
                                headers={"Content-Type": "application/json"})
    if settings.mehullm_api_token:
        req.add_header("Authorization", f"Bearer {settings.mehullm_api_token}")
    return urllib.request.urlopen(req, timeout=300)


def ask(message: str, conversation: str = "cli") -> None:
    run_id = ""
    try:
        stream = _post("/api/chat", {"message": message, "conversation_id": conversation})
    except urllib.error.URLError as e:
        print(f"{WARN}backend not reachable at {API} ({e.reason}){OFF}")
        print(f"{DIM}start it with: uv run --no-sync uvicorn mehullm.api.app:app "
              f"--host {settings.mehullm_host} --port {settings.mehullm_port}{OFF}")
        return
    for raw in stream:
        line = raw.decode("utf-8", "replace")
        if not line.startswith("data: "):
            continue
        e = json.loads(line[6:])
        run_id = e.get("run_id", run_id)
        kind = e["type"]

        if kind == "run_start":
            print(f"{DIM}{e['model']} · {e['tool_count']} tools{OFF}")
        elif kind == "text_delta":
            print(f"{DIM}{e['text']}{OFF}", end="", flush=True)
        elif kind == "tool_start":
            print(f"\n{TOOL}[{e['tool']} · {e['risk']}]{OFF}", flush=True)
        elif kind == "guardrail_blocked":
            print(f"\n{WARN}[blocked: {e['message']}]{OFF}", flush=True)
        elif kind == "confirmation_request":
            print(f"\n{WARN}CONFIRM {e['tool']}{OFF}\n  {e['summary']}")
            print(f"  args: {json.dumps(e['arguments'])[:300]}")
            yes = input("  approve? [y/N] ").strip().lower().startswith("y")
            _post(f"/api/chat/{run_id}/confirm",
                  {"interaction_id": e["interaction_id"],
                   "decision": "approve" if yes else "deny"}).read()
        elif kind == "voice_delta":
            print(f"\n\n{BOT}{e['text']}{OFF}")
        elif kind == "error":
            print(f"\n{WARN}error: {e['message'][:200]}{OFF}")
        elif kind == "done":
            print(f"{DIM}[{e['steps']} steps · {e['tool_calls']} tools · "
                  f"{e['total_ms']}ms · {e['trace_id']}]{OFF}\n")


def main() -> int:
    utf8_stdout()
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
        return 0
    print(f"{DIM}MehuLLM CLI on {API} — blank line or ctrl-c to quit{OFF}")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not q:
            return 0
        ask(q)


if __name__ == "__main__":
    raise SystemExit(main())
