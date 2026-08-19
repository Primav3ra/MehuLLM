"use client";

import { useRef, useState } from "react";

import { Orb, type OrbState } from "@/components/Orb";
import { OrbGL } from "@/components/OrbGL";
import type { AgentEvent } from "@/lib/events";
import { chat, confirm } from "@/lib/stream";

type Turn = {
  you: string;
  narration: string;
  answer: string;
  tools: { name: string; risk: string; ok?: boolean }[];
  blocked: string[];
};

type Card = {
  runId: string;
  interactionId: string;
  tool: string;
  summary: string;
  args: unknown;
  rule: string;
};

const blank = (you: string): Turn => ({ you, narration: "", answer: "", tools: [], blocked: [] });

export default function Page() {
  const [turns, setTurns] = useState<Turn[]>([]);
  // ?state=speaking|thinking|tools|error previews the orb without a backend.
  const preview =
    typeof window !== "undefined"
      ? (new URLSearchParams(window.location.search).get("state") as OrbState | null)
      : null;
  const [state, setState] = useState<OrbState>(preview ?? "idle");
  const [cards, setCards] = useState<Card[]>([]);
  const [draft, setDraft] = useState("");
  const [meta, setMeta] = useState("");
  const busy = state !== "idle" && state !== "error";
  const endRef = useRef<HTMLDivElement>(null);

  function patch(fn: (t: Turn) => void) {
    setTurns((prev) => {
      const next = [...prev];
      const last = { ...next[next.length - 1] };
      fn(last);
      next[next.length - 1] = last;
      return next;
    });
  }

  function handle(ev: AgentEvent) {
    switch (ev.type) {
      case "run_start":
        setMeta(`${ev.model as string} · ${ev.tool_count as number} tools`);
        setState("thinking");
        break;
      case "status":
        setState(ev.stage === "calling_tools" ? "tools" : "thinking");
        break;
      case "text_delta":
        patch((t) => { t.narration += ev.text as string; });
        break;
      case "tool_start":
        patch((t) => {
          t.tools = [...t.tools, { name: ev.tool as string, risk: String(ev.risk) }];
        });
        break;
      case "tool_result":
        patch((t) => {
          t.tools = t.tools.map((x) =>
            x.name === ev.tool && x.ok === undefined ? { ...x, ok: Boolean(ev.ok) } : x,
          );
        });
        break;
      case "guardrail_blocked":
        patch((t) => { t.blocked = [...t.blocked, String(ev.message)]; });
        break;
      case "confirmation_request":
        setCards((c) => [
          ...c,
          {
            runId: ev.run_id,
            interactionId: String(ev.interaction_id),
            tool: String(ev.tool),
            summary: String(ev.summary),
            args: ev.arguments,
            rule: String(ev.rule),
          },
        ]);
        break;
      case "confirmation_resolved":
        setCards((c) => c.filter((x) => x.interactionId !== ev.interaction_id));
        break;
      case "voice_start":
        setState("speaking");
        break;
      case "voice_delta":
        setState("speaking");
        patch((t) => { t.answer += ev.text as string; });
        break;
      case "error":
        setState("error");
        patch((t) => { t.answer = t.answer || `Error: ${String(ev.message)}`; });
        break;
      case "done":
        setState("idle");
        setMeta(`${ev.steps as number} steps · ${ev.tool_calls as number} tool calls · ${ev.total_ms as number} ms`);
        break;
    }
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }

  async function send() {
    const text = draft.trim();
    if (!text || busy) return;
    setDraft("");
    setTurns((p) => [...p, blank(text)]);
    setState("thinking");
    try {
      for await (const ev of chat(text)) handle(ev);
    } catch (e) {
      setState("error");
      patch((t) => { t.answer = `Could not reach the API: ${String(e)}`; });
    }
  }

  async function decide(card: Card, decision: "approve" | "deny") {
    setCards((c) => c.filter((x) => x.interactionId !== card.interactionId));
    await confirm(card.runId, card.interactionId, decision);
  }

  return (
    <>
      <OrbGL state={state} />
      <main className="shell">
        <h1 className="wordmark">
          Mehu<span>LLM</span>
        </h1>
        <Orb state={state} />

      {cards.map((c) => (
        <div className="confirm" key={c.interactionId}>
          <h4>Approve {c.tool}?</h4>
          <div className="narration">{c.summary}</div>
          <pre>{JSON.stringify(c.args, null, 2)}</pre>
          <div className="narration">Triggered by: {c.rule}</div>
          <div className="row">
            <button className="deny" onClick={() => decide(c, "deny")}>Deny</button>
            <button className="ok" onClick={() => decide(c, "approve")}>Approve</button>
          </div>
        </div>
      ))}

      {turns.length > 0 && (
        <div className="card">
          {turns.map((t, i) => (
            <div className="turn" key={i}>
              <div className="who">You</div>
              <div className="body">{t.you}</div>
              {t.narration && (
                <>
                  <div className="who" style={{ marginTop: 12 }}>Working</div>
                  <div className="body narration">{t.narration}</div>
                </>
              )}
              {t.tools.length > 0 && (
                <div className="tools">
                  {t.tools.map((x, j) => (
                    <span className="chip" key={j} data-risk={x.risk} data-ok={x.ok !== false}>
                      {x.name}
                    </span>
                  ))}
                </div>
              )}
              {t.blocked.map((b, j) => (
                <div className="narration" key={j} style={{ marginTop: 8 }}>Blocked: {b}</div>
              ))}
              {t.answer && (
                <>
                  <div className="who bot" style={{ marginTop: 14 }}>MehuLLM</div>
                  <div className="body">{t.answer}</div>
                </>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="composer">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask me anything..."
          disabled={busy}
        />
        <button onClick={send} disabled={busy || !draft.trim()}>Send</button>
      </div>

      {meta && <div className="meta">{meta}</div>}
        <div ref={endRef} />
      </main>
    </>
  );
}
