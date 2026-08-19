import type { AgentEvent } from "./events";

const API = process.env.NEXT_PUBLIC_API ?? "http://127.0.0.1:8000";
const TOKEN = process.env.NEXT_PUBLIC_TOKEN ?? "";

function headers(): HeadersInit {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (TOKEN) h.Authorization = `Bearer ${TOKEN}`;
  return h;
}

/** Parse SSE frames out of a byte stream. */
async function* frames(res: Response): AsyncGenerator<AgentEvent> {
  const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let cut: number;
    while ((cut = buf.indexOf("\n\n")) !== -1) {
      const raw = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      const line = raw.split("\n").find((l) => l.startsWith("data: "));
      if (line) {
        try {
          yield JSON.parse(line.slice(6)) as AgentEvent;
        } catch {
          // keep-alive pings and comments are not JSON
        }
      }
    }
  }
}

/**
 * Send a message and yield events. Reconnects from the last seq on a dropped
 * stream, so a pending confirmation survives a flaky connection.
 */
export async function* chat(
  message: string,
  conversationId = "default",
): AsyncGenerator<AgentEvent> {
  const res = await fetch(`${API}/api/chat`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });
  if (!res.ok || !res.body) throw new Error(`chat failed: ${res.status}`);

  let lastSeq = 0;
  let runId = "";
  try {
    for await (const ev of frames(res)) {
      lastSeq = ev.seq;
      runId = ev.run_id;
      yield ev;
      if (ev.type === "done") return;
    }
  } catch {
    // fall through to replay
  }

  if (!runId) return;
  const again = await fetch(
    `${API}/api/chat/${runId}/events?after_seq=${lastSeq}`,
    { headers: headers() },
  );
  if (!again.ok || !again.body) return;
  for await (const ev of frames(again)) {
    yield ev;
    if (ev.type === "done") return;
  }
}

export async function confirm(
  runId: string,
  interactionId: string,
  decision: "approve" | "deny",
  reason = "",
): Promise<void> {
  await fetch(`${API}/api/chat/${runId}/confirm`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ interaction_id: interactionId, decision, reason }),
  });
}

export async function health(): Promise<Record<string, unknown>> {
  const r = await fetch(`${API}/api/health`);
  return r.json();
}
