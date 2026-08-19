export type OrbState = "idle" | "thinking" | "tools" | "speaking" | "error";

const LABEL: Record<OrbState, string> = {
  idle: "Ready",
  thinking: "MehuLLM is thinking",
  tools: "Using tools",
  speaking: "MehuLLM is speaking",
  error: "Something went wrong",
};

export function Orb({ state }: { state: OrbState }) {
  return (
    <>
      <div className="orb-wrap">
        <div className="ring a" />
        <div className="ring b" />
        <div className="orb" data-state={state} />
      </div>
      <span className="pill">
        <span className="dot" data-quiet={state === "idle"} />
        {LABEL[state]}
      </span>
    </>
  );
}
