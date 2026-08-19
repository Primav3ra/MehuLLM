"use client";

import { useEffect, useRef, useState } from "react";

import type { OrbState } from "./Orb";

/** Per-state look: [core, body, rim] plus pulse rate and drift speed. */
const LOOK: Record<OrbState, { a: number[]; b: number[]; c: number[]; hz: number; speed: number }> = {
  idle:     { a: [1.0, 0.62, 0.45], b: [0.48, 0.36, 0.85], c: [0.55, 0.42, 0.95], hz: 0.16, speed: 0.55 },
  thinking: { a: [0.78, 0.55, 1.00], b: [0.38, 0.30, 0.88], c: [0.60, 0.48, 1.00], hz: 0.42, speed: 0.9 },
  tools:    { a: [1.00, 0.78, 0.42], b: [0.52, 0.34, 0.80], c: [0.95, 0.62, 0.40], hz: 0.62, speed: 1.2 },
  speaking: { a: [1.00, 0.48, 0.34], b: [0.62, 0.30, 0.72], c: [1.00, 0.55, 0.42], hz: 0.95, speed: 1.7 },
  error:    { a: [0.55, 0.50, 0.58], b: [0.28, 0.26, 0.34], c: [0.45, 0.42, 0.50], hz: 0.05, speed: 0.15 },
};

const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;

uniform vec2  uRes;
uniform float uTime;
uniform float uPulse;
uniform float uSpeed;
uniform vec3  uA;
uniform vec3  uB;
uniform vec3  uC;
uniform vec2  uCenter;   // sphere centre, device px (tracks the DOM anchor)
uniform float uRadius;   // sphere radius, device px
out vec4 frag;

float hash(vec3 p) {
  p = fract(p * 0.3183099 + 0.1);
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float vnoise(vec3 x) {
  vec3 i = floor(x), f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
                 mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
                 mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
}

float fbm(vec3 p) {
  float s = 0.0, amp = 0.5;
  for (int i = 0; i < 5; i++) {
    s += amp * vnoise(p);
    p *= 2.02;
    amp *= 0.5;
  }
  return s;
}

mat3 rotY(float a) {
  float c = cos(a), s = sin(a);
  return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);
}

void main() {
  // Background lives in viewport space and stays put while the page scrolls.
  vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;
  float t = uTime;

  // ---- lava lamp: domain-warped fbm, drifting with state ----
  vec2 q = uv * 1.15;
  float bt = t * 0.05 * uSpeed;
  vec2 warp = vec2(fbm(vec3(q * 1.25, bt)), fbm(vec3(q * 1.25 + 4.7, bt + 2.1)));
  float blob = fbm(vec3(q + warp * 1.1, bt * 0.8));
  vec3 deep = vec3(0.086, 0.070, 0.110);
  vec3 mid  = vec3(0.180, 0.140, 0.235);
  vec3 col = mix(deep, mid, smoothstep(0.25, 0.85, blob));
  col = mix(col, uC * 0.30, smoothstep(0.55, 1.0, blob) * 0.55);
  col += uB * 0.06 * smoothstep(0.7, 0.2, length(uv));

  // ---- the sphere: unit space around the DOM anchor, so it scrolls with it ----
  float R = 1.0 + 0.05 * uPulse;
  vec2 sp = (gl_FragCoord.xy - uCenter) / uRadius;
  float d = length(sp);
  if (d < R + 0.01) {
    float z = sqrt(max(R * R - d * d, 0.0));
    vec3 n = normalize(vec3(sp, z));
    vec3 local = rotY(t * 0.22 * uSpeed) * n;          // revolves
    float detail = fbm(local * 2.6 + vec3(0.0, t * 0.04, 0.0));

    vec3 L = normalize(vec3(-0.55, 0.62, 0.75));
    float diff = clamp(dot(n, L), 0.0, 1.0);
    float rim  = pow(1.0 - clamp(dot(n, vec3(0.0, 0.0, 1.0)), 0.0, 1.0), 2.6);
    float spec = pow(clamp(dot(reflect(-L, n), vec3(0.0, 0.0, 1.0)), 0.0, 1.0), 40.0);

    vec3 body = mix(uB, uA, clamp(detail * 0.85 + diff * 0.45, 0.0, 1.0));
    body *= 0.26 + 0.90 * diff;
    body += uA * spec * 0.85;
    body += uC * rim * (0.45 + 0.25 * uPulse);

    float edge = smoothstep(R, R - 0.02, d);
    col = mix(col, body, edge);
  }

  // ---- outer bloom ----
  float glow = exp(-max(d - R, 0.0) * 2.6);
  col += uA * glow * (0.14 + 0.34 * uPulse);

  col = pow(max(col, 0.0), vec3(0.92));
  frag = vec4(col, 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader | null {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.error("shader:", gl.getShaderInfoLog(sh));
    return null;
  }
  return sh;
}

export function OrbGL({ state }: { state: OrbState }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef<OrbState>(state);
  const [failed, setFailed] = useState(false);

  stateRef.current = state;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
    if (!gl) {
      setFailed(true);
      return;
    }

    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) {
      setFailed(true);
      return;
    }
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error("link:", gl.getProgramInfoLog(prog));
      setFailed(true);
      return;
    }
    gl.useProgram(prog);

    const u = {
      res: gl.getUniformLocation(prog, "uRes"),
      time: gl.getUniformLocation(prog, "uTime"),
      pulse: gl.getUniformLocation(prog, "uPulse"),
      speed: gl.getUniformLocation(prog, "uSpeed"),
      a: gl.getUniformLocation(prog, "uA"),
      b: gl.getUniformLocation(prog, "uB"),
      c: gl.getUniformLocation(prog, "uC"),
      center: gl.getUniformLocation(prog, "uCenter"),
      radius: gl.getUniformLocation(prog, "uRadius"),
    };

    // Cap DPR: a fullscreen 5-octave fbm at DPR 2 is wasteful on an iGPU.
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    function resize() {
      const w = Math.floor(canvas!.clientWidth * dpr);
      const h = Math.floor(canvas!.clientHeight * dpr);
      if (canvas!.width !== w || canvas!.height !== h) {
        canvas!.width = w;
        canvas!.height = h;
        gl!.viewport(0, 0, w, h);
      }
    }
    resize();
    window.addEventListener("resize", resize);

    // Smoothed so a state change eases instead of snapping.
    const cur = { a: [...LOOK.idle.a], b: [...LOOK.idle.b], c: [...LOOK.idle.c], hz: LOOK.idle.hz, speed: LOOK.idle.speed };
    let raf = 0;
    const t0 = performance.now();
    let phase = 0;
    let last = t0;

    function frame(now: number) {
      resize();
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      const want = LOOK[stateRef.current];
      const k = 1 - Math.exp(-dt * 3.2);
      for (let i = 0; i < 3; i++) {
        cur.a[i] += (want.a[i] - cur.a[i]) * k;
        cur.b[i] += (want.b[i] - cur.b[i]) * k;
        cur.c[i] += (want.c[i] - cur.c[i]) * k;
      }
      cur.hz += (want.hz - cur.hz) * k;
      cur.speed += (want.speed - cur.speed) * k;
      phase += dt * cur.hz * Math.PI * 2;

      gl!.uniform2f(u.res, canvas!.width, canvas!.height);
      gl!.uniform1f(u.time, (now - t0) / 1000);
      gl!.uniform1f(u.pulse, 0.5 + 0.5 * Math.sin(phase));
      gl!.uniform1f(u.speed, cur.speed);
      gl!.uniform3f(u.a, cur.a[0], cur.a[1], cur.a[2]);
      gl!.uniform3f(u.b, cur.b[0], cur.b[1], cur.b[2]);
      gl!.uniform3f(u.c, cur.c[0], cur.c[1], cur.c[2]);

      // Follow the layout anchor: correct size, and it scrolls with the page.
      const anchor = document.querySelector<HTMLElement>(".orb-wrap");
      const r = anchor?.getBoundingClientRect();
      const cx = (r ? r.left + r.width / 2 : canvas!.clientWidth / 2) * dpr;
      const cyTop = (r ? r.top + r.height / 2 : canvas!.clientHeight * 0.34) * dpr;
      const rad = (r ? Math.min(r.width, r.height) * 0.42 : 110) * dpr;
      gl!.uniform2f(u.center, cx, canvas!.height - cyTop);   // GL y is bottom-up
      gl!.uniform1f(u.radius, Math.max(rad, 1));
      gl!.drawArrays(gl!.TRIANGLES, 0, 3);
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    };
  }, []);

  if (failed) return null;                     // page falls back to the CSS orb
  return <canvas ref={ref} className="orb-gl" aria-hidden="true" />;
}
