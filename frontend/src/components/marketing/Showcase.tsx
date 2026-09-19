"use client";

import {
  Database, Cpu, LineChart, FlaskConical, Rocket, Radar,
  Check, X, Cable, LayoutDashboard, Boxes,
  CircuitBoard, Wifi, Cherry, Package, Binary, Terminal,
  Lock, Activity, ArrowUp, LayoutGrid,
} from "lucide-react";
import { Reveal, Counter } from "./primitives";

/** Inline CSS custom property for the per-icon accent. */
const accent = (c: string) => ({ ["--c"]: c }) as React.CSSProperties;

/* ============================ DASHBOARD SHOWCASE ============================ */
const DASH_POINTS = [
  { icon: Boxes, c: "#6366f1", title: "Dataset Explorer", desc: "Browse, filter and inspect every sample." },
  { icon: Cpu, c: "#a855f7", title: "Training Jobs", desc: "Queue GPU jobs and watch them converge live." },
  { icon: LineChart, c: "#06b6d4", title: "Model Metrics", desc: "Accuracy, loss, and per-class breakdowns." },
  { icon: FlaskConical, c: "#0ea5e9", title: "Model Testing", desc: "Validate against held-out test data." },
  { icon: Rocket, c: "#22c55e", title: "Deployments", desc: "Track every build across every target." },
  { icon: Radar, c: "#14b8a6", title: "Device Fleet", desc: "Monitor connected devices in real time." },
];

/* Coded product-UI mock of the PetalEdge dashboard. Pure CSS/SVG — no images.
   Palette + card language mirror the shipped app tokens (--app-surface/border/
   text in globals.css) so it reads like a real screenshot, not a placeholder. */
const DM_BARS = [40, 62, 48, 80, 66, 90, 72, 84];
const DM_MATRIX = [
  [94, 3, 2, 1],
  [2, 91, 4, 3],
  [1, 5, 96, 2],
  [3, 2, 1, 93],
];
const DM_LABELS = ["A", "B", "C", "D"];
const DM_DATASETS = [
  { n: "gestures_v4", samples: "1,240", s: 82, t: "trained", c: "#6366f1" },
  { n: "keywords_v2", samples: "860", s: 61, t: "labeling", c: "#8b5cf6" },
  { n: "defects_v7", samples: "2,015", s: 95, t: "trained", c: "#14b8a6" },
];

function DashboardMock() {
  return (
    <div className="pe-dash-mock" role="img" aria-label="PetalEdge product dashboard preview">
      {/* macOS browser chrome */}
      <div className="pe-dash-bar">
        <div className="pe-dash-dots">
          <i style={{ background: "#ff5f57" }} />
          <i style={{ background: "#febc2e" }} />
          <i style={{ background: "#28c840" }} />
        </div>
        <span className="pe-dash-url"><Lock /> petaledge.ai/dashboard</span>
      </div>

      <div className="pe-dash-body">
        {/* Accuracy / metric card with sparkline */}
        <div className="pe-dm-panel wide">
          <div className="pe-dm-head">
            <div>
              <span className="pe-dm-eyebrow"><Activity /> Model Accuracy</span>
              <div className="pe-dm-value">
                98.4<span className="pe-dm-unit">%</span>
                <span className="pe-dm-delta up"><ArrowUp /> 2.1%</span>
              </div>
            </div>
            <span className="pe-dm-pill trained">trained</span>
          </div>
          <div className="pe-spark-wrap">
            <svg className="pe-spark" viewBox="0 0 300 60" preserveAspectRatio="none">
              <defs>
                <linearGradient id="peSparkFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="rgba(99,102,241,0.34)" />
                  <stop offset="100%" stopColor="rgba(99,102,241,0)" />
                </linearGradient>
                <linearGradient id="peSparkLine" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#6366f1" />
                  <stop offset="55%" stopColor="#8b5cf6" />
                  <stop offset="100%" stopColor="#06b6d4" />
                </linearGradient>
              </defs>
              <line className="grid" x1="0" y1="20" x2="300" y2="20" />
              <line className="grid" x1="0" y1="40" x2="300" y2="40" />
              <path className="area" d="M0,48 C30,44 45,31 70,29 C100,26 120,33 150,22 C180,12 210,16 240,9 C265,5 285,7 300,4 L300,60 L0,60 Z" />
              <path className="line" d="M0,48 C30,44 45,31 70,29 C100,26 120,33 150,22 C180,12 210,16 240,9 C265,5 285,7 300,4" />
            </svg>
            <span className="pe-spark-end" />
          </div>
        </div>

        {/* GPU usage bar chart */}
        <div className="pe-dm-panel">
          <div className="pe-dm-head compact">
            <span className="pe-dm-eyebrow"><Cpu /> GPU Usage</span>
            <span className="pe-dm-num">76%</span>
          </div>
          <div className="pe-dm-bars">
            {DM_BARS.map((h, i) => (
              <i key={i} className={i === DM_BARS.length - 1 ? "hot" : ""}>
                <span style={{ height: `${h}%` }} />
              </i>
            ))}
          </div>
          <div className="pe-dm-foot"><span>min 40</span><span>avg 68</span><span>max 90</span></div>
        </div>

        {/* Confusion matrix */}
        <div className="pe-dm-panel">
          <div className="pe-dm-head compact">
            <span className="pe-dm-eyebrow"><LayoutGrid /> Confusion Matrix</span>
          </div>
          <div className="pe-cm">
            <span className="pe-cm-cell head" />
            {DM_LABELS.map((l) => (
              <span key={`col-${l}`} className="pe-cm-cell head">{l}</span>
            ))}
            {DM_MATRIX.map((row, r) => [
              <span key={`row-${r}`} className="pe-cm-cell head">{DM_LABELS[r]}</span>,
              ...row.map((v, c) => {
                const on = r === c;
                const t = v / 100;
                return (
                  <i
                    key={`${r}-${c}`}
                    className={on ? "on" : "off"}
                    style={{
                      background: on
                        ? `linear-gradient(135deg, rgba(99,102,241,${0.5 + t * 0.42}), rgba(139,92,246,${0.5 + t * 0.42}))`
                        : `rgba(139,92,246,${0.05 + t * 0.5})`,
                    }}
                  >
                    {v}
                  </i>
                );
              }),
            ])}
          </div>
          <div className="pe-cm-legend"><span>Low</span><i /><span>High</span></div>
        </div>

        {/* Datasets list with status pills + progress bars */}
        <div className="pe-dm-panel wide">
          <div className="pe-dm-head compact">
            <span className="pe-dm-eyebrow"><Database /> Datasets</span>
            <span className="pe-dm-num muted">3 active</span>
          </div>
          <div className="pe-dm-table">
            <div className="pe-dm-row head">
              <span>Dataset</span><span>Samples</span><span>Progress</span><span>Status</span>
            </div>
            {DM_DATASETS.map((row) => (
              <div key={row.n} className="pe-dm-row">
                <span className="cell name"><i className="dot" style={{ background: row.c, color: row.c }} />{row.n}</span>
                <span className="cell num">{row.samples}</span>
                <span className="pe-dm-bar-mini"><i className={row.t === "labeling" ? "b" : ""} style={{ width: `${row.s}%` }} /></span>
                <span className={`pe-dm-pill ${row.t}`}>{row.t}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export function DashboardShowcase() {
  return (
    <section className="pe-section" id="dashboard">
      <div className="pe-container">
        <div className="pe-dash-grid">
          <div>
            <Reveal as="span" className="pe-eyebrow"><LayoutDashboard /> Dashboard</Reveal>
            <Reveal as="h2" className="pe-h2" delay={60}>
              Manage Everything in <span className="pe-grad-text">One Dashboard</span>
            </Reveal>
            <ul className="pe-dash-list">
              {DASH_POINTS.map((p, i) => {
                const Icon = p.icon;
                return (
                  <Reveal as="li" key={p.title} className="pe-dash-item" delay={i * 60}>
                    <span className="pe-ico" style={accent(p.c)}><Icon /></span>
                    <div>
                      <h4>{p.title}</h4>
                      <p>{p.desc}</p>
                    </div>
                  </Reveal>
                );
              })}
            </ul>
          </div>

          <Reveal delay={140}>
            <DashboardMock />
          </Reveal>
        </div>
      </div>
    </section >
  );
}

/* ============================ DEPLOY ANYWHERE ============================ */
const TARGETS = [
  { name: "Raspberry Pi 4", sub: ".pxe package", meta: "aarch64 · TFLite", icon: Cherry, c: "#e11f52" },
  { name: "UNO Q", sub: ".pxe package", meta: "uno-q · EON compiled", icon: Package, c: "#6366f1" },
  { name: "TensorFlow Lite", sub: "Cross-platform", meta: ".tflite · float32 / int8", icon: Binary, c: "#f59e0b" },
];

export function DeployAnywhere() {
  return (
    <section className="pe-section" id="deploy">
      <div className="pe-container">
        <div className="pe-section-head">
          <Reveal as="span" className="pe-eyebrow"><Cable /> Deploy</Reveal>
          <Reveal as="h2" className="pe-h2" delay={60}>
            Deploy <span className="pe-grad-text">Anywhere</span>
          </Reveal>
          <Reveal as="p" className="pe-lede" delay={120}>
            Export optimized builds for the real hardware your product runs on — no
            lock-in, no rewrites.
          </Reveal>
        </div>

        <div className="pe-deploy-grid">
          {TARGETS.map((t, i) => {
            const Icon = t.icon;
            return (
              <Reveal
                key={t.name}
                className="pe-card pe-deploy-card"
                style={accent(t.c)}
                delay={(i % 3) * 70}
                onMouseMove={(e: React.MouseEvent<HTMLElement>) => {
                  const r = e.currentTarget.getBoundingClientRect();
                  e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`);
                  e.currentTarget.style.setProperty("--my", `${e.clientY - r.top}px`);
                }}
              >
                <span className="pe-ico pe-ico-grad"><Icon /></span>
                <div>
                  <h3>{t.name}</h3>
                  <p>{t.sub}</p>
                </div>
                <div className="pe-deploy-meta">{t.meta}</div>
              </Reveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}

/* ============================ STATS ============================ */
const STATS: { value: number; suffix: string; decimals?: number; label: string }[] = [
  { value: 500, suffix: "K+", label: "Projects" },
  { value: 50, suffix: "M+", label: "Samples" },
  { value: 1, suffix: "M+", label: "Deployments" },
  { value: 100, suffix: "+", label: "Edge Devices" },
  { value: 99.99, suffix: "%", decimals: 2, label: "Uptime" },
];

export function Stats() {
  return (
    <section className="pe-section" style={{ paddingTop: 0 }}>
      <div className="pe-container">
        <Reveal className="pe-stats-wrap">
          <div className="pe-stats-grid">
            {STATS.map((s, i) => (
              <div key={s.label} style={{ display: "contents" }}>
                <div>
                  <div className="pe-stat-num pe-grad-text">
                    <Counter value={s.value} suffix={s.suffix} decimals={s.decimals ?? 0} />
                  </div>
                  <div className="pe-stat-label">{s.label}</div>
                </div>
                {i < STATS.length - 1 && <div className="pe-stat-div" />}
              </div>
            ))}
          </div>
        </Reveal>
      </div>
    </section>
  );
}

/* ============================ WHY PETAL EDGE ============================ */
const ROWS = [
  "End-to-end Edge AI",
  "Automatic AI Labeling",
  "GPU Training",
  "Multi-format Export",
];

export function WhyPetalEdge() {
  return (
    <section className="pe-section" id="company">
      <div className="pe-container">
        <div className="pe-section-head">
          <Reveal as="span" className="pe-eyebrow"><Check /> Why Petal Edge</Reveal>
          <Reveal as="h2" className="pe-h2" delay={60}>
            The <span className="pe-grad-text">complete</span> edge AI stack
          </Reveal>
          <Reveal as="p" className="pe-lede" delay={120}>
            Most tools cover a slice of the lifecycle. Petal Edge covers all of it.
          </Reveal>
        </div>

        <div className="pe-why-grid">
          <Reveal className="pe-card pe-why-card pe-hi">
            <div className="pe-why-head">
              <span className="pe-ico pe-ico-grad"><Cpu /></span>
              <h3>Petal Edge</h3>
            </div>
            {ROWS.map((r) => (
              <div key={r} className="pe-why-row yes">
                <Check strokeWidth={3} /> {r}
                <span className="pe-tag-soft g">Included</span>
              </div>
            ))}
          </Reveal>

          <Reveal className="pe-card pe-why-card" delay={80}>
            <div className="pe-why-head">
              <span className="pe-ico"><Boxes /></span>
              <h3>Typical Tools</h3>
            </div>
            {ROWS.map((r, i) => {
              const has = i < 2; // limited coverage
              return (
                <div key={r} className={`pe-why-row ${has ? "yes" : "no"}`}>
                  {has ? <Check strokeWidth={3} /> : <X strokeWidth={2.5} />} {r}
                  <span className={`pe-tag-soft ${has ? "m" : "m"}`}>{has ? "Partial" : "Missing"}</span>
                </div>
              );
            })}
          </Reveal>
        </div>
      </div>
    </section>
  );
}
