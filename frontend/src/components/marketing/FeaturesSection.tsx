"use client";

import {
  Sparkles, Zap, Database, Bot, Cpu, GitBranch, Activity, Send,
  Check, ArrowRight, FlaskConical, BarChart3, PlayCircle, FileText,
  CloudUpload, ShieldCheck, Code2, Cherry, Package, Rocket,
  Server,
  PlugZap,
  TagsIcon,
  Workflow,
} from "lucide-react";
import { Reveal } from "./primitives";
import { SiTensorflow } from "react-icons/si";

<SiTensorflow className="w-7 h-7 text-[#FF6F00]" />

import { SiNvidia } from "react-icons/si";

<SiNvidia className="w-7 h-7 text-green-500" />

/** Inline CSS custom property for the per-icon accent. */
const accent = (c: string) => ({ ["--c"]: c }) as React.CSSProperties;

/* ---- Isometric stacked-layers illustration (decorative, CSS-token colors) -- */
function LayersArt() {
  const cx = 140;
  const w = 92;
  const h = 44;
  const d = 15;
  const tiers = [
    { cy: 204, top: "url(#peLayerC)", left: "#312e81", right: "#3730a3" }, // bottom / farthest
    { cy: 132, top: "url(#peLayerB)", left: "#4338ca", right: "#4f46e5" },
    { cy: 60, top: "url(#peLayerA)", left: "#5b21b6", right: "#7c3aed" },  // top / nearest
  ];
  const top = (cy: number) => `${cx},${cy - h} ${cx + w},${cy} ${cx},${cy + h} ${cx - w},${cy}`;
  const left = (cy: number) => `${cx - w},${cy} ${cx},${cy + h} ${cx},${cy + h + d} ${cx - w},${cy + d}`;
  const right = (cy: number) => `${cx + w},${cy} ${cx},${cy + h} ${cx},${cy + h + d} ${cx + w},${cy + d}`;
  return (
    <svg className="pe-feat2-art" viewBox="0 0 280 290" role="img" aria-label="Stacked model layers" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="peLayerA" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#a78bfa" /><stop offset="100%" stopColor="#818cf8" />
        </linearGradient>
        <linearGradient id="peLayerB" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#818cf8" /><stop offset="100%" stopColor="#60a5fa" />
        </linearGradient>
        <linearGradient id="peLayerC" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#6366f1" /><stop offset="100%" stopColor="#3b82f6" />
        </linearGradient>
        <radialGradient id="peLayerGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="rgba(124,58,237,0.55)" /><stop offset="100%" stopColor="rgba(124,58,237,0)" />
        </radialGradient>
      </defs>
      <ellipse cx="140" cy="158" rx="140" ry="134" fill="url(#peLayerGlow)" />
      {tiers.map((t, i) => (
        <g key={i}>
          <polygon points={left(t.cy)} fill={t.left} opacity="0.9" />
          <polygon points={right(t.cy)} fill={t.right} opacity="0.95" />
          <polygon points={top(t.cy)} fill={t.top} stroke="rgba(255,255,255,0.28)" strokeWidth="1" />
        </g>
      ))}
      {/* sparkle on the top tier */}
      <path
        d="M140 34 C142 46 146 50 158 52 C146 54 142 58 140 70 C138 58 134 54 122 52 C134 50 138 46 140 34 Z"
        fill="#fff" opacity="0.95"
      />
    </svg>
  );
}

/* ============================ FEATURE CARDS ============================ */
const FEATURES = [
  { icon: Database, c: "#8b5cf6", title: "Smart Dataset Management", desc: "Organize, filter, and split datasets with an explorer built for scale.", bullets: ["Advanced filtering", "Auto split", "Data explorer"] },
  { icon: TagsIcon, c: "#7c6cf6", title: "Automatic AI Labeling", desc: "Bootstrap annotations with AI-assisted labeling and human-in-the-loop review.", bullets: ["AI assistance", "Human review", "Quality control"] },
  { icon: Workflow, c: "#3b82f6", title: "GPU Training", desc: "Accelerated training jobs with live loss and accuracy telemetry.", bullets: ["Faster training", "Live metrics", "High accuracy"] },
  { icon: GitBranch, c: "#14b8a6", title: "Dataset Versioning", desc: "Track every change with immutable, reproducible dataset versions.", bullets: ["Version history", "Diff tracking", "Reproducible"] },
  { icon: Activity, c: "#22c55e", title: "Real-time Device Monitoring", desc: "Watch inference latency, throughput, and health across your fleet.", bullets: ["Live telemetry", "Health status", "Alerts"] },
  { icon: Send, c: "#3b82f6", title: "One-click Deployment", desc: "Ship optimized builds to any supported target in a single action.", bullets: ["Multi-target", "OTA ready", "Secure deploy"] },
];

/* Confusion matrix — diagonal is the correct-prediction count (highlighted). */
const CONFUSION = [
  [92, 3, 1],
  [2, 95, 3],
  [1, 4, 94],
];

const BAR = [
  { icon: BarChart3, c: "#8b5cf6", title: "Built for Scale", desc: "Handle projects and datasets of any size." },
  { icon: CloudUpload, c: "#14b8a6", title: "Cloud & Edge", desc: "Deploy anywhere — from cloud to microcontrollers." },
  { icon: ShieldCheck, c: "#8b5cf6", title: "Secure by Design", desc: "Your data, models, and devices are protected." },
  { icon: Code2, c: "#3b82f6", title: "Developer Friendly", desc: "APIs, SDKs, and CLI tools to fit your workflow." },
];

const GAUGE_VALUE = 0.93;
const GAUGE_R = 50;
const GAUGE_C = 2 * Math.PI * GAUGE_R;

export function FeaturesShowcase() {
  return (
    <section className="pe-section pe-features2" id="features">
      <div className="pe-container">
        <div className="pe-section-head">
          <Reveal as="span" className="pe-eyebrow"><Sparkles /> Features</Reveal>
          <Reveal as="h2" className="pe-h2" delay={60}>
            Everything you need to ship <span className="pe-grad-text">edge AI</span>
          </Reveal>
          <Reveal as="p" className="pe-lede" delay={120}>
            A complete toolkit for the entire lifecycle, engineered for teams that
            move fast without cutting corners.
          </Reveal>
        </div>

        <div className="pe-feat2-grid">
          {/* Tall left panel — spans both rows */}
          <Reveal className="pe-card pe-feat2-tall">
            <span className="pe-ico pe-ico-grad" style={accent("#8b5cf6")}><Zap /></span>
            <h3 className="pe-feat2-tall-h">
              Powerful features.<br />Built for <span className="pe-grad-text">speed.</span>
            </h3>
            <span className="pe-feat2-rule" />
            <p className="pe-feat2-tall-p">
              From dataset to deployment — the full workflow in one platform:
            </p>
            <ul className="pe-feat2-tall-list">
              <li><Check strokeWidth={3} /> Collect &amp; organize datasets</li>
              <li><Check strokeWidth={3} /> Label with AI assistance</li>
              <li><Check strokeWidth={3} /> Train on accelerated GPUs</li>
              <li><Check strokeWidth={3} /> Optimize for edge hardware</li>
              <li><Check strokeWidth={3} /> Deploy in one click</li>
              <li><Check strokeWidth={3} /> Monitor your fleet live</li>
            </ul>
            <LayersArt />
          </Reveal>

          {/* 6 feature cards */}
          {FEATURES.map((f, i) => {
            const Icon = f.icon;
            return (
              <Reveal
                key={f.title}
                className="pe-card pe-card-hover pe-feat2-card"
                delay={(i % 3) * 70}
                style={accent(f.c)}
              >
                <div className="pe-feat2-top">
                  <span className="pe-ico pe-ico-grad" style={accent(f.c)}><Icon /></span>
                  <h3>{f.title}</h3>
                </div>
                <p className="pe-feat2-desc">{f.desc}</p>
                <ul className="pe-feat2-bullets">
                  {f.bullets.map((b) => (
                    <li key={b}><Check strokeWidth={3} /> {b}</li>
                  ))}
                </ul>
                <span className="pe-feat2-arrow"><ArrowRight /></span>
              </Reveal>
            );
          })}

          {/* Model Testing & Evaluation panel — spans the feature columns */}
          <Reveal className="pe-card pe-feat2-test" delay={100}>
            <div className="pe-test-left">
              <span className="pe-ico pe-ico-grad" style={accent("#8b5cf6")}><FlaskConical /></span>
              <h3>Model Testing &amp; Evaluation</h3>
              <p>Confusion matrices, per-class metrics, and live classification testing.</p>
              <div className="pe-test-chips">
                <span className="pe-test-chip"><BarChart3 /> Per-class metrics</span>
                <span className="pe-test-chip"><PlayCircle /> Live testing</span>
                <span className="pe-test-chip"><FileText /> Export reports</span>
              </div>
            </div>

            <div className="pe-test-right">
              <div className="pe-cm-panel">
                <div className="pe-cm-title">Confusion Matrix</div>
                <div className="pe-cm3">
                  {CONFUSION.flatMap((row, r) =>
                    row.map((v, c) => (
                      <i key={`${r}-${c}`} className={r === c ? "on" : ""}>{v}</i>
                    ))
                  )}
                </div>
              </div>

              <div className="pe-gauge">
                <div className="pe-gauge-cap">mAP@0.50</div>
                <div className="pe-gauge-ring">
                  <svg viewBox="0 0 120 120" className="pe-gauge-svg" aria-hidden="true">
                    <defs>
                      <linearGradient id="peGaugeRing" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stopColor="#22c55e" />
                        <stop offset="100%" stopColor="#14b8a6" />
                      </linearGradient>
                    </defs>
                    <circle className="pe-gauge-track" cx="60" cy="60" r={GAUGE_R} />
                    <circle
                      className="pe-gauge-prog"
                      cx="60" cy="60" r={GAUGE_R}
                      strokeDasharray={`${GAUGE_VALUE * GAUGE_C} ${GAUGE_C}`}
                      transform="rotate(-90 60 60)"
                    />
                  </svg>
                  <span className="pe-gauge-val">0.93</span>
                </div>
                <div className="pe-gauge-label">Excellent</div>
              </div>
            </div>
          </Reveal>
        </div>

        {/* Bottom feature bar — reuses the shared .pe-feature-bar */}
        <div className="pe-feature-bar">
          {BAR.map((f, i) => {
            const Icon = f.icon;
            return (
              <Reveal key={f.title} className="pe-feature-col" delay={(i % 4) * 70}>
                <span className="pe-feature-ico" style={accent(f.c)}><Icon /></span>
                <div className="pe-feature-body">
                  <h4>{f.title}</h4>
                  <p>{f.desc}</p>
                </div>
              </Reveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}

/* ============================ DEPLOY ANYWHERE ============================ */
type Target = {
  name: string;
  sub: string;
  meta: string;
  c: string;
  icon?: React.ElementType;
  iconText?: string;
};

const TARGETS: Target[] = [
  { name: "Raspberry Pi", sub: ".pxe package", meta: "Pi", icon: Server, c: "#e11f52" },
  { name: "UNO Q", sub: ".pxe package", meta: "uno-q", icon: PlugZap, c: "#6366f1" },
  { name: "TensorFlow Lite", sub: "Cross-platform", meta: ".tflite", icon: SiTensorflow, c: "#f59e0b" },
];

export function DeploySection() {
  return (
    <section className="pe-section pe-deploy2" id="deploy">
      <div className="pe-container">
        <div className="pe-section-head">
          <Reveal as="span" className="pe-eyebrow"><Rocket /> Deploy</Reveal>
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
                className="pe-card pe-card-hover pe-deploy-card"
                style={accent(t.c)}
                delay={(i % 3) * 70}
                onMouseMove={(e: React.MouseEvent<HTMLElement>) => {
                  const r = e.currentTarget.getBoundingClientRect();
                  e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`);
                  e.currentTarget.style.setProperty("--my", `${e.clientY - r.top}px`);
                }}
              >
                <span className="pe-ico pe-ico-grad" style={accent(t.c)}>
                  {Icon ? <Icon /> : <b className="pe-ico-txt">{t.iconText}</b>}
                </span>
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
