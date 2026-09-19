"use client";

import {
  Brain, Cpu,
  Sparkles, Boxes, Wand2, Gauge, GitBranch, Radar, MousePointerClick,
  FlaskConical, CircuitBoard,
  Check, Layers, Zap, RadioTower, Cherry, Wifi,
  ShieldCheck, Cloud, Users,
} from "lucide-react";
import { Reveal } from "./primitives";
import SolutionWorkflowCarousel from "./solutions/workflow/SolutionWorkflowCarousel";
import { getSolution } from "@/lib/solutions.config";

/** Inline CSS custom property for the per-icon accent. */
const accent = (c: string) => ({ ["--c"]: c }) as React.CSSProperties;

/* ============================ TRUSTED BY ============================ */
const LOGOS = [
  { name: "Intel", icon: Cpu, c: "#0f6fc0" },
  { name: "NVIDIA", icon: Zap, c: "#76b900" },
  { name: "Qualcomm", icon: RadioTower, c: "#5163e6" },
  { name: "Arduino", icon: CircuitBoard, c: "#08979d" },
  { name: "Raspberry Pi", icon: Cherry, c: "#e11f52" },
  { name: "Espressif", icon: Wifi, c: "#e7352c" },
];

export function TrustedBy() {
  return (
    <section className="pe-container pe-trusted">
      <Reveal className="pe-logos">
        {LOGOS.map(({ name, icon: Icon, c }) => (
          <span key={name} className="pe-logo">
            <span className="pe-logo-ico" style={accent(c)}><Icon strokeWidth={1.9} /></span>
            {name}
          </span>
        ))}
      </Reveal>
    </section>
  );
}

/* ============================ WORKFLOW ============================ */

/* The homepage shows the same vision pipeline the Computer Vision solution page
   shows, from the same config entry — one definition, two surfaces. Non-null is
   safe here: the slug is part of the static SOLUTIONS table. */
const VISION_SOLUTION = getSolution("computer-vision")!;

/* Trust bar beneath the workflow — each column carries its own accent. */
const PLATFORM_BAR = [
  { icon: ShieldCheck, c: "#8b5cf6", title: "End-to-End Security", desc: "Your data and models are protected at every step." },
  { icon: Zap, c: "#3b82f6", title: "Scalable by Design", desc: "Built to grow with your data, team, and devices." },
  { icon: Cloud, c: "#14b8a6", title: "Works Anywhere", desc: "From edge devices to cloud — seamless and connected." },
  { icon: Users, c: "#22c55e", title: "Developer Friendly", desc: "Powerful APIs, SDKs, and tools to accelerate your workflow." },
];

export function Workflow() {
  return (
    <section className="pe-section" id="platform" aria-labelledby="platform-title">
      <div className="pe-container">
        <div className="pe-section-head">
          <Reveal as="span" className="pe-eyebrow"><Layers /> The Platform</Reveal>
          <Reveal as="h2" className="pe-h2" delay={60} id="platform-title">
            One Platform. <span className="pe-grad-text">Endless AI Possibilities.</span>
          </Reveal>
          <Reveal as="p" className="pe-lede" delay={120}>
            Every stage of the edge AI lifecycle — from raw signal to monitored fleet —
            lives in a single, connected workflow.
          </Reveal>
        </div>
      </div>

      {/* The pipeline itself, shared with /solutions/computer-vision rather than
          restated here: one stage at a time, on a clock, with the rail beneath.
          `embedded` keeps the carousel's own heading and <section> out of the
          way — this section already has both. It sits outside .pe-container
          because it brings its own; nesting the two would double the gutter. */}
      <SolutionWorkflowCarousel solution={VISION_SOLUTION} embedded />

      <div className="pe-container">
        <div className="pe-feature-bar">
          {PLATFORM_BAR.map((f, i) => {
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

/* ============================ FEATURES ============================ */
const FEATURES = [
  { icon: Boxes, c: "#6366f1", title: "Smart Dataset Management", desc: "Organize, filter, and split datasets with an explorer built for scale." },
  { icon: Wand2, c: "#8b5cf6", title: "Automatic AI Labeling", desc: "Bootstrap annotations with AI-assisted labeling and human-in-the-loop review." },
  { icon: Gauge, c: "#06b6d4", title: "GPU Training", desc: "Accelerated training jobs with live loss and accuracy telemetry." },
  { icon: GitBranch, c: "#14b8a6", title: "Dataset Versioning", desc: "Track every change with immutable, reproducible dataset versions." },
  { icon: Radar, c: "#22c55e", title: "Real-time Device Monitoring", desc: "Watch inference latency, throughput, and health across your fleet." },
  { icon: MousePointerClick, c: "#3b82f6", title: "One-click Deployment", desc: "Ship optimized builds to any supported target in a single action." },
  { icon: FlaskConical, c: "#0ea5e9", title: "Model Testing & Evaluation", desc: "Confusion matrices, per-class metrics, and live classification testing." },
];

export function Features() {
  return (
    <section className="pe-section" id="features">
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

        <div className="pe-feat-grid">
          {FEATURES.map((f, i) => {
            const Icon = f.icon;
            return (
              <Reveal key={f.title} className="pe-card pe-card-hover pe-feat-card" delay={(i % 3) * 70}>
                <span className="pe-ico" style={accent(f.c)}><Icon /></span>
                <h3>{f.title}</h3>
                <p>{f.desc}</p>
              </Reveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}

/* ============================ MODEL SHOWCASE ============================ */
type Model = {
  title: string;
  desc: string;
  acc: string;
  train: string;
  deploy: string;
  viz: React.ReactNode;
};

const MODELS: Model[] = [
  {
    title: "Object Detection", desc: "Locate and classify multiple objects with bounding boxes.",
    acc: "mAP 0.91", train: "GPU", deploy: "Edge",
    viz: (<>
      <span className="pe-viz-bbox" data-l="cup" style={{ left: "20%", top: "30%", width: "34%", height: "44%" }} />
      <span className="pe-viz-bbox" data-l="pen" style={{ left: "60%", top: "48%", width: "26%", height: "30%", ["--c" as string]: "#06b6d4" } as React.CSSProperties} />
    </>),
  },
  {
    title: "Vision Pro", desc: "High-accuracy real-time detection tuned for edge throughput.",
    acc: "mAP 0.94", train: "GPU", deploy: "ESP32",
    viz: (<>
      <span className="pe-viz-bbox" data-l="0.97" style={{ left: "26%", top: "24%", width: "48%", height: "52%" }} />
      <span className="pe-viz-dot" style={{ left: "30%", top: "30%" }} />
    </>),
  },
  {
    title: "NanoVision", desc: "Tiny centroid detection for MCUs.",
    acc: "60 FPS", train: "GPU", deploy: "MCU",
    viz: (<>
      <span className="pe-viz-dot" style={{ left: "28%", top: "36%" }} />
      <span className="pe-viz-dot" style={{ left: "58%", top: "30%", ["--c" as string]: "#22c55e" } as React.CSSProperties} />
      <span className="pe-viz-dot" style={{ left: "46%", top: "62%", ["--c" as string]: "#8b5cf6" } as React.CSSProperties} />
    </>),
  },
  {
    title: "EdgeDetect Lite", desc: "Single-shot detection with an efficient mobile backbone.",
    acc: "mAP 0.88", train: "GPU", deploy: "Pi 4",
    viz: (<>
      <span className="pe-viz-bbox" data-l="face" style={{ left: "22%", top: "26%", width: "40%", height: "48%" }} />
      <span className="pe-viz-bbox" data-l="hand" style={{ left: "56%", top: "52%", width: "28%", height: "26%", ["--c" as string]: "#8b5cf6" } as React.CSSProperties} />
    </>),
  },
];

export function ModelShowcase() {
  return (
    <section className="pe-section" id="model">
      <div className="pe-container">
        <div className="pe-section-head">
          <Reveal as="span" className="pe-eyebrow"><Brain /> Models</Reveal>
          <Reveal as="h2" className="pe-h2" delay={60}>
            Train Any <span className="pe-grad-text">Edge AI Model</span>
          </Reveal>
          <Reveal as="p" className="pe-lede" delay={120}>
            Create production-ready object detection models with GPU-accelerated training.
          </Reveal>
        </div>

        <div className="pe-model-grid">
          {MODELS.map((m, i) => (
            <Reveal key={m.title} className="pe-card pe-card-hover pe-model-card" delay={(i % 4) * 70}>
              <div className="pe-model-viz">{m.viz}</div>
              <h3>{m.title}</h3>
              <p>{m.desc}</p>
              <div className="pe-badges">
                <span className="pe-badge ok"><Check strokeWidth={3} /> {m.acc}</span>
                <span className="pe-badge">{m.train}</span>
                <span className="pe-badge">{m.deploy}</span>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
