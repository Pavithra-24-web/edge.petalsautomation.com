"use client";

import {
  Brain, Check, TrendingUp, Timer,
  Crosshair, Box, LayoutGrid, Smartphone,
  Coffee, Pen, Car, User, Hand,
  Zap, Target, LineChart, ShieldCheck,
} from "lucide-react";
import { Reveal } from "./primitives";

/** Inline CSS custom property for a per-element accent color. */
const cvar = (name: string, c: string) =>
  ({ [name]: c }) as React.CSSProperties;

/* ============================ MODEL CARDS ============================ */
type Model = {
  title: string;
  icon: React.ElementType;
  c: string;               // per-card neon theme color
  desc: string;
  bullets: string[];
  metric: { icon: React.ElementType; label: string };
  train: string;
  deploy: string;
  viz: React.ReactNode;
};

const MODELS: Model[] = [
  {
    title: "Object Detection",
    icon: Crosshair,
    c: "#22c55e",
    desc: "Locate and classify multiple objects within images using bounding boxes.",
    bullets: [
      "Detect multiple objects in real-time",
      "High accuracy with minimal latency",
      "Ideal for surveillance, robotics & more",
    ],
    metric: { icon: TrendingUp, label: "mAP 0.91" },
    train: "GPU",
    deploy: "Edge",
    viz: (
      <>
        <Coffee className="pe-viz-subject" style={{ left: "20%", top: "34%", width: 34, height: 34 }} />
        <Pen className="pe-viz-subject" style={{ left: "64%", top: "48%", width: 26, height: 26 }} />
        <span className="pe-viz-bbox" data-l="cup 0.93" style={{ left: "15%", top: "28%", width: "36%", height: "46%" }} />
        <span className="pe-viz-bbox" data-l="pen 0.89" style={{ left: "58%", top: "42%", width: "30%", height: "36%", ...cvar("--c", "#06b6d4") }} />
      </>
    ),
  },
  {
    title: "Vision Pro",
    icon: Box,
    c: "#3b82f6",
    desc: "High-accuracy real-time detection tuned for edge throughput.",
    bullets: [
      "State-of-the-art YOLO architecture",
      "Optimized for speed & accuracy",
      "Perfect for edge deployment",
    ],
    metric: { icon: TrendingUp, label: "mAP 0.94" },
    train: "GPU",
    deploy: "ESP32",
    viz: (
      <>
        <Car className="pe-viz-subject" style={{ left: "50%", top: "50%", width: 66, height: 66, transform: "translate(-50%, -50%)" }} />
        <span className="pe-viz-bbox" data-l="0.97" style={{ left: "22%", top: "26%", width: "56%", height: "48%", ...cvar("--c", "#3b82f6") }} />
      </>
    ),
  },
  {
    title: "NanoVision",
    icon: LayoutGrid,
    c: "#8b5cf6",
    desc: "Tiny centroid detection for MCUs.",
    bullets: [
      "Ultra-lightweight & fast",
      "Detects multiple tiny objects",
      "Built for microcontrollers",
    ],
    metric: { icon: Timer, label: "60 FPS" },
    train: "GPU",
    deploy: "MCU",
    viz: (
      <>
        <span className="pe-viz-dot" style={{ left: "30%", top: "38%" }} />
        <span className="pe-viz-dot" style={{ left: "60%", top: "30%", ...cvar("--c", "#22c55e") }} />
        <span className="pe-viz-dot" style={{ left: "48%", top: "64%", ...cvar("--c", "#8b5cf6") }} />
      </>
    ),
  },
  {
    title: "EdgeDetect Lite",
    icon: Smartphone,
    c: "#ec4899",
    desc: "Single-shot detection with an efficient mobile backbone.",
    bullets: [
      "Lightweight & mobile-friendly",
      "Great balance of speed & accuracy",
      "Optimized for on-device inference",
    ],
    metric: { icon: TrendingUp, label: "mAP 0.88" },
    train: "GPU",
    deploy: "Pi 4",
    viz: (
      <>
        <User className="pe-viz-subject" style={{ left: "22%", top: "30%", width: 36, height: 36 }} />
        <Hand className="pe-viz-subject" style={{ left: "62%", top: "48%", width: 30, height: 30 }} />
        <span className="pe-viz-bbox" data-l="face 0.95" style={{ left: "16%", top: "24%", width: "38%", height: "50%", ...cvar("--c", "#ec4899") }} />
        <span className="pe-viz-bbox" data-l="hand 0.90" style={{ left: "56%", top: "42%", width: "30%", height: "34%", ...cvar("--c", "#8b5cf6") }} />
      </>
    ),
  },
];

/* ============================ FEATURE BAR ============================ */
const FEATURES = [
  { icon: Zap, c: "#22c55e", title: "GPU Accelerated", desc: "Train faster with the power of modern GPUs." },
  { icon: Target, c: "#3b82f6", title: "Edge Optimized", desc: "Deploy anywhere — from cloud to microcontrollers." },
  { icon: LineChart, c: "#8b5cf6", title: "High Performance", desc: "Best-in-class accuracy with real-time inference." },
  { icon: ShieldCheck, c: "#ec4899", title: "Production Ready", desc: "Built for reliability, scalability, and real-world impact." },
];

export default function ModelSection() {
  return (
    <section className="pe-section pe-models" id="model">
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
          {MODELS.map((m, i) => {
            const Icon = m.icon;
            const MetricIcon = m.metric.icon;
            return (
              <Reveal
                key={m.title}
                className="pe-card pe-card-hover pe-model-card"
                delay={(i % 4) * 70}
                style={cvar("--card-c", m.c)}
              >
                <div className="pe-model-viz">{m.viz}</div>
                <div className="pe-model-head">
                  <span className="pe-model-ico"><Icon /></span>
                  <h3>{m.title}</h3>
                </div>
                <p>{m.desc}</p>
                <ul className="pe-model-bullets">
                  {m.bullets.map((b) => (
                    <li key={b}><Check strokeWidth={3} /> {b}</li>
                  ))}
                </ul>
                <div className="pe-badges">
                  <span className="pe-badge pe-metric"><MetricIcon /> {m.metric.label}</span>
                  <span className="pe-badge">{m.train}</span>
                  <span className="pe-badge">{m.deploy}</span>
                </div>
              </Reveal>
            );
          })}
        </div>

        <div className="pe-feature-bar">
          {FEATURES.map((f, i) => {
            const Icon = f.icon;
            return (
              <Reveal key={f.title} className="pe-feature-col" delay={(i % 4) * 70}>
                <span className="pe-feature-ico" style={cvar("--c", f.c)}><Icon /></span>
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
