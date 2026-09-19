"use client";

import {
  ArrowRight, PlayCircle, Check,
  CloudUpload, Database, Brain, Zap, Rocket, Activity, RefreshCw,
} from "lucide-react";
import PetalEdgeLogo from "@/components/PetalEdgeLogo";
import { Reveal } from "./primitives";
import HeroTitle from "./HeroTitle";

const NODES = [
  { label: "Collect Data", icon: CloudUpload, color: "#4f46e5" },
  { label: "Build", icon: Database, color: "#6366f1" },
  { label: "Train", icon: Brain, color: "#8b5cf6" },
  { label: "Optimize", icon: Zap, color: "#a855f7" },
  { label: "Deploy", icon: Rocket, color: "#06b6d4" },
  { label: "Monitor", icon: Activity, color: "#14b8a6" },
  { label: "Feedback", icon: RefreshCw, color: "#22c55e" },
];

const TRUST = ["No credit card", "Enterprise Ready", "GPU Training", "Open Source Compatible"];

const RADIUS = 42; // percent of orbit box

export default function Hero() {
  return (
    <section className="pe-hero" id="top">
      <div className="pe-container">
        <div className="pe-hero-grid">
          {/* ---- Left ---- */}
          <div>
            <Reveal className="pe-hero-badge">
              <span className="pe-pill">NEW</span>
              <span>
                UNO Q&nbsp;(.pxe) export is live —&nbsp;<b>deploy in one click</b>
              </span>
            </Reveal>

            <HeroTitle />

            <Reveal as="p" className="pe-hero-sub" delay={120}>
              Collect sensor data, build production-ready AI models, optimize them for
              embedded hardware, deploy to edge devices, and monitor them in real time
              — all from one intelligent platform.
            </Reveal>

            <Reveal className="pe-hero-cta" delay={180}>
              <a href="/login/" className="pe-btn pe-btn-primary pe-btn-lg">
                Get Started Free <ArrowRight />
              </a>
              <a href="/book-demo" className="pe-btn pe-btn-ghost pe-btn-lg">
                <PlayCircle /> Book Live Demo
              </a>
            </Reveal>

            <Reveal as="ul" className="pe-trust" delay={240}>
              {TRUST.map((t) => (
                <li key={t}>
                  <Check strokeWidth={3} /> {t}
                </li>
              ))}
            </Reveal>
          </div>

          {/* ---- Right: orbit workflow ---- */}
          <Reveal delay={200}>
            <div className="pe-orbit-wrap" aria-hidden="true">
              {/* Depth backdrop — base radial glow + a drifting aurora / gradient
                  mesh so the node graph reads as floating in luminous space. */}
              <div className="pe-orbit-aura" />
              <div className="pe-orbit-halo" />
              <div className="pe-aurora pe-aurora-1" />
              <div className="pe-aurora pe-aurora-2" />
              <div className="pe-aurora pe-aurora-3" />
              <div className="pe-orbit-shadow" />

              {/* Floating particles — distributed on a soft ring around the
                  core so they read as sparks orbiting the illustration, then
                  gently rise (existing float animation). */}
              {[...Array(16)].map((_, i) => {
                const a = (i / 16) * Math.PI * 2;
                const r = 34 + (i % 3) * 7; // 34–48% band around center
                return (
                  <span
                    key={i}
                    className="pe-particle"
                    style={{
                      left: `${50 + r * Math.cos(a)}%`,
                      top: `${50 + r * Math.sin(a)}%`,
                      animationDuration: `${5 + (i % 5)}s`,
                      animationDelay: `${i * 0.45}s`,
                    }}
                  />
                );
              })}

              {/* Gradient glow pooled directly under the central chip. */}
              <div className="pe-orbit-core-glow" />
              <div className="pe-orbit-core">
                <span className="pe-orbit-core-ring" />
                <PetalEdgeLogo
                  variant="icon"
                  size={82}
                  className="pe-orbit-logo"
                  label="Petal Edge"
                />
              </div>

              <div className="pe-orbit-spin">
                {/* Connecting wires from the core to each node. A bright signal
                    pulse travels outward along each wire (see .pe-signal). */}
                <svg
                  className="pe-orbit-lines"
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                >
                  <defs>
                    <linearGradient id="peOrbitLine" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stopColor="#818cf8" stopOpacity="0.55" />
                      <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.04" />
                    </linearGradient>
                    <linearGradient id="peSignal" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stopColor="#c7d2fe" stopOpacity="0" />
                      <stop offset="45%" stopColor="#818cf8" stopOpacity="1" />
                      <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  {NODES.map((n, i) => {
                    const angle = (i / NODES.length) * 2 * Math.PI - Math.PI / 2;
                    const x = 50 + RADIUS * Math.cos(angle);
                    const y = 50 + RADIUS * Math.sin(angle);
                    return (
                      <g key={n.label}>
                        <line
                          x1="50" y1="50" x2={x} y2={y}
                          stroke="url(#peOrbitLine)"
                          strokeWidth="0.5"
                          strokeLinecap="round"
                        />
                        <line
                          className="pe-signal"
                          x1="50" y1="50" x2={x} y2={y}
                          stroke="url(#peSignal)"
                          strokeWidth="0.9"
                          strokeLinecap="round"
                          pathLength={100}
                          style={{ animationDelay: `${i * 0.34}s` }}
                        />
                      </g>
                    );
                  })}
                </svg>

                {NODES.map((n, i) => {
                  const angle = (i / NODES.length) * 2 * Math.PI - Math.PI / 2;
                  const left = 50 + RADIUS * Math.cos(angle);
                  const top = 50 + RADIUS * Math.sin(angle);
                  const Icon = n.icon;
                  return (
                    <div
                      key={n.label}
                      className="pe-orbit-slot"
                      style={{ left: `${left}%`, top: `${top}%` }}
                    >
                      <div className="pe-orbit-node">
                        <div
                          className="pe-node-card"
                          style={
                            {
                              animationDelay: `${i * 0.5}s`,
                              "--c": n.color,
                            } as React.CSSProperties
                          }
                        >
                          <span className="pe-node-ico" style={{ background: n.color }}>
                            <Icon />
                          </span>
                          {n.label}
                          <span className="pe-node-status" />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
