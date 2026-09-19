"use client";

import { CheckCircle2, AlertTriangle } from "lucide-react";

type Props = {
  ready: boolean;
};

export default function DeploymentHero({ ready }: Props) {
  return (
    <section className="pe-dep2-surface pe-dep2-hero" aria-labelledby="pe-dep2-hero-title">
      <div className="min-w-0">
        <div className="pe-dep2-hero-eyebrow">Workspace · Deployment</div>
        <h1 id="pe-dep2-hero-title" className="pe-dep2-hero-title">
          Deploy your model to the{" "}
          <span className="pe-dep2-hero-title-accent">edge</span>
        </h1>
        <p className="pe-dep2-hero-sub">
          Build and download production-ready packages for your target device.
        </p>
        <div className="pe-dep2-hero-meta">
          {ready ? (
            <span className="pe-dep2-pill pe-dep2-pill--success" role="status">
              <CheckCircle2 size={12} aria-hidden="true" />
              Model Ready
            </span>
          ) : (
            <span className="pe-dep2-pill pe-dep2-pill--warn" role="status">
              <AlertTriangle size={12} aria-hidden="true" />
              Training Required
            </span>
          )}
        </div>
      </div>
    </section>
  );
}
