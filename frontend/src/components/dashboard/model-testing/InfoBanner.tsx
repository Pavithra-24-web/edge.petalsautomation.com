"use client";

import Link from "next/link";
import { CheckSquare } from "lucide-react";

interface InfoBannerProps {
  projectName?: string | null;
  impulseName?: string | null;
}

export default function InfoBanner({ projectName, impulseName }: InfoBannerProps) {
  return (
    <div className="pe-mt-hero" aria-label="model-testing-hero">
      <svg
        className="pe-mt-hero-wave"
        viewBox="0 0 320 90"
        fill="none"
        aria-hidden="true"
      >
        <path
          d="M0 45 Q 40 10 80 45 T 160 45 T 240 45 T 320 45"
          stroke="rgba(255,255,255,0.65)"
          strokeWidth="2"
          fill="none"
        />
        <path
          d="M0 60 Q 40 30 80 60 T 160 60 T 240 60 T 320 60"
          stroke="rgba(255,255,255,0.35)"
          strokeWidth="1.5"
          fill="none"
        />
      </svg>

      <div className="pe-mt-hero-crumb">
        {projectName && (
          <>
            <span>{projectName}</span>
            <span className="pe-mt-hero-crumb-sep">/</span>
          </>
        )}
        <span className="pe-mt-hero-crumb-current">
          {impulseName ?? "Model testing"}
        </span>
      </div>

      <div className="pe-mt-hero-info">
        <span className="pe-mt-hero-info-icon" aria-hidden="true">
          <CheckSquare size={15} />
        </span>
        <span>
          This lists all test data. You can manage this data through{" "}
          <Link href="/dashboard/data">Data acquisition</Link>.
        </span>
      </div>
    </div>
  );
}
