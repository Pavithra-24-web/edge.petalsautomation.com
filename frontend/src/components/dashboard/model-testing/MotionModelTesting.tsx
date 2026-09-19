"use client";
import { useRouter } from "next/navigation";
import { FlaskConical } from "lucide-react";
import ImpulseNotReady from "@/components/dashboard/ImpulseNotReady";

function MotionTestingIcon() {
  return (
    <svg
      width="56"
      height="56"
      viewBox="0 0 24 24"
      fill="none"
      stroke="url(#pe-motion-test-grad)"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="pe-motion-test-grad" x1="0" y1="0" x2="24" y2="24" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#8b5cf6" />
          <stop offset="1" stopColor="#a855f7" />
        </linearGradient>
      </defs>
      <path d="M4 19V9m5 10V5m5 14v-7m5 7V11" />
    </svg>
  );
}

/**
 * Model Testing — Motion content component.
 *
 * Object Detection's Model Testing page inspects a trained model with an
 * image + bounding-box viewer (SampleInspectorCard's DetectionViewer) and a
 * per-row image thumbnail (SampleThumbnail) — both genuinely image-specific.
 * Motion's equivalent (CSV testing, live USB testing, signal prediction, a
 * confidence graph) is windowed-scoring work against a trained motion model,
 * which doesn't exist until motion training lands. Plan of record:
 * [Phase 7.6](../../../../../docs/Motion%20recognition/motion_phase7_6.md)
 * fills this shell with the windowed scoring path and a motion sample
 * inspector, on the existing model_testing_worker / ModelTestRun rows.
 */
export default function MotionModelTesting() {
  const router = useRouter();

  return (
    <ImpulseNotReady
      title={<>Model testing arrives with <span className="pe-warn-title-accent">motion scoring</span></>}
      description="Once wired up, this page will run CSV and live USB test samples against your trained model and show a confidence graph per recording."
      icon={<MotionTestingIcon />}
      tip="Model testing validates a trained model — train your impulse first, then come back here."
      actions={
        <button
          type="button"
          onClick={() => router.push("/dashboard/impulse/training")}
          className="pe-warn-cta"
        >
          <FlaskConical size={15} /> Go to Training
        </button>
      }
    />
  );
}
