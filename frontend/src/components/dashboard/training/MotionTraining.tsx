"use client";
import { useRouter } from "next/navigation";
import { Activity } from "lucide-react";
import ImpulseNotReady from "@/components/dashboard/ImpulseNotReady";

function MotionTrainingIcon() {
  return (
    <svg
      width="56"
      height="56"
      viewBox="0 0 24 24"
      fill="none"
      stroke="url(#pe-motion-train-grad)"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="pe-motion-train-grad" x1="0" y1="0" x2="24" y2="24" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#8b5cf6" />
          <stop offset="1" stopColor="#a855f7" />
        </linearGradient>
      </defs>
      <path d="M3 3v18h18" />
      <path d="M7 16l3-6 3 3 4-8" />
    </svg>
  );
}

/**
 * Training — Motion content component.
 *
 * Object Detection's Training page picks a vision architecture (EdgeDetect
 * Lite, NanoVision, Vision Pro) and drives the shared training worker for
 * image models. Motion's own architectures (Dense, Conv1D, LSTM) are chosen
 * on the Impulse page's learning block, and the training worker / model
 * builder are not yet wired for them — that lands with
 * [Phase 6](../../../../../docs/Motion%20recognition/motion_phase6.md).
 * Phase 0 renders this surface's shell rather than a training UI that would
 * call a backend path nothing currently serves.
 *
 * Once wired, this page will show live progress plus loss, accuracy,
 * precision, recall, F1 and a confusion matrix (motion_phase0.md §6.4).
 */
export default function MotionTraining() {
  const router = useRouter();

  return (
    <ImpulseNotReady
      title={<>Training arrives with <span className="pe-warn-title-accent">motion learning</span></>}
      description="Once motion training is wired up, this page will track live progress and show loss, accuracy, precision, recall, F1 and a confusion matrix for your impulse."
      icon={<MotionTrainingIcon />}
      tip="You can still design your impulse — pick a processing block and a learning block — while training catches up."
      actions={
        <button
          type="button"
          onClick={() => router.push("/dashboard/impulse")}
          className="pe-warn-cta"
        >
          <Activity size={15} /> Back to Impulse design
        </button>
      }
    />
  );
}
