"use client";
import { useRouter } from "next/navigation";
import type { LucideIcon } from "lucide-react";
import { Rocket } from "lucide-react";
import ImpulseNotReady from "@/components/dashboard/ImpulseNotReady";

interface MotionPhasePendingProps {
  /** Page name used in the title — "{feature} arrives with motion support". */
  feature: string;
  /** What this page will do once it's wired up for motion. */
  description: string;
  /** Page-specific tip shown in the bottom pill. */
  tip: string;
  /** Icon representing this page. */
  icon: LucideIcon;
}

/**
 * Shared "not built yet" full-page shell for the four impulse routes that
 * carry no motion content in Phase 0 — evaluation, retrain,
 * live-classification, post-processing (motion_phase0.md §0.6 / §4.2). One
 * component, reused with page-specific copy at each of the four call sites,
 * built on the same `ImpulseNotReady` shell already used by `MotionTraining`
 * and `MotionModelTesting` so every "arrives later" moment in the motion
 * workflow reads consistently. Never rendered for an object-detection
 * project.
 *
 * Evaluation and Live Classification get real motion content later, in
 * place, on this same route (Phase 7 and Phase 7.5 respectively); Retrain
 * and Post-processing keep this shell through the MVP.
 */
export default function MotionPhasePending({ feature, description, tip, icon: Icon }: MotionPhasePendingProps) {
  const router = useRouter();

  return (
    <ImpulseNotReady
      title={
        <>
          {feature} arrives with <span className="pe-warn-title-accent">motion support</span>
        </>
      }
      description={description}
      icon={<Icon size={56} strokeWidth={1.75} className="text-violet-500" aria-hidden="true" />}
      tip={tip}
      actions={
        <button
          type="button"
          onClick={() => router.push("/dashboard/impulse")}
          className="pe-warn-cta"
        >
          <Rocket size={15} /> Go to Impulse design
        </button>
      }
    />
  );
}
