"use client";
import DeploymentContent from "./DeploymentContent";

/**
 * Motion's deployment shell — same content as Object Detection (see
 * DeploymentContent's header comment). It will show "impulse not fully
 * trained" until motion training exists; Phase 8 adds the ESP32 profile
 * §6.6 asks for.
 */
export default function MotionDeployment() {
  return <DeploymentContent />;
}
