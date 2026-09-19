"use client";

import {
  Activity, BarChart3, Binary, BrainCircuit, Cable, Cherry, CircuitBoard,
  Cloud, CloudUpload, Code2, Cpu, Database, FileText, FlaskConical, Gauge,
  GitBranch, Layers, MousePointerClick, Package, PlayCircle, PlugZap, Radar,
  RadioTower, Rocket, Server, ShieldCheck, Sparkles, Wand2,
} from "lucide-react";
import type { IconName } from "@/lib/solutions.config";

/** Name → component map. Keep in sync with IconName in solutions.config.ts. */
const ICONS: Record<IconName, React.ElementType> = {
  Activity, BarChart3, Binary, BrainCircuit, Cable, Cherry, CircuitBoard,
  Cloud, CloudUpload, Code2, Cpu, Database, FileText, FlaskConical, Gauge,
  GitBranch, Layers, MousePointerClick, Package, PlayCircle, PlugZap, Radar,
  RadioTower, Rocket, Server, ShieldCheck, Sparkles, Wand2,
};

export function SolIcon({ name }: { name: IconName }) {
  const Icon = ICONS[name] ?? Cpu;
  return <Icon />;
}

/** Inline CSS custom property for a per-item accent (matches Showcase.tsx). */
export const accent = (c: string) => ({ ["--c"]: c }) as React.CSSProperties;
