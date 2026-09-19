"use client";
import useImage from "use-image";
import { AlertTriangle, Loader2 } from "lucide-react";
import { AnnotationEditor } from "../AnnotationEditor";

// Labeling Queue plan, Phase 4 §3/§4.A. Thin wrapper: resolves the presigned
// URL → useImage → <AnnotationEditor/>. AnnotationEditor is rendered
// unmodified — this component owns none of the drawing/shortcut/undo logic.

type QueueCanvasProps = {
  filename: string;
  imageUrl: string | null;
  loading: boolean;
  loadError: boolean;
  missing: boolean;
  initialBoxes: any[];
  resetKey: string;
  labels: any[];
  onChange: (boxes: any[], primary?: { id?: string; name?: string } | null) => void | Promise<void>;
  onCreateLabel: (name: string) => Promise<{ id: string; name: string } | null>;
  colorForLabel: (label: string) => string;
};

export default function QueueCanvas({
  filename, imageUrl, loading, loadError, missing,
  initialBoxes, resetKey, labels, onChange, onCreateLabel, colorForLabel,
}: QueueCanvasProps) {
  const [image] = useImage(imageUrl || "", "anonymous");

  if (missing) {
    return (
      <div className="labeling-queue-canvas-status">
        <AlertTriangle size={22} className="text-amber-500" />
        <p>This image was deleted. Moving to the next one…</p>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="labeling-queue-canvas-status">
        <AlertTriangle size={22} className="text-red-500" />
        <p>Couldn&apos;t load this image. Try Next or Previous.</p>
      </div>
    );
  }

  if (loading || !imageUrl || !image) {
    return (
      <div className="labeling-queue-canvas-status">
        <Loader2 size={22} className="animate-spin" />
        <p>Loading {filename}…</p>
      </div>
    );
  }

  return (
    <AnnotationEditor
      image={image}
      initialBoxes={initialBoxes}
      resetKey={resetKey}
      labels={labels}
      onChange={onChange}
      onCreateLabel={onCreateLabel}
      colorForLabel={colorForLabel}
    />
  );
}
