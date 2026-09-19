"use client";
import { useState } from "react";
import type { LiveClassificationDetection } from "@/types/live-classification";

interface Props {
  imageUrl: string | null;
  detections?: LiveClassificationDetection[];
  classificationLabel?: string;
  sensorType: string | null;
}

export default function LiveClassificationImagePreview({
  imageUrl,
  detections,
  classificationLabel,
  sensorType,
}: Props) {
  const [imgError, setImgError] = useState(false);

  if (!imageUrl || imgError) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-center text-gray-600">
        <div className="text-4xl mb-3">📡</div>
        <p className="text-xs">No visual preview available</p>
        {sensorType && (
          <p className="text-xs text-gray-700 mt-1">Sensor: {sensorType}</p>
        )}
      </div>
    );
  }

  const hasBoxes = detections && detections.length > 0;

  return (
    <div className="relative rounded-lg overflow-hidden bg-gray-800">
      <img
        src={imageUrl}
        alt="Sample"
        className="w-full h-auto block"
        onError={() => setImgError(true)}
      />

      {/* Classification label badge */}
      {classificationLabel && !hasBoxes && (
        <div className="absolute top-2 left-2">
          <span className="badge-green text-xs">{classificationLabel}</span>
        </div>
      )}

      {/* Detection bounding boxes */}
      {hasBoxes && (
        <svg
          className="absolute inset-0 w-full h-full"
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
        >
          {detections!.map((det, i) => {
            const { x1, y1, x2, y2 } = det.bbox;
            return (
              <g key={i}>
                <rect
                  x={x1}
                  y={y1}
                  width={x2 - x1}
                  height={y2 - y1}
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="0.008"
                />
                <rect
                  x={x1}
                  y={Math.max(0, y1 - 0.04)}
                  width={Math.min(0.35, x2 - x1)}
                  height={0.04}
                  fill="#10b981"
                />
                <text
                  x={x1 + 0.005}
                  y={Math.max(0.03, y1 - 0.007)}
                  fontSize="0.03"
                  fill="white"
                >
                  {det.label}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
