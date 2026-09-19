"use client";

import { useEffect, useState } from "react";
import { ImageOff } from "lucide-react";
import { samplesApi } from "@/utils/api";

interface SampleThumbnailProps {
  sampleId: string | null;
  size?: "sm" | "lg" | "xl";
}

export default function SampleThumbnail({ sampleId, size = "sm" }: SampleThumbnailProps) {
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    setUrl(null);
    setError(false);
    if (!sampleId) return;

    let cancelled = false;
    samplesApi
      .download(sampleId)
      .then((res) => {
        if (!cancelled) {
          setUrl((res as any)?.data?.url ?? null);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError(true);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [sampleId]);

  const dim =
    size === "xl"
      ? "w-full h-56"
      : size === "lg"
        ? "w-32 h-32"
        : "w-11 h-11";
  const iconSize = size === "xl" ? 32 : size === "lg" ? 24 : 16;

  if (!sampleId || error || url === null) {
    return (
      <div
        className={`${dim} rounded-lg flex items-center justify-center flex-shrink-0 overflow-hidden
                    bg-violet-50 dark:bg-slate-800 border border-violet-100 dark:border-slate-700`}
        title={!sampleId ? "No sample linked" : error ? "Preview unavailable" : "Loading..."}
      >
        <ImageOff size={iconSize} className="text-slate-400" />
      </div>
    );
  }

  return (
    <img
      src={url}
      alt="Sample preview"
      className={`${dim} rounded-lg object-cover flex-shrink-0 border border-violet-100 dark:border-slate-700`}
      onError={() => setError(true)}
    />
  );
}
