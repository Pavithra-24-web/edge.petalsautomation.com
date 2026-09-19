import type { LiveClassificationRuntimeInfo } from "@/types/live-classification";
import { getRuntimeBadgeMeta } from "@/lib/live-classification-helpers";

interface Props {
  runtime: LiveClassificationRuntimeInfo;
}

export default function LiveClassificationModelBadge({ runtime }: Props) {
  const badge = getRuntimeBadgeMeta(runtime);

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      {badge.versionChip && (
        <span className="badge-blue">{badge.versionChip}</span>
      )}
      <span className="badge-gray">{badge.artifactChip}</span>
      {badge.architecture && (
        <span className="text-gray-400">{badge.architecture}</span>
      )}
      {badge.accuracyText && (
        <span className="text-gray-500">
          {badge.accuracyText}
        </span>
      )}
    </div>
  );
}
