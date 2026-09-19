import { CheckCircle, XCircle } from "lucide-react";
import {
  fmtPct,
  isCorrectPrediction,
  resultMode,
  scoresToRows,
} from "@/lib/live-classification-helpers";
import type { LiveClassificationRunResult } from "@/types/live-classification";
import LiveClassificationDetectionsTable from "./LiveClassificationDetectionsTable";

interface Props {
  result: LiveClassificationRunResult;
}

export default function LiveClassificationSummaryCard({ result }: Props) {
  const mode = resultMode(result);
  const match = isCorrectPrediction(result.label, result.ground_truth_label);
  const modelMeta = [
    result.model_version != null ? `v${result.model_version}` : null,
    result.model_architecture,
  ].filter(Boolean).join(" · ");

  return (
    <div className="card space-y-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-gray-100">Classification result</h3>
          <p className="text-xs text-gray-500 mt-0.5 truncate max-w-[18rem]">
            {result.sample_name}
          </p>
        </div>
        <div className="text-right shrink-0">
          <p className="text-xs text-gray-500">{modelMeta}</p>
        </div>
      </div>

      {result.ground_truth_label && (
        <div className="flex items-center gap-2 text-xs">
          <span className="text-gray-500">Ground truth:</span>
          <span className="text-gray-300 font-medium">{result.ground_truth_label}</span>
          {match === true && <CheckCircle size={13} className="text-green-400" />}
          {match === false && <XCircle size={13} className="text-red-400" />}
        </div>
      )}

      {mode === "classification" && result.label != null && (
        <>
          <div className="flex items-baseline gap-3">
            <p className="text-2xl font-bold text-brand-400">{result.label}</p>
            <p className="text-sm text-gray-400">
              {result.confidence != null ? fmtPct(result.confidence) : ""}
            </p>
          </div>

          {result.scores && (
            <div className="space-y-2">
              {scoresToRows(result.scores).map(({ label, score }) => (
                <div key={label}>
                  <div className="flex justify-between text-xs text-gray-400 mb-0.5">
                    <span className={label === result.label ? "text-brand-400 font-medium" : ""}>
                      {label}
                    </span>
                    <span>{fmtPct(score)}</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-1.5">
                    <div
                      className={`h-1.5 rounded-full transition-all duration-500 ${
                        label === result.label ? "bg-brand-500" : "bg-gray-600"
                      }`}
                      style={{ width: `${score * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {mode === "detection" && (
        <>
          <div className="flex items-baseline gap-2">
            <p className="text-2xl font-bold text-brand-400">{result.count ?? 0}</p>
            <p className="text-sm text-gray-400">
              object{(result.count ?? 0) !== 1 ? "s" : ""} detected
            </p>
          </div>
          <LiveClassificationDetectionsTable detections={result.detections ?? []} />
        </>
      )}
    </div>
  );
}
