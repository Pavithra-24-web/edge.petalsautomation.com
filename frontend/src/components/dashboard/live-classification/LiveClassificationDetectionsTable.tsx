import type { LiveClassificationDetection } from "@/types/live-classification";
import { fmtPct } from "@/lib/live-classification-helpers";

interface Props {
  detections: LiveClassificationDetection[];
}

export default function LiveClassificationDetectionsTable({ detections }: Props) {
  if (detections.length === 0) {
    return (
      <p className="text-xs text-gray-500 py-2">No objects detected above threshold.</p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-gray-800">
            <th className="table-header text-left py-2 pr-3">Label</th>
            <th className="table-header text-right py-2 pr-3">Score</th>
            <th className="table-header text-right py-2 pr-3">x1</th>
            <th className="table-header text-right py-2 pr-3">y1</th>
            <th className="table-header text-right py-2 pr-3">x2</th>
            <th className="table-header text-right py-2">y2</th>
          </tr>
        </thead>
        <tbody>
          {detections.map((det, i) => (
            <tr key={i} className="border-b border-gray-800/50 last:border-0">
              <td className="table-cell pl-0 font-medium text-brand-400">{det.label}</td>
              <td className="table-cell text-right pr-3">{fmtPct(det.confidence)}</td>
              <td className="table-cell text-right pr-3 text-gray-500">
                {det.bbox.x1.toFixed(3)}
              </td>
              <td className="table-cell text-right pr-3 text-gray-500">
                {det.bbox.y1.toFixed(3)}
              </td>
              <td className="table-cell text-right pr-3 text-gray-500">
                {det.bbox.x2.toFixed(3)}
              </td>
              <td className="table-cell text-right text-gray-500">
                {det.bbox.y2.toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
