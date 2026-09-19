"use client";

interface AccuracySummaryProps {
  accuracy: number;
}

export default function AccuracySummary({ accuracy }: AccuracySummaryProps) {
  return (
    <div className="flex items-center gap-4 py-3">
      <div
        className="w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0"
        style={{
          background: "linear-gradient(135deg, #7c3aed 0%, #6366f1 55%, #818cf8 100%)",
          boxShadow:
            "0 1px 0 rgba(255,255,255,0.3) inset, 0 10px 22px -10px rgba(124,58,237,0.5)",
        }}
      >
        <span className="text-white font-bold text-lg leading-none">%</span>
      </div>

      <div>
        <div className="mb-0.5">
          <span className="text-[10px] font-bold uppercase tracking-widest text-[color:var(--app-text-muted)]">
            Accuracy
          </span>
        </div>
        <p className="text-2xl font-bold leading-tight text-[color:var(--app-text)]">
          {accuracy.toFixed(2)}%
        </p>
      </div>
    </div>
  );
}
