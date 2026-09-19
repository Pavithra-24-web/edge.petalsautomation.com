"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip as RTooltip, CartesianGrid, ResponsiveContainer, Legend,
} from "recharts";
import { Activity, AlertTriangle } from "lucide-react";
import type { SensorFramePayload } from "@/types/devices";

// Rolling multi-axis live chart for a device's active sensor stream —
// Motion Phase 1 Batch 4. Frames arrive one at a time (a new `frame` prop
// reference per `sensor.frame` Studio event, see MotionLabeling.tsx's
// useStudioWS handler); buffered here in a fixed-size ring and repainted at
// most once per animation frame so a fast device can't drive React
// re-renders faster than the screen can show them. Nothing here is
// persisted — the buffer lives in a ref and is discarded on unmount or
// when streaming stops.

const MAX_POINTS = 150;
// Batch 7: a stream marked `active` that hasn't delivered a frame in this
// long is flagged stale rather than left looking silently healthy — covers
// both a server-side pause (e.g. a recording just claimed the wire — see
// stream_control.pause_for_recording) that the frontend hasn't caught up to
// yet, and a genuinely stuck/lost connection.
const STALE_AFTER_MS = 4000;
const AXIS_COLORS = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#fb7185"];

const AXIS_TICK = { fontSize: 11, fill: "var(--app-text-soft)" };
const GRID_STROKE = "var(--app-border)";
const TOOLTIP_STYLE = {
  background: "var(--app-surface)",
  border: "1px solid var(--app-border)",
  borderRadius: 8,
  fontSize: 12,
  color: "var(--app-text)",
} as const;

interface ChartPoint {
  t: number;
  [axis: string]: number;
}

export default function SignalPreview({
  frame,
  active,
}: {
  /** Latest decoded `sensor.frame` payload — a new object each time a frame
   *  arrives, even with identical values, since object identity (not deep
   *  equality) is what drives the buffering effect below. */
  frame: SensorFramePayload | null;
  /** Whether a sensor stream is currently active for the connected device. */
  active: boolean;
}) {
  const bufferRef = useRef<ChartPoint[]>([]);
  const seqRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const [data, setData] = useState<ChartPoint[]>([]);
  const [seriesCount, setSeriesCount] = useState(0);
  const lastFrameAtRef = useRef<number | null>(null);
  const [stale, setStale] = useState(false);

  useEffect(() => {
    if (!frame) return;
    lastFrameAtRef.current = Date.now();
    setStale(false);
    seqRef.current += 1;
    const point: ChartPoint = { t: seqRef.current };
    frame.values.forEach((v, i) => { point[`axis${i}`] = v; });
    bufferRef.current = [...bufferRef.current, point].slice(-MAX_POINTS);
    if (frame.values.length !== seriesCount) setSeriesCount(frame.values.length);

    if (rafRef.current == null) {
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        setData(bufferRef.current);
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame]);

  useEffect(() => {
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // Stream stopped — drop the buffer so a stale waveform never lingers.
  useEffect(() => {
    if (!active) {
      bufferRef.current = [];
      seqRef.current = 0;
      lastFrameAtRef.current = null;
      setData([]);
      setStale(false);
    } else {
      // Give the first frame its own grace window rather than flagging
      // stale immediately on Start.
      lastFrameAtRef.current = Date.now();
    }
  }, [active]);

  // Poll for staleness while active — a frame arriving resets the clock
  // above; nothing arriving for STALE_AFTER_MS flips the banner on.
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => {
      const last = lastFrameAtRef.current;
      setStale(last != null && Date.now() - last >= STALE_AFTER_MS);
    }, 1000);
    return () => window.clearInterval(id);
  }, [active]);

  const series = useMemo(
    () => Array.from({ length: seriesCount }, (_, i) => ({
      key: `axis${i}`,
      name: `Axis ${i + 1}`,
      color: AXIS_COLORS[i % AXIS_COLORS.length],
    })),
    [seriesCount],
  );

  if (!active) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-center gap-1.5">
        <Activity size={18} style={{ color: "var(--app-text-soft)" }} />
        <p className="text-xs" style={{ color: "var(--app-text-muted)" }}>
          Start the sensor stream to see live values.
        </p>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-center gap-1.5">
        {stale ? (
          <>
            <AlertTriangle size={16} className="text-amber-400" />
            <p className="text-xs" style={{ color: "var(--app-text-muted)" }}>
              No data received yet — check the device connection.
            </p>
          </>
        ) : (
          <p className="text-xs" style={{ color: "var(--app-text-muted)" }}>Waiting for the first frame…</p>
        )}
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: 220, position: "relative" }}>
      {stale && (
        <div
          className="absolute top-1.5 right-1.5 z-10 flex items-center gap-1 text-[11px] font-medium px-2 py-1 rounded-full"
          style={{
            color: "#f59e0b",
            background: "color-mix(in srgb, #f59e0b 14%, var(--app-surface))",
            border: "1px solid color-mix(in srgb, #f59e0b 35%, transparent)",
          }}
        >
          <AlertTriangle size={11} /> Stream may be stale
        </div>
      )}
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid stroke={GRID_STROKE} vertical={false} />
          <XAxis dataKey="t" tick={false} height={8} axisLine={{ stroke: GRID_STROKE }} />
          <YAxis tick={AXIS_TICK} width={40} domain={["auto", "auto"]} />
          <RTooltip
            contentStyle={TOOLTIP_STYLE}
            labelFormatter={() => ""}
            formatter={(value: any, name: any) => [Number(value).toFixed(3), name]}
          />
          <Legend
            verticalAlign="top"
            align="right"
            height={24}
            wrapperStyle={{ fontSize: 11, color: "var(--app-text-soft)" }}
          />
          {series.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.name}
              stroke={s.color}
              strokeWidth={1.75}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
