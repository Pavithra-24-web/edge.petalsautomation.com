"use client";
import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react';

interface Point {
  x: number;
  y: number;
  label: string;
  sample_id?: string;
}

interface FeatureExplorerProps {
  points: Point[];
  className?: string;
  onPointClick?: (point: Point) => void;
  selectedPointId?: string | null;
}

const COLORS = [
  '#6366f1', '#10b981', '#f59e0b', '#ef4444',
  '#ec4899', '#8b5cf6', '#06b6d4', '#f97316'
];

// Map a value from [inMin, inMax] to [outMin, outMax]
function mapRange(value: number, inMin: number, inMax: number, outMin: number, outMax: number) {
  if (inMin === inMax) return (outMin + outMax) / 2;
  return ((value - inMin) / (inMax - inMin)) * (outMax - outMin) + outMin;
}

export default function FeatureExplorer({ points, className, onPointClick, selectedPointId }: FeatureExplorerProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ w: 300, h: 240 });
  const [tooltip, setTooltip] = useState<{ visible: boolean; x: number; y: number; point: Point | null }>({
    visible: false, x: 0, y: 0, point: null,
  });

  // Measure container
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        setDims({ w: entry.contentRect.width, h: entry.contentRect.height });
      }
    });
    ro.observe(el);
    setDims({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  const PADDING = { top: 20, right: 20, bottom: 20, left: 20 };
  const plotW = dims.w - PADDING.left - PADDING.right;
  const plotH = dims.h - PADDING.top - PADDING.bottom;

  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    if (!points.length) return { xMin: -1, xMax: 1, yMin: -1, yMax: 1 };
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const xPad = (Math.max(...xs) - Math.min(...xs)) * 0.08 || 0.5;
    const yPad = (Math.max(...ys) - Math.min(...ys)) * 0.08 || 0.5;
    return {
      xMin: Math.min(...xs) - xPad,
      xMax: Math.max(...xs) + xPad,
      yMin: Math.min(...ys) - yPad,
      yMax: Math.max(...ys) + yPad,
    };
  }, [points]);

  const labelColorMap = useMemo(() => {
    const map: Record<string, string> = {};
    const labels = [...new Set(points.map(p => p.label))];
    labels.forEach((l, i) => { map[l] = COLORS[i % COLORS.length]; });
    return map;
  }, [points]);

  const projected = useMemo(() => points.map(p => ({
    ...p,
    cx: PADDING.left + mapRange(p.x, xMin, xMax, 0, plotW),
    cy: PADDING.top + mapRange(p.y, yMin, yMax, plotH, 0),
    color: labelColorMap[p.label] || '#6366f1',
  })), [points, xMin, xMax, yMin, yMax, plotW, plotH, labelColorMap]);

  const handlePointClick = useCallback((e: React.MouseEvent, point: Point) => {
    e.preventDefault();
    e.stopPropagation();
    onPointClick?.(point);
  }, [onPointClick]);

  const handleMouseEnter = useCallback((e: React.MouseEvent, point: Point & { cx: number; cy: number }) => {
    setTooltip({ visible: true, x: point.cx, y: point.cy, point });
  }, []);

  const handleMouseLeave = useCallback(() => {
    setTooltip(t => ({ ...t, visible: false }));
  }, []);

  if (!points || points.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500">
        <p className="text-sm">No data to visualize</p>
      </div>
    );
  }

  const labels = Object.keys(labelColorMap);

  return (
    <div ref={containerRef} className={`w-full h-full flex flex-col select-none ${className ?? ''}`}>
      {/* SVG Chart - takes all available space */}
      <div className="flex-1 min-h-0 relative">
        <svg
          ref={svgRef}
          width={dims.w}
          height={dims.h}
          className="w-full h-full"
          style={{ display: 'block' }}
        >
          {/* Grid lines (subtle) */}
          {[0.25, 0.5, 0.75].map(t => (
            <g key={t}>
              <line
                x1={PADDING.left + plotW * t} y1={PADDING.top}
                x2={PADDING.left + plotW * t} y2={PADDING.top + plotH}
                stroke="#374151" strokeWidth={0.5} strokeDasharray="3 4"
              />
              <line
                x1={PADDING.left} y1={PADDING.top + plotH * t}
                x2={PADDING.left + plotW} y2={PADDING.top + plotH * t}
                stroke="#374151" strokeWidth={0.5} strokeDasharray="3 4"
              />
            </g>
          ))}

          {/* Data Points — rendered as native SVG circles with direct click handlers */}
          {projected.map((p, idx) => {
            const isSelected = p.sample_id === selectedPointId;
            const isDimmed = selectedPointId && !isSelected;
            return (
              <circle
                key={p.sample_id ?? idx}
                cx={p.cx}
                cy={p.cy}
                r={isSelected ? 8 : 5}
                fill={p.color}
                stroke={isSelected ? '#ffffff' : 'transparent'}
                strokeWidth={isSelected ? 2.5 : 0}
                opacity={isDimmed ? 0.15 : 0.85}
                style={{ cursor: 'pointer', transition: 'opacity 0.2s, r 0.15s' }}
                onClick={(e) => handlePointClick(e, p)}
                onMouseEnter={(e) => handleMouseEnter(e, p)}
                onMouseLeave={handleMouseLeave}
              />
            );
          })}

          {/* Tooltip tick */}
          {tooltip.visible && tooltip.point && (
            <circle
              cx={(projected.find(p => p.sample_id === tooltip.point?.sample_id))?.cx ?? 0}
              cy={(projected.find(p => p.sample_id === tooltip.point?.sample_id))?.cy ?? 0}
              r={10}
              fill="none"
              stroke="#fff"
              strokeWidth={1}
              opacity={0.4}
              style={{ pointerEvents: 'none' }}
            />
          )}
        </svg>

        {/* Floating Tooltip */}
        {tooltip.visible && tooltip.point && (() => {
          const proj = projected.find(p => p.sample_id === tooltip.point?.sample_id);
          if (!proj) return null;
          const left = proj.cx + 14;
          const top = proj.cy - 20;
          return (
            <div
              style={{ position: 'absolute', left, top, pointerEvents: 'none', zIndex: 50 }}
              className="bg-gray-900 border border-gray-700 rounded-lg px-2.5 py-1.5 shadow-xl text-[10px] whitespace-nowrap"
            >
              <p className="font-bold text-white capitalize">{tooltip.point.label}</p>
              <p className="text-gray-400 mt-0.5">{tooltip.point.sample_id?.slice(0, 8)}…</p>
            </div>
          );
        })()}
      </div>

      {/* Legend */}
      <div className="feature-explorer-legend pt-2 flex flex-wrap gap-x-4 gap-y-1 px-4 pb-2 border-t border-gray-800/50 bg-gray-900/20 flex-shrink-0">
        {labels.map(label => (
          <div key={label} className="flex items-center gap-1.5">
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: labelColorMap[label] }}
            />
            <span className="text-[10px] font-medium text-gray-400 truncate max-w-[80px]">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
