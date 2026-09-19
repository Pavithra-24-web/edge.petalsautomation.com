"use client";
import { useEffect, useState, useRef } from "react";
import { XCircle, Hand, MousePointer2, Undo, Redo, ZoomIn, ZoomOut, Lock } from "lucide-react";
import { Stage, Layer, Image as KonvaImage, Rect, Transformer, Text, Group } from 'react-konva';
import { v4 as uuidv4 } from 'uuid';

// ─── Shared Annotation Editor ─────────────────────────────────────────────────
// Single Konva editor used by both Data Labeling and AI Labeling. Owns its own
// tool/zoom/pan/history/modal state; the parent supplies the image, initial
// boxes, label list, and persistence callbacks.

// Last label the user committed, remembered so the next box's prompt opens
// prefilled and can be accepted with Enter. Module scope on purpose: the grid
// mounts one editor per card and the preview remounts on sample change, so
// component state would forget it between boxes.
let lastUsedLabelName = "";

type AnnotationEditorProps = {
  image: any;
  initialBoxes: any[];
  resetKey: any;
  labels: any[];
  onChange: (boxes: any[], primary?: { id?: string; name?: string } | null) => void | Promise<void>;
  onCreateLabel: (name: string) => Promise<{ id: string; name: string } | null>;
  colorForLabel: (label: string) => string;
  readOnly?: boolean;
};

function normalizeBoxLabelStr(value: any) {
  const text = String(value ?? "").trim();
  if (!text) return "";
  const lowered = text.toLowerCase();
  if (lowered === "unlabeled" || lowered === "unlabelled" || lowered === "unknown") return "";
  return text;
}

function normalizeBoxGeometryShape(b: any): any {
  if (!b || typeof b !== "object") return b;
  const out = { ...b };
  if (out.w === undefined && out.width !== undefined) { out.w = out.width; delete out.width; }
  if (out.h === undefined && out.height !== undefined) { out.h = out.height; delete out.height; }
  return out;
}

export function AnnotationEditor({ image, initialBoxes, resetKey, labels, onChange, onCreateLabel, colorForLabel, readOnly = false }: AnnotationEditorProps) {
  const [history, setHistory] = useState<any[][]>([[]]);
  const [historyStep, setHistoryStep] = useState(0);
  const [tool, setTool] = useState<"draw" | "pan">("draw");
  const [zoom, setZoom] = useState(1);
  const [baseScale, setBaseScale] = useState(1);
  const [stagePos, setStagePos] = useState({ x: 0, y: 0 });
  const [drawing, setDrawing] = useState(false);
  const [currentBox, setCurrentBox] = useState<any>(null);
  // Fixed anchor (drag origin) in image coords. Kept separate from currentBox,
  // whose x/y move to the top-left as the pointer crosses the anchor — so the
  // box can be drawn from any corner in any direction.
  const drawStartRef = useRef<{ x: number; y: number } | null>(null);
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null);
  const [stageSize, setStageSize] = useState({ width: 800, height: 600 });
  const [modalOpen, setModalOpen] = useState(false);
  const [pendingBox, setPendingBox] = useState<any>(null);
  const [modalLabel, setModalLabel] = useState("");
  const trRef = useRef<any>(null);
  const stageContainerRef = useRef<HTMLDivElement>(null);

  // Re-init history when the editing target changes
  useEffect(() => {
    const init = (initialBoxes || []).map((b: any) => ({
      ...normalizeBoxGeometryShape(b),
      label: normalizeBoxLabelStr(b?.label ?? b?.className),
      id: b.id || uuidv4(),
    }));
    setHistory([init]);
    setHistoryStep(0);
    setSelectedBoxId(null);
    setTool(readOnly ? "pan" : "draw");
  }, [resetKey]);

  // Measure canvas
  useEffect(() => {
    const el = stageContainerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setStageSize({ width: el.clientWidth, height: el.clientHeight });
    });
    ro.observe(el);
    setStageSize({ width: el.clientWidth, height: el.clientHeight });
    return () => ro.disconnect();
  }, [resetKey]);

  // Fit image
  useEffect(() => {
    if (image && stageSize.width > 0 && stageSize.height > 0) {
      const scale = Math.min(stageSize.width / image.width, stageSize.height / image.height) * 0.90;
      setBaseScale(scale);
      setZoom(scale);
      setStagePos({
        x: (stageSize.width - image.width * scale) / 2,
        y: (stageSize.height - image.height * scale) / 2,
      });
    }
  }, [image, stageSize]);

  function pushHistoryState(newBoxes: any[]) {
    const newHist = history.slice(0, historyStep + 1);
    newHist.push(newBoxes);
    setHistory(newHist);
    setHistoryStep(newHist.length - 1);
  }

  function handleWheel(e: any) {
    e.evt.preventDefault();
    const scaleBy = 1.1;
    const stage = e.target.getStage();
    const oldScale = zoom;
    const pointer = stage.getPointerPosition();
    const mousePointTo = {
      x: (pointer.x - stagePos.x) / oldScale,
      y: (pointer.y - stagePos.y) / oldScale,
    };
    const newScale = e.evt.deltaY > 0 ? oldScale / scaleBy : oldScale * scaleBy;
    setZoom(newScale);
    setStagePos({ x: pointer.x - mousePointTo.x * newScale, y: pointer.y - mousePointTo.y * newScale });
  }

  function zoomBy(factor: number) {
    const newScale = Math.min(Math.max(zoom * factor, 0.1), 10);
    const cx = stageSize.width / 2;
    const cy = stageSize.height / 2;
    const pointTo = { x: (cx - stagePos.x) / zoom, y: (cy - stagePos.y) / zoom };
    setZoom(newScale);
    setStagePos({ x: cx - pointTo.x * newScale, y: cy - pointTo.y * newScale });
  }

  function handleStageDragEnd(e: any) {
    if (e.target === e.target.getStage()) setStagePos({ x: e.target.x(), y: e.target.y() });
  }

  function handleMouseDown(e: any) {
    const clickedOnEmpty = e.target === e.target.getStage() || e.target.name() === "image_bg";
    if (clickedOnEmpty) setSelectedBoxId(null);
    if (readOnly || tool !== "draw") return;
    // Allow starting a new box on top of an already-labeled region so nested /
    // overlapping labels can be drawn. The only exemption is the currently
    // selected box: a mousedown on it begins a drag/resize of that box (its Rect
    // is draggable when selected), so we must not also start a new draw there.
    const onSelectedBox =
      !clickedOnEmpty && selectedBoxId != null &&
      typeof e.target.id === "function" && e.target.id() === selectedBoxId;
    if (onSelectedBox) return;
    const stage = e.target.getStage();
    const point = stage.getRelativePointerPosition();
    setDrawing(true);
    drawStartRef.current = { x: point.x, y: point.y };
    setCurrentBox({ x: point.x, y: point.y, w: 0, h: 0 });
  }

  function handleMouseMove(e: any) {
    if (readOnly || !drawing || !drawStartRef.current || tool !== "draw") return;
    const stage = e.target.getStage();
    const point = stage.getRelativePointerPosition();
    const start = drawStartRef.current;
    // Normalize against the fixed anchor so dragging in any direction yields a
    // top-left origin with positive width/height.
    setCurrentBox({
      x: Math.min(start.x, point.x),
      y: Math.min(start.y, point.y),
      w: Math.abs(point.x - start.x),
      h: Math.abs(point.y - start.y),
    });
  }

  function handleMouseUp() {
    if (readOnly || !drawing) return;
    setDrawing(false);
    drawStartRef.current = null;
    if (currentBox && currentBox.w > 10 && currentBox.h > 10) {
      setPendingBox(currentBox);
      // Prefill with the last committed label; the input selects itself on
      // focus so Enter accepts it and typing replaces it.
      setModalLabel(lastUsedLabelName);
      setModalOpen(true);
    } else {
      setCurrentBox(null);
    }
  }

  // Abandon an in-progress draw without committing it. Runs when the pointer
  // leaves the stage or the window loses focus mid-drag, so a mouse-up that
  // happens outside the canvas/browser can't leave the editor stuck "drawing".
  function cancelDrawing() {
    drawStartRef.current = null;
    setDrawing(false);
    setCurrentBox(null);
  }

  // The window may lose focus while a drag is in flight (alt-tab, release
  // outside the browser). Clear the gesture so a stray return doesn't resume it.
  useEffect(() => {
    window.addEventListener("blur", cancelDrawing);
    return () => window.removeEventListener("blur", cancelDrawing);
  }, []);

  async function handleSetLabel() {
    if (readOnly || !pendingBox || !modalLabel.trim()) return;
    let finalLabelId = "";
    let finalLabelName = modalLabel.trim();
    const existing = labels.find(l => l.name.toLowerCase() === finalLabelName.toLowerCase());
    if (existing) {
      finalLabelId = existing.id;
      finalLabelName = existing.name;
    } else {
      const created = await onCreateLabel(finalLabelName);
      if (!created) return;
      finalLabelId = created.id;
      finalLabelName = created.name;
    }
    lastUsedLabelName = finalLabelName;
    const newBox = { ...pendingBox, label: finalLabelName, label_id: finalLabelId, id: pendingBox.id || uuidv4() };
    const currBoxes = history[historyStep] || [];
    const newBoxes = pendingBox.id ? currBoxes.map(b => b.id === pendingBox.id ? newBox : b) : [...currBoxes, newBox];
    pushHistoryState(newBoxes);
    await onChange(newBoxes, { id: finalLabelId, name: finalLabelName });
    setModalOpen(false);
    setPendingBox(null);
    setCurrentBox(null);
  }

  function handleTransformEnd(e: any, boxId: string) {
    if (readOnly) return;
    const boxes = history[historyStep];
    if (!boxes) return;
    const idx = boxes.findIndex((b: any) => b.id === boxId);
    if (idx === -1) return;
    const node = e.target;
    const scaleX = node.scaleX();
    const scaleY = node.scaleY();
    node.scaleX(1); node.scaleY(1);
    const updated = {
      ...boxes[idx],
      x: node.x(), y: node.y(),
      w: Math.max(5, node.width() * scaleX),
      h: Math.max(5, node.height() * scaleY),
    };
    const newBoxes = [...boxes]; newBoxes[idx] = updated;
    pushHistoryState(newBoxes);
    onChange(newBoxes);
  }

  function handleDragEnd(e: any, boxId: string) {
    if (readOnly) return;
    const boxes = history[historyStep];
    if (!boxes) return;
    const idx = boxes.findIndex((b: any) => b.id === boxId);
    if (idx === -1) return;
    const updated = { ...boxes[idx], x: e.target.x(), y: e.target.y() };
    const newBoxes = [...boxes]; newBoxes[idx] = updated;
    pushHistoryState(newBoxes);
    onChange(newBoxes);
  }

  function deleteSelectedBox() {
    if (readOnly || !selectedBoxId || !history[historyStep]) return;
    const newBoxes = history[historyStep].filter(b => b.id !== selectedBoxId);
    pushHistoryState(newBoxes);
    onChange(newBoxes);
    setSelectedBoxId(null);
  }

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.key === "Delete" || e.key === "Backspace") && selectedBoxId && !modalOpen) {
        deleteSelectedBox();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedBoxId, historyStep, history, modalOpen]);

  useEffect(() => {
    if (!trRef.current) return;
    const stage = trRef.current.getStage();
    if (selectedBoxId && stage) {
      const node = stage.findOne(`#${selectedBoxId}`);
      if (node) trRef.current.nodes([node]);
      else trRef.current.nodes([]);
    } else {
      trRef.current.nodes([]);
    }
    trRef.current.getLayer()?.batchDraw();
  }, [selectedBoxId, historyStep, zoom]);

  const handleUndo = () => !readOnly && historyStep > 0 && setHistoryStep(historyStep - 1);
  const handleRedo = () => !readOnly && historyStep < history.length - 1 && setHistoryStep(historyStep + 1);

  return (
    <>
      <div
        className="data-acq-editor-canvas flex-1 min-h-0 relative overflow-hidden"
        style={{ cursor: tool === "draw" ? "crosshair" : "grab" }}
        ref={stageContainerRef}
      >
        <Stage
          width={stageSize.width}
          height={stageSize.height}
          x={stagePos.x}
          y={stagePos.y}
          scaleX={zoom}
          scaleY={zoom}
          draggable={tool === "pan"}
          onDragEnd={handleStageDragEnd}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={cancelDrawing}
        >
          <Layer>
            <KonvaImage image={image} name="image_bg" />
            {(history[historyStep] || []).map((b: any, i: number) => {
              const confStr = b.confidence ? ` ${(b.confidence * 100).toFixed(0)}%` : "";
              const labelText = `${b.label || "object"}${confStr}`;
              // AI predictions keep confidence-based coloring (green/yellow/red);
              // manually-labeled boxes get a distinct, stable per-class color from
              // the shared palette so two classes render in two colors and the same
              // class always keeps its color.
              const boxColor = b.confidence !== undefined
                ? (b.confidence >= 0.8 ? "#22c55e" : b.confidence >= 0.5 ? "#eab308" : "#ef4444")
                : colorForLabel(b.label || "");
              const isSelected = selectedBoxId === b.id;
              const textPx = labelText.length * 7;
              const deletePx = 18;
              const badgeW = (isSelected ? textPx + deletePx + 2 : textPx) / zoom;
              return (
                <Group
                  key={b.id || i}
                  onClick={() => { if (!readOnly && tool === "draw") setSelectedBoxId(b.id); }}
                  onDblClick={() => {
                    if (!readOnly && tool === "draw" && b.id) {
                      setPendingBox(b);
                      setModalLabel(b.label || "");
                      setModalOpen(true);
                    }
                  }}
                >
                  <Rect
                    id={b.id}
                    x={b.x} y={b.y} width={b.w ?? b.width} height={b.h ?? b.height}
                    stroke={isSelected ? "#ffffff" : boxColor}
                    strokeWidth={2 / zoom}
                    fill={`${boxColor}20`}
                    draggable={!readOnly && tool === "draw" && isSelected}
                    onDragEnd={(e) => handleDragEnd(e, b.id)}
                    onTransformEnd={(e) => handleTransformEnd(e, b.id)}
                  />
                  <Group x={b.x} y={b.y - (20 / zoom)}>
                    <Rect fill={boxColor} height={18 / zoom} width={badgeW} cornerRadius={2 / zoom} />
                    <Text text={labelText} fill="white" fontSize={11 / zoom} padding={3 / zoom} />
                    {isSelected && (
                      <Group
                        x={(textPx + 2) / zoom}
                        onClick={(e: any) => { e.cancelBubble = true; deleteSelectedBox(); }}
                      >
                        <Rect width={deletePx / zoom} height={18 / zoom} fill="rgba(0,0,0,0.3)" cornerRadius={2 / zoom} />
                        <Text text="✕" fill="white" fontSize={11 / zoom} x={3 / zoom} y={1 / zoom} />
                      </Group>
                    )}
                  </Group>
                </Group>
              );
            })}
            {drawing && currentBox && (
              <Rect
                x={currentBox.x} y={currentBox.y}
                width={currentBox.w} height={currentBox.h}
                stroke="white" dash={[4 / zoom, 2 / zoom]} strokeWidth={1.5 / zoom}
                fill="#ffffff20"
              />
            )}
            <Transformer ref={trRef} keepRatio={false} flipEnabled={false}
              boundBoxFunc={(oldBox, newBox) => newBox.width < 5 || newBox.height < 5 ? oldBox : newBox}
            />
          </Layer>
        </Stage>

        <div className="absolute bottom-5 left-0 w-full flex justify-center pointer-events-none z-10">
          <div className="data-acq-toolbar pointer-events-auto">
            {readOnly && (
              <>
                <span className="data-acq-toolbar-locked flex items-center gap-1.5 px-2 text-[11px] text-green-400">
                  <Lock size={13} /> Applied — locked
                </span>
                <div className="data-acq-toolbar-divider" />
              </>
            )}
            <button title="Select / Draw (S)" disabled={readOnly} onClick={() => { setTool("draw"); setSelectedBoxId(null); }}
              className={`data-acq-toolbar-btn ${tool === "draw" ? "is-active" : ""}`}>
              <MousePointer2 size={15} />
            </button>
            <button title="Pan / Hand (H)" onClick={() => { setTool("pan"); setSelectedBoxId(null); }}
              className={`data-acq-toolbar-btn ${tool === "pan" ? "is-active" : ""}`}>
              <Hand size={15} />
            </button>
            <div className="data-acq-toolbar-divider" />
            <button title="Undo (Ctrl+Z)" disabled={readOnly || historyStep <= 0} onClick={handleUndo}
              className="data-acq-toolbar-btn">
              <Undo size={15} />
            </button>
            <button title="Redo (Ctrl+Shift+Z)" disabled={readOnly || historyStep >= history.length - 1} onClick={handleRedo}
              className="data-acq-toolbar-btn">
              <Redo size={15} />
            </button>
            <div className="data-acq-toolbar-divider" />
            <button title="Zoom out" onClick={() => zoomBy(1 / 1.25)} className="data-acq-toolbar-btn">
              <ZoomOut size={15} />
            </button>
            <span className="data-acq-toolbar-zoom">
              {Math.round((zoom / baseScale) * 100)}%
            </span>
            <button title="Zoom in" onClick={() => zoomBy(1.25)} className="data-acq-toolbar-btn">
              <ZoomIn size={15} />
            </button>
          </div>
        </div>
      </div>

      {modalOpen && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-sm overflow-hidden shadow-2xl animate-fade-in">
            <div className="px-5 py-4 border-b border-gray-800 flex justify-between items-center">
              <h3 className="font-bold text-gray-100">Enter label</h3>
              <button className="text-gray-500 hover:text-gray-200 transition-colors"
                onClick={() => { setModalOpen(false); setPendingBox(null); }}>
                <XCircle size={16} />
              </button>
            </div>
            <div className="p-5">
              <input type="text" autoFocus className="input w-full" placeholder="e.g. coffee_mug"
                value={modalLabel} onChange={e => setModalLabel(e.target.value)}
                onFocus={e => e.currentTarget.select()}
                onKeyDown={e => { if (e.key === 'Enter') handleSetLabel(); }}
              />
            </div>
            <div className="px-5 py-4 bg-gray-950/50 border-t border-gray-800 flex justify-end gap-3">
              <button onClick={() => { setModalOpen(false); setPendingBox(null); }} className="btn-ghost">
                Cancel
              </button>
              <button onClick={handleSetLabel} className="btn-primary" disabled={!modalLabel.trim()}>
                Set label
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
