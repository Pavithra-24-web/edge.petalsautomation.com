"use client";
/**
 * Anchored tour step. Highlights a target element (matched by CSS selector)
 * with a dimmed cutout and floats a TourCard beside it.
 *
 * Positioning is done with getBoundingClientRect() — no external positioning
 * library. The target is polled briefly after mount because a step may have
 * just navigated to a new route and the element might not be in the DOM yet.
 * If it never appears, we fall back to a centered card so the tour never
 * dead-ends.
 */
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import TourCard from "./TourCard";
import type { OnboardingStep } from "./steps";

const CARD_WIDTH = 340;
const GAP = 14; // space between target and card
const PAD = 6; // how far the highlight ring extends past the target
const VIEWPORT_MARGIN = 12;

// Steps that get a slightly roomier card: welcome (0), create-project (1) and
// finish (7). Every other step keeps the default width.
const WIDE_STEP_INDEXES = new Set([0, 1, 7]);
const WIDE_CARD_WIDTH = 400;
function cardWidthFor(stepIndex: number) {
  return WIDE_STEP_INDEXES.has(stepIndex) ? WIDE_CARD_WIDTH : CARD_WIDTH;
}

type Rect = { top: number; left: number; width: number; height: number };

type TourSpotlightProps = {
  step: OnboardingStep;
  stepIndex: number;
  total: number;
  isFirst: boolean;
  isLast: boolean;
  onNext: () => void;
  onBack: () => void;
  onSkip: () => void;
  onGoTo: (index: number) => void;
};

export default function TourSpotlight(props: TourSpotlightProps) {
  const { step } = props;
  const [rect, setRect] = useState<Rect | null>(null);
  const [ready, setReady] = useState(false);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const [cardHeight, setCardHeight] = useState(0);

  // Locate + measure the target, retrying briefly to survive route changes and
  // targets that mount only after a data load (e.g. the "create project" button
  // that appears once the dashboard's projects fetch resolves). We render the
  // card promptly and upgrade to the anchored position as soon as the target
  // shows up, so a data-gated step never sits blank while we poll. When the
  // target is already present it is found synchronously here and anchors in the
  // first render — no centered flash.
  useEffect(() => {
    if (!step.target) {
      setRect(null);
      setReady(true);
      return;
    }
    setRect(null);
    setReady(true);
    let raf = 0;
    let tries = 0;
    let cancelled = false;

    const measure = () => {
      if (cancelled) return;
      const el = document.querySelector(step.target as string) as HTMLElement | null;
      if (el) {
        el.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });
        const r = el.getBoundingClientRect();
        setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
        return;
      }
      tries += 1;
      if (tries > 15) {
        // ~1.5s elapsed — give up and stay centered.
        return;
      }
      raf = window.setTimeout(measure, 100) as unknown as number;
    };
    measure();
    return () => {
      cancelled = true;
      window.clearTimeout(raf);
    };
  }, [step.id, step.target]);

  // Keep the highlight glued to the target during scroll / resize.
  useEffect(() => {
    if (!step.target) return;
    const update = () => {
      const el = document.querySelector(step.target as string) as HTMLElement | null;
      if (!el) return;
      const r = el.getBoundingClientRect();
      setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
    };
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [step.target]);

  useLayoutEffect(() => {
    if (cardRef.current) setCardHeight(cardRef.current.offsetHeight);
  }, [rect, ready, step.id]);

  if (!ready) return null;

  const centered = !rect;
  const cardWidth = cardWidthFor(props.stepIndex);
  const cardStyle = centered
    ? centeredStyle(cardWidth)
    : anchoredStyle(rect as Rect, step.placement ?? "bottom", cardHeight, cardWidth);

  return createPortal(
    <div className="pe-tour-root" role="dialog" aria-modal="true">
      {/* Dimmed backdrop. When anchored, the highlight ring is drawn as a
          giant box-shadow around the cutout element below. */}
      {centered ? (
        <div className="pe-tour-backdrop" />
      ) : (
        <div
          className="pe-tour-cutout"
          style={{
            top: (rect as Rect).top - PAD,
            left: (rect as Rect).left - PAD,
            width: (rect as Rect).width + PAD * 2,
            height: (rect as Rect).height + PAD * 2,
          }}
        />
      )}

      <div ref={cardRef} className="pe-tour-popover" style={cardStyle}>
        <TourCard
          title={step.title}
          body={step.body}
          stepIndex={props.stepIndex}
          total={props.total}
          isFirst={props.isFirst}
          isLast={props.isLast}
          onNext={props.onNext}
          onBack={props.onBack}
          onSkip={props.onSkip}
          onGoTo={props.onGoTo}
        />
      </div>
    </div>,
    document.body,
  );
}

function centeredStyle(width: number): React.CSSProperties {
  return {
    position: "fixed",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width,
  };
}

function anchoredStyle(
  rect: Rect,
  placement: "top" | "bottom" | "left" | "right",
  cardHeight: number,
  width: number,
): React.CSSProperties {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  let top = 0;
  let left = 0;

  switch (placement) {
    case "right":
      left = rect.left + rect.width + GAP;
      top = rect.top + rect.height / 2 - cardHeight / 2;
      break;
    case "left":
      left = rect.left - width - GAP;
      top = rect.top + rect.height / 2 - cardHeight / 2;
      break;
    case "top":
      left = rect.left + rect.width / 2 - width / 2;
      top = rect.top - cardHeight - GAP;
      break;
    case "bottom":
    default:
      left = rect.left + rect.width / 2 - width / 2;
      top = rect.top + rect.height + GAP;
      break;
  }

  // Clamp inside the viewport.
  left = Math.max(VIEWPORT_MARGIN, Math.min(left, vw - width - VIEWPORT_MARGIN));
  const maxTop = vh - (cardHeight || 0) - VIEWPORT_MARGIN;
  top = Math.max(VIEWPORT_MARGIN, Math.min(top, Math.max(VIEWPORT_MARGIN, maxTop)));

  return { position: "fixed", top, left, width };
}
