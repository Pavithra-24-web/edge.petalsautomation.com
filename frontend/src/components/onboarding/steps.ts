/**
 * Declarative config for the first-time-user onboarding tour.
 *
 * Steps are shown in array order. Two kinds:
 *  - "modal": a centered card with no anchor (welcome / finish).
 *  - "spotlight": highlights a `target` element (matched by CSS selector) and
 *    shows a tooltip next to it. If the target is not on the page, the tooltip
 *    falls back to a centered card so the tour never dead-ends.
 *
 * `route` (optional): the tour navigates here before showing the step. We only
 * navigate to pages that are safe for a brand-new account (the dashboard). All
 * other steps anchor to the always-present sidebar nav, so we never force a
 * user into a data-gated page (training/deploy) that would show an empty state.
 */
export type OnboardingStep = {
  id: string;
  kind: "modal" | "spotlight";
  title: string;
  body: string;
  /** CSS selector of the element to highlight (spotlight steps only). */
  target?: string;
  /** Route to push before showing this step. */
  route?: string;
  /** Preferred tooltip side relative to the target. */
  placement?: "top" | "bottom" | "left" | "right";
};

export const ONBOARDING_STEPS: OnboardingStep[] = [
  {
    id: "welcome",
    kind: "modal",
    title: "Welcome to PetalEdge 🌸",
    body: "Let's take a quick tour of how you go from raw data to a model deployed on an edge device. It takes about a minute — you can skip anytime.",
    route: "/dashboard",
  },
  {
    id: "create-project",
    kind: "spotlight",
    title: "1. Create a project",
    body: "Everything starts with a project — it holds your data, impulses and trained models. Create your first one to get going.",
    target: '[data-tour="create-project"]',
    route: "/dashboard",
    placement: "bottom",
  },
  {
    id: "data-acquisition",
    kind: "spotlight",
    title: "2. Acquire data",
    body: "Upload or capture the training samples your model will learn from. This is where every dataset begins.",
    target: '[data-tour="nav-data"]',
    placement: "right",
  },
  {
    id: "data-labeling",
    kind: "spotlight",
    title: "3. Label your data",
    body: "Draw bounding boxes or assign classes so each sample carries a ground-truth label the model can train against.",
    target: '[data-tour="nav-labeling"]',
    placement: "right",
  },
  {
    id: "create-impulse",
    kind: "spotlight",
    title: "4. Design an impulse & train",
    body: "An impulse is your ML pipeline: processing (DSP) blocks plus a learning block. Add blocks, generate features, then train the model right here.",
    target: '[data-tour="nav-create-impulse"]',
    placement: "right",
  },
  {
    id: "model-testing",
    kind: "spotlight",
    title: "5. Test & evaluate",
    body: "Check how your trained model performs on held-out test data before you ship it.",
    target: '[data-tour="nav-testing"]',
    placement: "right",
  },
  {
    id: "deployment",
    kind: "spotlight",
    title: "6. Deploy to the edge",
    body: "Package your model for Arduino, Raspberry Pi and more, then flash it to a connected device.",
    target: '[data-tour="nav-deployment"]',
    placement: "right",
  },
  {
    id: "finish",
    kind: "modal",
    title: "You're all set! 🚀",
    body: "That's the full workflow: data → labeling → impulse → training → testing → deployment. Start by creating a project. You can replay this tour anytime from the account menu.",
    route: "/dashboard",
  },
];
