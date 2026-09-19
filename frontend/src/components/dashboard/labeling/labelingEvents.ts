/**
 * Mutation bus for sample changes that affect the Dataset page's live
 * unlabeled counter (`useUnlabeledCount`). Mirrors the `data-labeling:go-home`
 * event precedent in Sidebar.tsx / ObjectDetectionLabeling.tsx.
 */
export const SAMPLES_CHANGED = "labeling:samples-changed";

export const notifySamplesChanged = () =>
  window.dispatchEvent(new Event(SAMPLES_CHANGED));
