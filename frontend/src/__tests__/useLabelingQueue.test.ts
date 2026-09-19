/**
 * Reducer tests for the Labeling Queue controller (Phase 4 §4.B/§4.C,
 * validated here per Phase 6 §6.A #4-#7/#9/#11 and Phase 4.F).
 *
 * Exercises `queueReducer` directly (same pattern as devicesReducer.test.ts)
 * rather than mounting `useLabelingQueue`, since the hook's snapshot-fetch
 * side effects (samplesApi + next/navigation) are already covered by the
 * backend ordering/dedup tests and by manual QA (Phase 6.B).
 */
import { queueReducer, initialState, firstRemainingAfter } from "@/hooks/useLabelingQueue";
import type { QueueItem, QueueState } from "@/hooks/useLabelingQueue";

function makeItem(overrides: Partial<QueueItem>): QueueItem {
  return {
    id: "id-1",
    filename: "IMG_0001.jpg",
    sample_type: "training",
    created_at: "2026-01-01T00:00:00Z",
    status: "remaining",
    ...overrides,
  };
}

function itemsOf(n: number, statuses?: QueueItem["status"][]): QueueItem[] {
  return Array.from({ length: n }, (_, i) =>
    makeItem({ id: `id-${i}`, filename: `IMG_${i}.jpg`, status: statuses?.[i] ?? "remaining" }),
  );
}

describe("firstRemainingAfter", () => {
  it("finds the next remaining item after the current index", () => {
    const items = itemsOf(4, ["completed", "remaining", "completed", "remaining"]);
    expect(firstRemainingAfter(items, 0)).toBe(1);
  });

  it("wraps once to the start when nothing remains after the index", () => {
    const items = itemsOf(4, ["remaining", "completed", "completed", "completed"]);
    expect(firstRemainingAfter(items, 3)).toBe(0);
  });

  it("returns -1 when nothing is remaining anywhere", () => {
    const items = itemsOf(3, ["completed", "completed", "completed"]);
    expect(firstRemainingAfter(items, 0)).toBe(-1);
  });
});

describe("queueReducer — build / snapshot", () => {
  it("LOAD_SNAPSHOT with no resumeId starts at index 0", () => {
    const items = itemsOf(3);
    const next = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items,
      resumeId: null,
      totalSamplesInProject: 10,
    });
    expect(next.status).toBe("ready");
    expect(next.index).toBe(0);
    expect(next.items).toHaveLength(3);
    expect(next.totalSamplesInProject).toBe(10);
  });

  it("LOAD_SNAPSHOT resumes at the matching item's index", () => {
    const items = itemsOf(5);
    const next = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items,
      resumeId: "id-3",
      totalSamplesInProject: 5,
    });
    expect(next.index).toBe(3);
  });

  it("LOAD_SNAPSHOT falls back to index 0 when the resume id no longer exists", () => {
    const items = itemsOf(3);
    const next = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items,
      resumeId: "id-does-not-exist",
      totalSamplesInProject: 3,
    });
    expect(next.index).toBe(0);
  });

  it("LOAD_SNAPSHOT on an empty queue is a valid ready state (nothing-to-label)", () => {
    const next = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: [],
      resumeId: null,
      totalSamplesInProject: 40,
    });
    expect(next.status).toBe("ready");
    expect(next.items).toHaveLength(0);
    expect(next.index).toBe(0);
  });

  it("LOAD_ERROR sets status and error message", () => {
    const next = queueReducer(initialState, { type: "LOAD_ERROR", error: "boom" });
    expect(next.status).toBe("error");
    expect(next.error).toBe("boom");
  });
});

describe("queueReducer — MARK_SAVED advance", () => {
  it("marks the item completed and advances to the next remaining item", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(4),
      resumeId: null,
      totalSamplesInProject: 4,
    });
    state = queueReducer(state, { type: "MARK_SAVED", id: "id-0" });
    expect(state.items[0].status).toBe("completed");
    expect(state.index).toBe(1);
  });

  it("wraps once to the start when the remaining work is behind the current pointer", () => {
    // Completed: 0,1 ; current at 2 (remaining) ; remaining also at 3... use a
    // case where only an earlier item remains after marking the last one saved.
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(3, ["remaining", "completed", "remaining"]),
      resumeId: "id-2",
      totalSamplesInProject: 3,
    });
    expect(state.index).toBe(2);
    state = queueReducer(state, { type: "MARK_SAVED", id: "id-2" });
    // item 2 saved -> only id-0 remains -> wraps to index 0
    expect(state.items[2].status).toBe("completed");
    expect(state.index).toBe(0);
  });

  it("stays put when it is the last remaining item (session complete)", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(3, ["completed", "completed", "remaining"]),
      resumeId: "id-2",
      totalSamplesInProject: 3,
    });
    state = queueReducer(state, { type: "MARK_SAVED", id: "id-2" });
    expect(state.items.every(i => i.status === "completed")).toBe(true);
    // no remaining item to jump to -> index does not change
    expect(state.index).toBe(2);
  });

  it("does not reshuffle other entries' positions (in-place update only, §2.3)", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(4),
      resumeId: null,
      totalSamplesInProject: 4,
    });
    const idsBefore = state.items.map(i => i.id);
    state = queueReducer(state, { type: "MARK_SAVED", id: "id-1" });
    expect(state.items.map(i => i.id)).toEqual(idsBefore);
  });
});

describe("queueReducer — navigation", () => {
  it("NEXT/PREV move the pointer over all items, including completed ones", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(3, ["completed", "remaining", "remaining"]),
      resumeId: null,
      totalSamplesInProject: 3,
    });
    state = queueReducer(state, { type: "NEXT" });
    expect(state.index).toBe(1);
    state = queueReducer(state, { type: "PREV" });
    expect(state.index).toBe(0); // back onto the completed item — reopenable
    expect(state.items[0].status).toBe("completed");
  });

  it("NEXT is a no-op at the last item; PREV is a no-op at the first", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(2),
      resumeId: null,
      totalSamplesInProject: 2,
    });
    state = queueReducer(state, { type: "PREV" });
    expect(state.index).toBe(0);
    state = queueReducer(state, { type: "NEXT" });
    state = queueReducer(state, { type: "NEXT" });
    expect(state.index).toBe(1);
  });

  it("GOTO ignores out-of-range indices", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(2),
      resumeId: null,
      totalSamplesInProject: 2,
    });
    const before = state.index;
    state = queueReducer(state, { type: "GOTO", index: 99 });
    expect(state.index).toBe(before);
    state = queueReducer(state, { type: "GOTO", index: -1 });
    expect(state.index).toBe(before);
  });

  it("a single-item queue keeps NEXT/PREV as safe no-ops", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(1),
      resumeId: null,
      totalSamplesInProject: 1,
    });
    state = queueReducer(state, { type: "NEXT" });
    expect(state.index).toBe(0);
    state = queueReducer(state, { type: "PREV" });
    expect(state.index).toBe(0);
  });

  it("an empty queue keeps NEXT/PREV/MARK_SAVED as safe no-ops", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: [],
      resumeId: null,
      totalSamplesInProject: 0,
    });
    state = queueReducer(state, { type: "NEXT" });
    state = queueReducer(state, { type: "PREV" });
    state = queueReducer(state, { type: "MARK_SAVED", id: "nonexistent" });
    expect(state.items).toHaveLength(0);
    expect(state.index).toBe(0);
  });
});

describe("queueReducer — MARK_MISSING", () => {
  it("flags a deleted-mid-queue sample as missing without moving the pointer", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(3),
      resumeId: null,
      totalSamplesInProject: 3,
    });
    state = queueReducer(state, { type: "MARK_MISSING", id: "id-1" });
    expect(state.items[1].status).toBe("missing");
    expect(state.index).toBe(0);
  });
});

describe("queueReducer — RECONCILE", () => {
  it("flips an item to remaining when it is still in the live unlabeled set", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(2, ["remaining", "remaining"]),
      resumeId: null,
      totalSamplesInProject: 2,
    });
    state = queueReducer(state, {
      type: "RECONCILE",
      liveItems: [makeItem({ id: "id-0" }), makeItem({ id: "id-1" })],
    });
    expect(state.items[0].status).toBe("remaining");
    expect(state.items[1].status).toBe("remaining");
  });

  it("flips an item to completed when it drops out of the live unlabeled set (labeled elsewhere)", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(2, ["remaining", "remaining"]),
      resumeId: null,
      totalSamplesInProject: 2,
    });
    // id-0 was labeled in another tab and no longer appears in the live queue
    state = queueReducer(state, { type: "RECONCILE", liveItems: [makeItem({ id: "id-1" })] });
    expect(state.items[0].status).toBe("completed");
    expect(state.items[1].status).toBe("remaining");
  });

  it("re-adds a sample to remaining if its labels were removed elsewhere (re-unlabeled)", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(1, ["completed"]),
      resumeId: null,
      totalSamplesInProject: 1,
    });
    state = queueReducer(state, { type: "RECONCILE", liveItems: [makeItem({ id: "id-0" })] });
    expect(state.items[0].status).toBe("remaining");
  });

  it("does not overwrite a status already known to be missing", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(2),
      resumeId: null,
      totalSamplesInProject: 2,
    });
    state = queueReducer(state, { type: "MARK_MISSING", id: "id-0" });
    state = queueReducer(state, { type: "RECONCILE", liveItems: [makeItem({ id: "id-1" })] });
    expect(state.items[0].status).toBe("missing");
  });

  it("collects unseen live ids as pendingNewItems instead of appending them silently", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(1),
      resumeId: null,
      totalSamplesInProject: 1,
    });
    const newItem = makeItem({ id: "id-new", filename: "IMG_new.jpg" });
    state = queueReducer(state, {
      type: "RECONCILE",
      liveItems: [makeItem({ id: "id-0" }), newItem],
    });
    expect(state.items).toHaveLength(1); // not appended yet
    expect(state.pendingNewItems.map(i => i.id)).toEqual(["id-new"]);
  });
});

describe("queueReducer — APPEND_PENDING / DISMISS_PENDING", () => {
  it("APPEND_PENDING appends pending items and clears the pending list", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(1),
      resumeId: null,
      totalSamplesInProject: 1,
    });
    state = queueReducer(state, {
      type: "RECONCILE",
      liveItems: [makeItem({ id: "id-0" }), makeItem({ id: "id-new" })],
    });
    state = queueReducer(state, { type: "APPEND_PENDING" });
    expect(state.items.map(i => i.id)).toEqual(["id-0", "id-new"]);
    expect(state.pendingNewItems).toHaveLength(0);
  });

  it("APPEND_PENDING is a no-op when there is nothing pending", () => {
    const state = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(1),
      resumeId: null,
      totalSamplesInProject: 1,
    });
    const next = queueReducer(state, { type: "APPEND_PENDING" });
    expect(next).toBe(state);
  });

  it("DISMISS_PENDING clears pending items without appending them", () => {
    let state: QueueState = queueReducer(initialState, {
      type: "LOAD_SNAPSHOT",
      items: itemsOf(1),
      resumeId: null,
      totalSamplesInProject: 1,
    });
    state = queueReducer(state, {
      type: "RECONCILE",
      liveItems: [makeItem({ id: "id-0" }), makeItem({ id: "id-new" })],
    });
    state = queueReducer(state, { type: "DISMISS_PENDING" });
    expect(state.items).toHaveLength(1);
    expect(state.pendingNewItems).toHaveLength(0);
  });
});
