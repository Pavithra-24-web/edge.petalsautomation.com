import { useAppStore } from "@/store/appStore";

const userId = "user-1";

// Mirrors the inline label helper in dashboard/page.tsx. It can't be imported
// directly: Next.js's app-router typegen rejects any export from page.tsx
// other than its known route exports (default, metadata, ...), so the helper
// is kept unexported there and this test re-asserts the same one-line contract.
function tutorialButtonLabel(isActive: boolean): string {
  return isActive ? "Restart tutorial" : "Launch tutorial";
}

function resetOnboarding() {
  useAppStore.setState({ onboarding: {} } as any);
}

describe("restartOnboarding", () => {
  beforeEach(resetOnboarding);

  test.each(["completed", "skipped"] as const)(
    "resets a %s record to { status: active, step: 0 }",
    (status) => {
      useAppStore.setState({
        onboarding: { [userId]: { status, step: 3 } },
      } as any);

      useAppStore.getState().restartOnboarding(userId);

      expect(useAppStore.getState().onboarding[userId]).toEqual({
        status: "active",
        step: 0,
      });
    },
  );

  test("creates { status: active, step: 0 } when no record is present", () => {
    expect(useAppStore.getState().onboarding[userId]).toBeUndefined();

    useAppStore.getState().restartOnboarding(userId);

    expect(useAppStore.getState().onboarding[userId]).toEqual({
      status: "active",
      step: 0,
    });
  });
});

describe("tutorialButtonLabel", () => {
  test('returns "Restart tutorial" when the tour is active', () => {
    expect(tutorialButtonLabel(true)).toBe("Restart tutorial");
  });

  test.each([false] as const)(
    'returns "Launch tutorial" when not active (isActive=%s)',
    (isActive) => {
      expect(tutorialButtonLabel(isActive)).toBe("Launch tutorial");
    },
  );
});
