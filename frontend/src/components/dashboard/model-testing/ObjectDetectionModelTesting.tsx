"use client";

/**
 * ModelTestingPage — data-fetching orchestrator
 *
 * Impulse resolution chain (never collapses to blank):
 *   URL ?impulseId=X
 *     → appStore.activeImpulse  (cross-checked against active project)
 *       → first impulse for activeProject (API)
 *         → first impulse for any project (API)
 *           → empty envelope — cards still render with empty-state messages
 *
 * After classify-all we do a FULL refetch (page-data + test-data + versions)
 * rather than manually merging state, so the UI is always consistent with the DB.
 *
 * Rendering contract (never violated):
 *   Page shell       — always rendered
 *   Both cards       — always rendered
 *   Card inner body  — skeleton | empty-state | data
 */

import { useEffect, useState, useCallback, useRef } from "react";
import { RefreshCw } from "lucide-react";
import ImpulseNotReady from "@/components/dashboard/ImpulseNotReady";
import { useSearchParams } from "next/navigation";
import toast from "react-hot-toast";

import { useAppStore } from "@/store/appStore";
import { impulsesApi, projectsApi, modelTestingApi } from "@/utils/api";
import { useTrainingValidity } from "@/hooks/useTrainingValidity";

import InfoBanner from "./InfoBanner";
import TestDataCard from "./TestDataCard";
import ModelTestingOutputCard from "./ModelTestingOutputCard";
import SampleInspectorCard from "./SampleInspectorCard";

import type {
  ModelTestingPageData,
  TestSampleRow,
  ModelVersion,
} from "@/types/model-testing";

// ─────────────────────────────────────────────────────────────────────────────

export default function ObjectDetectionModelTesting() {
  const searchParams = useSearchParams();

  const {
    activeProject,
    activeImpulse: storeImpulse,
    setActiveImpulse,
    setActiveProject,
  } = useAppStore();

  // ── Resolved references ──────────────────────────────────────────────────
  const [resolvedProject, setResolvedProject] = useState<any>(activeProject ?? null);
  const [resolvedImpulse, setResolvedImpulse] = useState<any>(storeImpulse ?? null);

  // Keep a ref that is always current so callbacks never close over stale state
  const resolvedImpulseRef = useRef<any>(storeImpulse ?? null);
  const resolvedProjectRef = useRef<any>(activeProject ?? null);

  const setImpulse = (i: any) => {
    resolvedImpulseRef.current = i;
    setResolvedImpulse(i);
    if (i) setActiveImpulse(i);
  };

  const setProject = (p: any) => {
    resolvedProjectRef.current = p;
    setResolvedProject(p);
    if (p) setActiveProject(p);
  };

  // ── Data state ──────────────────────────────────────────────────────────
  const [pageData, setPageData] = useState<ModelTestingPageData | null>(null);
  const [testSamples, setTestSamples] = useState<TestSampleRow[]>([]);
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([]);
  const [selectedSample, setSelectedSample] = useState<TestSampleRow | null>(null);

  // ── Readiness signal ─────────────────────────────────────────────────────
  // Single source of truth: the latest training run's status. This is shared
  // with every training-dependent surface so the Training Output panel and
  // this page can never disagree about whether the project is "trained".
  const {
    hasValidTrainingOutput,
    loading: trainingValidityLoading,
  } = useTrainingValidity(resolvedImpulse?.id ?? null);

  // ── Loading flags ────────────────────────────────────────────────────────
  const [resolving, setResolving] = useState(true);
  const [loadingPageData, setLoadingPageData] = useState(false);
  const [loadingTestData, setLoadingTestData] = useState(false);
  const [loadingVersions, setLoadingVersions] = useState(false);
  const [classifying, setClassifying] = useState(false);

  // ── Reference resolution ─────────────────────────────────────────────────

  const resolveProject = useCallback(async () => {
    if (activeProject?.id) {
      setProject(activeProject);
      return activeProject;
    }
    try {
      const { data } = await projectsApi.list();
      const list: any[] = Array.isArray(data) ? data : (data?.projects ?? []);
      const first = list[0];
      if (first) { setProject(first); return first; }
    } catch {
      // no projects available
    }
    return null;
  }, [activeProject]); // eslint-disable-line react-hooks/exhaustive-deps

  const resolveImpulse = useCallback(async (project: any) => {
    // 1. URL param
    const urlId = searchParams?.get("impulseId");
    if (urlId) {
      try {
        const { data } = await impulsesApi.get(urlId);
        if (data?.id) { setImpulse(data); return data; }
      } catch { /* fall through */ }
    }

    // 2. Persisted store impulse (validate it belongs to the resolved project)
    if (storeImpulse?.id) {
      const ok = !project?.id || storeImpulse.project_id === project?.id;
      if (ok) { setImpulse(storeImpulse); return storeImpulse; }
    }

    // 3. First impulse for the resolved project
    if (project?.id) {
      try {
        const { data } = await impulsesApi.list(project.id);
        const list: any[] = Array.isArray(data)
          ? data
          : (data?.impulses ?? data?.items ?? []);
        const first = list[0];
        if (first) { setImpulse(first); return first; }
      } catch { /* no impulses */ }
    }
    return null;
  }, [storeImpulse, searchParams]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Data loaders ─────────────────────────────────────────────────────────

  const loadPageData = useCallback(async (
    impulseId: string | null,
    projectId: string | null,
  ) => {
    setLoadingPageData(true);
    try {
      const { data } = await modelTestingApi.pageData(impulseId, projectId);

      // Sync resolved impulse with what the backend actually resolved
      if (data?.impulse?.id && data.impulse.id !== resolvedImpulseRef.current?.id) {
        setImpulse(data.impulse);
      }

      setPageData(data);
    } catch (err: any) {
      if ((err?.response?.status ?? 0) >= 500) {
        toast.error("Failed to load model testing data");
      }
      // Do NOT set pageData to null — keep whatever was there before
    } finally {
      setLoadingPageData(false);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const loadTestData = useCallback(async (impulseId: string) => {
    setLoadingTestData(true);
    try {
      const pageSize = 200;
      const firstRes = await modelTestingApi.testData(impulseId, 1, pageSize);
      const firstData = firstRes.data ?? {};
      const total = Number(firstData.total ?? 0);
      const totalPages = Math.max(1, Number(firstData.total_pages ?? 1));
      const items: TestSampleRow[] = [...(firstData.items ?? [])];

      if (total > items.length && totalPages > 1) {
        const pageRequests: Promise<any>[] = [];
        for (let page = 2; page <= totalPages; page += 1) {
          pageRequests.push(modelTestingApi.testData(impulseId, page, pageSize));
        }
        const pageResponses = await Promise.all(pageRequests);
        for (const response of pageResponses) {
          items.push(...(response.data?.items ?? []));
        }
      }

      setTestSamples(items);
      // Keep selection current if the sample is still in the new list;
      // otherwise auto-select the first sample.
      setSelectedSample((prev) => {
        if (prev) {
          const refreshed = items.find((s) => s.id === prev.id);
          if (refreshed) return refreshed;
        }
        return items[0] ?? null;
      });
    } catch {
      setTestSamples([]);
      setSelectedSample(null);
    } finally {
      setLoadingTestData(false);
    }
  }, []);

  const loadVersions = useCallback(async (impulseId: string) => {
    setLoadingVersions(true);
    try {
      const { data } = await modelTestingApi.modelVersions(impulseId);
      setModelVersions(Array.isArray(data) ? data : []);
    } catch {
      setModelVersions([]);
    } finally {
      setLoadingVersions(false);
    }
  }, []);

  /** Full refresh — used after classify-all to keep all sections in sync. */
  const refreshAll = useCallback(async (impulseId: string, projectId: string | null) => {
    await Promise.all([
      loadPageData(impulseId, projectId),
      loadTestData(impulseId),
      loadVersions(impulseId),
    ]);
  }, [loadPageData, loadTestData, loadVersions]);

  // ── Bootstrap ─────────────────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      setResolving(true);
      try {
        const project = await resolveProject();
        const impulse = await resolveImpulse(project);
        if (cancelled) return;
        if (impulse?.id) {
          await Promise.all([
            loadPageData(impulse.id, project?.id ?? null),
            loadTestData(impulse.id),
            loadVersions(impulse.id),
          ]);
        } else {
          // No impulse — no trained model can exist
          await loadPageData(null, project?.id ?? null);
        }
      } finally {
        if (!cancelled) setResolving(false);
      }
    }

    bootstrap();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Classify all ─────────────────────────────────────────────────────────

  const handleClassifyAll = useCallback(async () => {
    // Use the ref (always current) so the closure is never stale
    const impulse = resolvedImpulseRef.current;
    // Fallback: impulse that the backend resolved and returned in pageData
    const impulseId = impulse?.id
      ?? (pageData as any)?.impulse?.id
      ?? null;

    if (!impulseId) {
      toast.error("No impulse resolved yet — please wait or navigate to Create impulse.");
      return;
    }

    const projectId = impulse?.project_id
      ?? (pageData as any)?.project?.id
      ?? resolvedProjectRef.current?.id
      ?? null;

    setClassifying(true);
    try {
      const { data } = await modelTestingApi.classifyAll(
        impulseId,
        (pageData as any)?.selected_model_version?.id ?? null,
      );

      const runId = data?.run_id ?? null;
      if (!runId) {
        throw new Error("Missing run_id from classify-all response");
      }

      let attempts = 0;
      const maxAttempts = 120; // ~2 minutes at 1s polling
      while (attempts < maxAttempts) {
        const { data: statusData } = await modelTestingApi.classifyAllStatus(runId);
        const status = statusData?.status;

        if (status === "completed") {
          break;
        }
        if (status === "failed" || status === "cancelled") {
          throw new Error(statusData?.error || `Classification ${status}`);
        }

        attempts += 1;
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }

      // Full refetch to guarantee every section is fresh after completion
      await refreshAll(impulseId, projectId);

      toast.success("Classification complete");
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      // Show the actual backend message — it now distinguishes "no test-split
      // samples" from "no samples at all", which is actionable for the user.
      toast.error(detail ?? "Classification failed — please try again.");
      // Do NOT clear any state — keep all cards visible with previous data
    } finally {
      setClassifying(false);
    }
    // pageData is read inside but via the setter closure — keep dep list tight
    // to avoid stale page_data.selected_model_version from old renders
  }, [pageData, refreshAll]);

  // ── Derived display values ────────────────────────────────────────────────

  const isLoadingCards = resolving || loadingPageData;
  // Show "Almost there!" whenever the latest training run is not in the
  // "completed" state — that includes cancelled, failed, running, pending, and
  // never-trained. Older completed-run artifacts are never displayed as active
  // results.
  const notReady =
    !resolving && !trainingValidityLoading && !!resolvedImpulse?.id && !hasValidTrainingOutput;
  // No impulse exists at all (neither resolved client-side nor returned by the
  // backend page-data). Without this branch the page falls through to the full
  // test-data UI instead of the "Almost there!" warning every sibling page shows.
  const hasImpulse = !!(resolvedImpulse?.id ?? (pageData as any)?.impulse?.id);
  const noImpulse = !resolving && !loadingPageData && !hasImpulse;
  const projectName = resolvedProject?.name ?? (pageData as any)?.project?.name ?? null;
  const impulseName = resolvedImpulse?.name ?? (pageData as any)?.impulse?.name ?? null;

  // Until the bootstrap effect settles, we can't tell whether to render the
  // main UI or the "Almost there!" warning. Render a loading state first so
  // the warning is the first non-loading frame for untrained impulses.
  if (resolving || (resolvedImpulse?.id && trainingValidityLoading)) {
    return (
      <div className="flex h-[calc(100vh-theme('spacing.16'))] items-center justify-center pt-2">
        <span className="inline-flex items-center gap-2 text-sm text-[var(--app-text-soft)]">
          <RefreshCw size={14} className="animate-spin" />
          Verifying model status...
        </span>
      </div>
    );
  }

  if (noImpulse) {
    return (
      <ImpulseNotReady
        description="No impulse to test yet. Create an impulse and train a model before you can validate it."
        tip="Head to Create impulse to design one, then come back here once training finishes."
      />
    );
  }

  if (notReady) {
    return (
      <ImpulseNotReady
        description="Your impulse is not fully trained. Use the items in the navigation bar to configure and train your model before you can validate your model."
        tip="Train your model first — model testing validates a trained model against your test set."
      />
    );
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="pe-mt-page model-testing-page mx-auto max-w-[96rem] space-y-6">
      <InfoBanner
        projectName={resolving && !projectName ? null : projectName}
        impulseName={resolving && !impulseName ? null : impulseName}
      />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        <div className="lg:col-span-7 min-w-0">
          <TestDataCard
            samples={testSamples}
            loading={isLoadingCards || loadingTestData}
            classifying={classifying}
            selectedSampleId={selectedSample?.id ?? null}
            onClassifyAll={handleClassifyAll}
            onSelectSample={setSelectedSample}
          />
        </div>

        <div className="lg:col-span-5 min-w-0">
          <SampleInspectorCard sample={selectedSample} />
        </div>
      </div>

      <ModelTestingOutputCard
        loading={isLoadingCards || loadingVersions}
        accuracy={(pageData as any)?.accuracy ?? null}
        metrics={(pageData as any)?.metrics ?? []}
        modelVersions={modelVersions}
        activeVersion={(pageData as any)?.selected_model_version ?? null}
      />
    </div>
  );
}
