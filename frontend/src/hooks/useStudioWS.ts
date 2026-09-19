"use client";
import { useEffect, useRef, useState } from "react";
import { devicesApi } from "@/utils/api";
import type { StudioEvent } from "@/types/devices";

const MAX_RETRIES = 5;
const RETRY_DELAY_MS = 3000;

export function useStudioWS(
  projectId: string | null,
  onEvent: (e: StudioEvent) => void,
): { connected: boolean } {
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const unmountedRef = useRef(false);
  const retriesRef = useRef(0);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connectInFlightRef = useRef(false);
  const runIdRef = useRef(0);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  useEffect(() => {
    console.log("[useStudioWS] effect start", { projectId });
    if (!projectId) return;
    const runId = ++runIdRef.current;
    unmountedRef.current = false;
    retriesRef.current = 0;

    async function connect() {
      console.log("[useStudioWS] connect()", { projectId, runId, currentRun: runIdRef.current });
      if (unmountedRef.current || runId !== runIdRef.current) return;
      if (connectInFlightRef.current) return;
      if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
        return;
      }

      connectInFlightRef.current = true;
      let token: string;
      try {
        const { data } = await devicesApi.socketToken(projectId!);
        token = data.token;
      } catch {
        connectInFlightRef.current = false;
        scheduleRetry();
        return;
      }

      if (unmountedRef.current || runId !== runIdRef.current) {
        connectInFlightRef.current = false;
        return;
      }

      const url = devicesApi.studioWsUrl() + "?token=" + token;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("[useStudioWS] ws open", { projectId, runId });
        connectInFlightRef.current = false;
        if (!unmountedRef.current && runId === runIdRef.current) {
          setConnected(true);
          retriesRef.current = 0;
        }
      };

      ws.onmessage = (e) => {
        if (unmountedRef.current) return;
        try {
          const event: StudioEvent = JSON.parse(e.data);
          onEventRef.current(event);
        } catch {
          // ignore malformed frames
        }
      };

      ws.onerror = () => {
        connectInFlightRef.current = false;
        setConnected(false);
      };

      ws.onclose = (event) => {
        console.log("[useStudioWS] ws close", {
          projectId,
          runId,
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean,
        });
        connectInFlightRef.current = false;
        setConnected(false);
        if (wsRef.current === ws) wsRef.current = null;
        if (!unmountedRef.current && runId === runIdRef.current) scheduleRetry();
      };
    }

    function scheduleRetry() {
      if (unmountedRef.current) return;
      if (retriesRef.current >= MAX_RETRIES) return;
      retriesRef.current += 1;
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
      retryTimerRef.current = setTimeout(connect, RETRY_DELAY_MS);
    }

    connect();

    return () => {
      console.log("[useStudioWS] cleanup", { projectId, runId });
      unmountedRef.current = true;
      runIdRef.current += 1;
      connectInFlightRef.current = false;
      if (retryTimerRef.current) {
        clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
      }
      wsRef.current?.close();
      wsRef.current = null;
      setConnected(false);
    };
  }, [projectId]);

  return { connected };
}
