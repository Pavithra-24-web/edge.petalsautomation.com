// Registry of boards this Connect Device flow recognizes. A new board is one
// entry here (plus a matching mock profile in connectionService.ts) — no
// change to any dialog/list/card component.

import { Cpu, CircuitBoard } from "lucide-react";
import type { SupportedBoardId, SupportedBoardMeta } from "./types";

export const SUPPORTED_BOARDS: Record<SupportedBoardId, SupportedBoardMeta> = {
  arduino_uno_q: {
    id: "arduino_uno_q",
    label: "Arduino UNO Q",
    tagline: "Onboard IMU · USB",
    vendorId: "2341",
    icon: Cpu,
  },
  raspberry_pi: {
    id: "raspberry_pi",
    label: "Raspberry Pi",
    tagline: "USB-connected sensor board",
    vendorId: "2E8A",
    icon: CircuitBoard,
  },
};

export const SUPPORTED_BOARD_ORDER: SupportedBoardId[] = ["arduino_uno_q", "raspberry_pi"];
