export type DeviceMode = "idle" | "sampling" | "inference";

// Decoded `sensor.frame` Studio-WS payload (see SignalPreview.tsx). The
// backend broadcast side (Batch 3.2, app/motion/websocket/sensor_frames.py)
// isn't wired up yet, so this only covers the field the preview chart reads.
export interface SensorFramePayload {
  values: number[];
}

/** Runtime health snapshot reported on the device heartbeat and merged into
 *  device_metadata.diagnostics. All fields optional — the device omits any it
 *  can't read. */
export interface DeviceDiagnostics {
  cpu_percent?: number;
  mem_percent?: number;
  mem_total_mb?: number;
  disk_percent?: number;
  temp_c?: number;
  uptime_s?: number;
  sensors?: { imu?: boolean; camera?: boolean };
}

export interface Device {
  id: string;
  device_id: string;
  name: string;
  device_type: string;
  is_online: boolean;
  is_connected?: boolean;
  firmware_version?: string;
  /** Transport the device last reported: wifi, ethernet, serial, ble, … */
  connection?: string;
  ip_address?: string;
  last_seen?: string;
  deployment_target?: string;
  device_profile?: string;
  installed_deployment_id?: string;
  installed_model_version?: string;
  supports_snapshot_streaming?: boolean;
  device_metadata?: Record<string, any> & { diagnostics?: DeviceDiagnostics };
  /** Reported sensors. Each entry is typically `{ name, type?, units? }`;
   *  the list endpoint now includes this so a sensor picker can be driven
   *  straight from devicesApi.list. */
  sensors?: any[];
  // runtime-only (from WS, not REST)
  mode?: DeviceMode;
}

export interface UpdateStatus {
  status: "requested" | "downloading" | "flashing" | "done" | "failed";
  deployment_id?: string;
  message?: string;
  updated_at?: string;
}

export interface CompatibleDeployment {
  deployment_id: string;
  deployment_target?: string;
  device_profile?: string;
  download_url?: string;
}

// Project-scoped device key — list item (safe summary, no secret material).
// Mirrors backend DeviceKeySummary.
export interface DeviceKey {
  id: string;
  project_id: string;
  name?: string | null;
  api_key_prefix?: string | null;
  is_active: boolean;
  created_at: string;
}

// Returned once on creation — api_key and hmac_key are not retrievable afterwards.
// Mirrors backend DeviceKeyResponse.
export interface DeviceKeySecret {
  id: string;
  project_id: string;
  name?: string | null;
  api_key: string;
  hmac_key: string;
  is_active: boolean;
  created_at: string;
}

// Published device-client release manifest — mirrors backend contract §14.4.
// Served by GET /api/v1/device-client/latest (and /{version}).
export interface DeviceClientManifest {
  version: string;        // e.g. "v1.0.0"
  filename: string;       // petal-device-v1.0.0.tar.gz
  size: number;           // tarball size in bytes
  sha256: string;         // 64 hex chars
  download_url: string;   // byte-serving path for the tarball
  created_at: string;     // ISO 8601
}

export interface StudioEvent {
  type: string;
  project_id: string;
  device_id?: string;
  ts: string;
  payload?: Record<string, any>;
  devices?: Array<Record<string, any>>;
}
