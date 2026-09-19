"use client";
import SupportedDeviceCard from "./SupportedDeviceCard";
import type { DiscoveredDevice } from "./types";

export default function SupportedDeviceList({
  devices,
  connectingId,
  onConnect,
}: {
  devices: DiscoveredDevice[];
  connectingId: string | null;
  onConnect: (device: DiscoveredDevice) => void;
}) {
  return (
    <div className="space-y-2.5" role="list" aria-label="Discovered devices">
      {devices.map((device) => (
        <div key={device.scanId} role="listitem">
          <SupportedDeviceCard
            device={device}
            connecting={connectingId === device.scanId}
            disabled={connectingId !== null && connectingId !== device.scanId}
            onConnect={() => onConnect(device)}
          />
        </div>
      ))}
    </div>
  );
}
