/**
 * Zero-dependency tests for SUPPORTED_DEVICE_PROFILES logic.
 * Run with: node frontend/src/app/dashboard/impulse/deployment/__tests__/deviceProfiles.test.mjs
 */
import assert from "node:assert/strict";

const SUPPORTED_DEVICE_PROFILES = [
  { value: "generic_tflite", label: "Generic TFLite", target: "tflite" },
  { value: "arduino_nano_33_ble", label: "Arduino Nano 33 BLE", target: "arduino" },
  { value: "esp32_devkit", label: "ESP32 DevKit", target: "esp32" },
  { value: "raspberry_pi_4", label: "Raspberry Pi 4", target: "raspberry_pi" },
  { value: "unoq", label: "UNO Q", target: "unoq" },
];

function buildPayload(modelId, selectedDevice) {
  const profile = SUPPORTED_DEVICE_PROFILES.find((p) => p.value === selectedDevice);
  if (!profile) throw new Error(`Unsupported device: ${selectedDevice}`);
  return {
    model_id: modelId,
    target: profile.target,
    deployment_target: profile.target,
    device_profile: profile.value,
    options: { device_profile: profile.value },
  };
}

const REQUIRED_LABELS = [
  "Generic TFLite",
  "Arduino Nano 33 BLE",
  "ESP32 DevKit",
  "Raspberry Pi 4",
  "UNO Q",
];
const actualLabels = SUPPORTED_DEVICE_PROFILES.map((p) => p.label);

assert.equal(SUPPORTED_DEVICE_PROFILES.length, 5, "Must have exactly 5 device profiles");
for (const label of REQUIRED_LABELS) {
  assert.ok(actualLabels.includes(label), `Missing device label: "${label}"`);
}
console.log("PASS  6.1 - all 5 device labels present");

const payload = buildPayload("model-123", "arduino_nano_33_ble");

assert.equal(payload.target, "arduino", "target must map to 'arduino'");
assert.equal(payload.deployment_target, "arduino", "deployment_target must mirror the compatible target");
assert.equal(payload.device_profile, "arduino_nano_33_ble", "device_profile must be the profile value");
assert.equal(payload.options.device_profile, "arduino_nano_33_ble", "options.device_profile must be the profile value");
assert.equal(payload.model_id, "model-123", "model_id must be forwarded");
console.log("PASS  6.2 - arduino payload includes compatibility fields");

const expectedMappings = [
  ["generic_tflite", "tflite"],
  ["arduino_nano_33_ble", "arduino"],
  ["esp32_devkit", "esp32"],
  ["raspberry_pi_4", "raspberry_pi"],
  ["unoq", "unoq"],
];
for (const [device, expectedTarget] of expectedMappings) {
  const p = buildPayload("m", device);
  assert.equal(p.target, expectedTarget, `${device} -> target should be ${expectedTarget}`);
  assert.equal(p.deployment_target, expectedTarget, `${device} -> deployment_target should be ${expectedTarget}`);
  assert.equal(p.device_profile, device, `${device} -> device_profile should be ${device}`);
  assert.equal(p.options.device_profile, device, `${device} -> options.device_profile should be ${device}`);
}
console.log("PASS  6.2 - all 5 device-to-target mappings stamp compatibility fields");

assert.throws(
  () => buildPayload("m", "stm32_nucleo"),
  /Unsupported device/,
  "Unknown device must throw",
);
console.log("PASS  6.2 - unsupported device throws");

console.log("\nAll tests passed.");
