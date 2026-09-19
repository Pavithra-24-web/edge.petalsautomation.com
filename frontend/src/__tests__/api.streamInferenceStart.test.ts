/**
 * Focused tests for devicesApi.streamInferenceStart payload construction.
 *
 * Mocks axios so the HTTP layer is exercised without a real server.
 */
import axios from "axios";

jest.mock("axios", () => {
  const mockPost = jest.fn().mockResolvedValue({ data: { stream_id: "s1", stream_type: "inference" } });
  const mockCreate = jest.fn().mockReturnValue({
    post: mockPost,
    interceptors: {
      request: { use: jest.fn() },
      response: { use: jest.fn() },
    },
  });
  return { default: { create: mockCreate }, create: mockCreate };
});

// Import after mock is in place
import { devicesApi } from "@/utils/api";

const mockPost = (devicesApi as any)._axiosInstance
  ? (devicesApi as any)._axiosInstance.post
  : null;

// Resolve the underlying mock post from the axios instance
function getPost() {
  // api.ts calls axios.create() at module load; the returned instance has .post mocked
  const axiosMod = jest.requireMock("axios");
  const instance = axiosMod.default.create.mock.results[0]?.value;
  return instance?.post as jest.Mock;
}

beforeEach(() => {
  getPost()?.mockClear();
});

describe("devicesApi.streamInferenceStart", () => {
  it("sends empty body when called with no optional args", async () => {
    await devicesApi.streamInferenceStart("dev-1");
    const [url, body] = getPost().mock.calls[0];
    expect(url).toBe("/devices/dev-1/streams/inference/start");
    expect(body).toEqual({});
  });

  it("sends fomo_threshold when provided", async () => {
    await devicesApi.streamInferenceStart("dev-1", 0.7);
    const [, body] = getPost().mock.calls[0];
    expect(body).toMatchObject({ fomo_threshold: 0.7 });
  });

  it("sends sensor/frequency/sample_length_ms when opts provided", async () => {
    await devicesApi.streamInferenceStart("dev-1", undefined, {
      sensor: "accelerometer",
      frequency: 62.5,
      sample_length_ms: 2000,
    });
    const [, body] = getPost().mock.calls[0];
    expect(body).toEqual({
      sensor: "accelerometer",
      frequency: 62.5,
      sample_length_ms: 2000,
    });
  });

  it("omits undefined opts fields from payload", async () => {
    await devicesApi.streamInferenceStart("dev-1", undefined, { sensor: "mic" });
    const [, body] = getPost().mock.calls[0];
    expect(body).toEqual({ sensor: "mic" });
    expect("frequency" in body).toBe(false);
    expect("sample_length_ms" in body).toBe(false);
  });

  it("combines fomo_threshold with sensor config opts", async () => {
    await devicesApi.streamInferenceStart("dev-2", 0.5, {
      sensor: "microphone",
      frequency: 16000,
      sample_length_ms: 1000,
    });
    const [url, body] = getPost().mock.calls[0];
    expect(url).toBe("/devices/dev-2/streams/inference/start");
    expect(body).toEqual({
      fomo_threshold: 0.5,
      sensor: "microphone",
      frequency: 16000,
      sample_length_ms: 1000,
    });
  });

  it("does not send fomo_threshold when fomoThreshold is null/undefined", async () => {
    await devicesApi.streamInferenceStart("dev-1", undefined, { sensor: "accel" });
    const [, body] = getPost().mock.calls[0];
    expect("fomo_threshold" in body).toBe(false);
  });
});
