/*
 * Petal Edge — ESP32 device client reference implementation
 *
 * Libraries required (Arduino/PlatformIO):
 *   - ArduinoWebsockets  (Links2004/arduinoWebSockets or gilmaimon/ArduinoWebsockets)
 *   - ArduinoJson        (bblanchon/ArduinoJson)
 *   - HTTPClient         (built-in ESP32 Arduino SDK)
 *   - WiFi               (built-in ESP32 Arduino SDK)
 *
 * Protocol source of truth:
 *   backend/app/realtime/schemas.py   — all message types
 *   backend/app/realtime/commands.py  — server→device command shapes
 *
 * Flash this to your ESP32, set the CONFIG block below, then open Serial
 * at 115200 to watch the connection lifecycle.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoWebsockets.h>
#include <ArduinoJson.h>

using namespace websockets;

// ── CONFIG ────────────────────────────────────────────────────────────────────
static const char* WIFI_SSID       = "YourSSID";
static const char* WIFI_PASS       = "YourPassword";
static const char* BACKEND_HOST    = "192.168.1.22";   // LAN IP, NOT localhost
static const uint16_t BACKEND_PORT = 8010;
static const char* API_KEY         = "ef_your_project_key";
static const char* DEVICE_ID       = "esp32-001";
static const char* FIRMWARE_VER    = "1.0.0";
static const bool  SUPPORTS_SNAP   = false;

// ── intervals (ms) ────────────────────────────────────────────────────────────
static const uint32_t HEARTBEAT_INTERVAL_MS = 20000;
static const uint32_t WS_RECONNECT_DELAY_MS = 5000;
static const uint32_t HELLO_TIMEOUT_MS      = 10000;

// ── runtime state ─────────────────────────────────────────────────────────────
static WebsocketsClient ws;
static String devicePk = "";           // set after hello-ack
static bool wsConnected   = false;
static bool snapshotActive  = false;
static bool inferenceActive = false;
static uint32_t lastHeartbeat  = 0;
static uint32_t lastWsAttempt  = 0;
static uint32_t lastSnapshotMs = 0;
static uint32_t lastInferMs    = 0;

// ── forward declarations ──────────────────────────────────────────────────────
void sendJson(const JsonDocument& doc);
void onWsMessage(WebsocketsMessage msg);
void handleCommand(const JsonDocument& msg);
void onStartSample(const JsonDocument& payload, const char* cid);
void onStartSnapshot(const char* cid);
void onStartInference(const char* cid);
void onModelUpdate(const JsonDocument& payload, const char* cid);
bool uploadSample(const char* label, uint8_t* data, size_t len,
                  const char* sampleToken, const char* hmacKey, const char* path);

// ─────────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println("[petal] connecting WiFi...");
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
    Serial.printf("\n[petal] WiFi ok ip=%s\n", WiFi.localIP().toString().c_str());

    ws.onMessage(onWsMessage);
    ws.onEvent([](WebsocketsEvent ev, String data) {
        if (ev == WebsocketsEvent::ConnectionClosed) {
            Serial.println("[ws] disconnected");
            wsConnected = false;
            devicePk    = "";
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();

    // ── maintain WS ──────────────────────────────────────────────────────────
    if (!wsConnected && (now - lastWsAttempt > WS_RECONNECT_DELAY_MS)) {
        lastWsAttempt = now;
        char url[128];
        snprintf(url, sizeof(url), "ws://%s:%u/ws/device", BACKEND_HOST, BACKEND_PORT);
        Serial.printf("[ws] connecting %s\n", url);
        if (ws.connect(url)) {
            // send hello immediately
            StaticJsonDocument<512> hello;
            hello["type"]                     = "hello";
            hello["version"]                  = "1";
            hello["apiKey"]                   = API_KEY;
            hello["deviceId"]                 = DEVICE_ID;
            hello["deviceType"]               = "esp32";
            hello["connection"]               = "wifi";
            hello["firmwareVersion"]          = FIRMWARE_VER;
            hello["protocolVersion"]          = "2";
            hello["supportsSnapshotStreaming"] = SUPPORTS_SNAP;
            sendJson(hello);
            Serial.println("[ws] hello sent, waiting for ack...");
        } else {
            Serial.println("[ws] connect failed, will retry");
        }
    }

    if (wsConnected) {
        ws.poll();
    }

    // ── heartbeat ─────────────────────────────────────────────────────────────
    if (wsConnected && devicePk.length() > 0 &&
        (now - lastHeartbeat > HEARTBEAT_INTERVAL_MS)) {
        lastHeartbeat = now;

        char url[256];
        snprintf(url, sizeof(url),
                 "http://%s:%u/api/v1/devices/%s/heartbeat",
                 BACKEND_HOST, BACKEND_PORT, devicePk.c_str());

        StaticJsonDocument<128> hb;
        hb["firmware_version"] = FIRMWARE_VER;
        hb["ip_address"]       = WiFi.localIP().toString();
        if (SUPPORTS_SNAP) hb["supports_snapshot_streaming"] = true;

        String body;
        serializeJson(hb, body);

        HTTPClient http;
        http.begin(url);
        http.addHeader("Content-Type", "application/json");
        int code = http.POST(body);
        if (code == 200) {
            Serial.println("[heartbeat] ok");
        } else {
            Serial.printf("[heartbeat] failed HTTP %d\n", code);
        }
        http.end();
    }

    // ── snapshot tick ─────────────────────────────────────────────────────────
    if (wsConnected && snapshotActive && (now - lastSnapshotMs > 100)) {
        lastSnapshotMs = now;
        // Send JSON header; Phase 3: follow with binary frame
        StaticJsonDocument<128> snap;
        snap["type"]   = "snapshot";
        snap["width"]  = 96;
        snap["height"] = 96;
        snap["format"] = "jpeg";
        sendJson(snap);
        // ws.sendBinary(frameData, frameLen);   // Phase 3
    }

    // ── inference tick ────────────────────────────────────────────────────────
    if (wsConnected && inferenceActive && (now - lastInferMs > 200)) {
        lastInferMs = now;
        StaticJsonDocument<256> res;
        res["type"] = "inference-result";
        JsonObject pl = res.createNestedObject("payload");
        // TODO: replace with real model output
        pl["label"]      = "unknown";
        pl["confidence"] = 0.0f;
        sendJson(res);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void onWsMessage(WebsocketsMessage raw) {
    StaticJsonDocument<1024> msg;
    DeserializationError err = deserializeJson(msg, raw.data());
    if (err) {
        Serial.printf("[ws] bad JSON: %s\n", err.c_str());
        return;
    }

    const char* t = msg["type"] | "";

    if (strcmp(t, "hello-ack") == 0) {
        if (msg["success"].as<bool>()) {
            devicePk    = msg["id"].as<String>();
            wsConnected = true;
            lastHeartbeat = 0;   // fire heartbeat immediately
            Serial.printf("[ws] hello-ack ok device_pk=%s\n", devicePk.c_str());
        } else {
            Serial.println("[ws] hello-ack failed, disconnecting");
            ws.close();
        }

    } else if (strcmp(t, "ping") == 0) {
        StaticJsonDocument<64> pong;
        pong["type"] = "pong";
        sendJson(pong);

    } else if (strcmp(t, "start-sample") == 0) {
        onStartSample(msg["payload"], msg["correlationId"] | "");

    } else if (strcmp(t, "start-snapshot") == 0) {
        snapshotActive = true;
        Serial.println("[cmd] snapshot start");

    } else if (strcmp(t, "stop-snapshot") == 0) {
        snapshotActive = false;
        Serial.println("[cmd] snapshot stop");

    } else if (strcmp(t, "start-inference-stream") == 0) {
        inferenceActive = true;
        Serial.println("[cmd] inference start");

    } else if (strcmp(t, "stop-inference-stream") == 0) {
        inferenceActive = false;
        Serial.println("[cmd] inference stop");

    } else if (strcmp(t, "model-update") == 0) {
        onModelUpdate(msg["payload"], msg["correlationId"] | "");

    } else {
        Serial.printf("[ws] unhandled type=%s\n", t);
    }
}

// ── sampling ──────────────────────────────────────────────────────────────────
void onStartSample(const JsonDocument& payload, const char* cid) {
    const char* label     = payload["label"] | "";
    uint32_t    lengthMs  = payload["length"] | 5000;
    const char* hmacKey   = payload["hmacKey"] | "";
    const char* sampleTok = payload["sampleToken"] | "";
    const char* path      = payload["path"] | "";

    // ack immediately so server unblocks
    StaticJsonDocument<128> ack;
    ack["type"]          = "sample-ack";
    ack["success"]       = true;
    ack["correlationId"] = cid;
    sendJson(ack);

    StaticJsonDocument<128> started;
    started["type"]          = "sample-started";
    started["label"]         = label;
    started["length"]        = lengthMs;
    started["correlationId"] = cid;
    sendJson(started);

    // TODO: run real sensor acquisition here (blocking or async task)
    size_t dataLen = 128;   // placeholder
    uint8_t* data  = (uint8_t*)malloc(dataLen);
    if (!data) {
        StaticJsonDocument<128> fail;
        fail["type"]          = "sample-failed";
        fail["error"]         = "OOM";
        fail["correlationId"] = cid;
        sendJson(fail);
        return;
    }
    memset(data, 0, dataLen);
    delay(lengthMs);   // replace with real acquisition

    bool ok = uploadSample(label, data, dataLen, sampleTok, hmacKey, path);
    free(data);

    if (ok) {
        StaticJsonDocument<128> stopped;
        stopped["type"]          = "sample-stopped";
        stopped["label"]         = label;
        stopped["correlationId"] = cid;
        sendJson(stopped);
    } else {
        StaticJsonDocument<128> fail;
        fail["type"]          = "sample-failed";
        fail["error"]         = "upload failed";
        fail["correlationId"] = cid;
        sendJson(fail);
    }
}

bool uploadSample(const char* label, uint8_t* data, size_t len,
                  const char* sampleToken, const char* hmacKey, const char* path) {
    // `path` comes from the start-sample command (Phase 4; see
    // device_client/PROTOCOL.md) and is absolute from host root. Older
    // backends may omit it — fall back to the legacy training path for
    // those, but log it since it silently miscategorises testing/anomaly
    // samples into training (P0-A5).
    char url[256];
    if (path != nullptr && strlen(path) > 0) {
        snprintf(url, sizeof(url), "http://%s:%u%s", BACKEND_HOST, BACKEND_PORT, path);
    } else {
        Serial.println("[sample] start-sample command had no 'path' field; "
                        "falling back to the training ingestion path (backend may be outdated)");
        snprintf(url, sizeof(url),
                 "http://%s:%u/api/v1/ingestion/training/data",
                 BACKEND_HOST, BACKEND_PORT);
    }

    HTTPClient http;
    http.begin(url);
    http.addHeader("Content-Type", "application/octet-stream");
    // The current backend requires x-api-key even when per-sample auth is used.
    http.addHeader("x-api-key", API_KEY);

    if (strlen(sampleToken) > 0) {
        http.addHeader("x-sample-token", sampleToken);
        // TODO: compute HMAC-SHA256(hmacKey, data) and set x-signature header
    }
    if (strlen(label) > 0) {
        http.addHeader("x-label", label);
    }

    int code = http.POST(data, len);
    http.end();
    return (code == 200 || code == 201);
}

// ── OTA update ────────────────────────────────────────────────────────────────
void onModelUpdate(const JsonDocument& payload, const char* cid) {
    const char* url     = payload["url"] | "";
    const char* version = payload["version"] | "";

    if (strlen(url) == 0) {
        StaticJsonDocument<128> rej;
        rej["type"]          = "update-rejected";
        rej["error"]         = "missing url";
        rej["correlationId"] = cid;
        sendJson(rej);
        return;
    }

    Serial.printf("[ota] update url=%s version=%s\n", url, version);

    auto sendLifecycle = [&](const char* type, const char* extra_key = nullptr, const char* extra_val = nullptr) {
        StaticJsonDocument<128> ev;
        ev["type"]          = type;
        ev["correlationId"] = cid;
        if (extra_key) ev[extra_key] = extra_val;
        sendJson(ev);
    };

    sendLifecycle("update-accepted");
    sendLifecycle("update-download-started");

    // TODO: use esp_https_ota or custom chunked download
    // esp_err_t ret = esp_https_ota(&ota_config);
    bool downloadOk = true;   // stub

    if (!downloadOk) {
        sendLifecycle("update-install-failed", "message", "download error");
        return;
    }

    sendLifecycle("update-install-started");

    // TODO: verify CRC / signature, write to OTA partition, reboot
    bool installOk = true;   // stub

    if (installOk) {
        StaticJsonDocument<128> done;
        done["type"]          = "update-install-succeeded";
        done["version"]       = version;
        done["correlationId"] = cid;
        sendJson(done);
        Serial.println("[ota] done, rebooting...");
        delay(1000);
        ESP.restart();
    } else {
        sendLifecycle("update-install-failed", "message", "flash error");
    }
}

// ── helper ────────────────────────────────────────────────────────────────────
void sendJson(const JsonDocument& doc) {
    String out;
    serializeJson(doc, out);
    ws.send(out);
}
