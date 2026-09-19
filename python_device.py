import time
import requests

BASE_URL = "http://192.168.1.11:8010"  # same as your UI
DEVICE_ID = "a1a2bfc1-cc3e-4a94-b9b5-41cfb87917f5"

while True:
    r = requests.post(
        f"{BASE_URL}/api/v1/devices/{DEVICE_ID}/heartbeat",
        json={
            "firmware_version": "1.0.0",
            "ip_address": "127.0.0.1"
        }
    )
    print("Heartbeat:", r.status_code)
    time.sleep(10)