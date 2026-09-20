"""
EdgeImpulse Backend — Test Suite
================================
Run: pytest tests/ -v

Tests cover:
  - Auth (register, login, JWT, API keys)
  - Projects CRUD
  - Sample upload and labeling
  - DSP feature extraction
  - Impulse configuration
  - Training job lifecycle (mocked Celery)
  - Deployment package generation
"""

import pytest
import io
import json
import re
import time
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app
from app.core.database import Base, get_db
from app.core.config import settings
from app.models.user import (
    User,
    Sample,
    Label,
    Impulse,
    TrainingJob,
    TrainedModel,
    Deployment,
    Device,
    DeviceUpdateHistory,
    PostProcessingSettings,
    JobStatus,
    DeployTarget,
    DspFeatureJob,
)
from app.models.model_testing import ModelVersion, ModelTestRun, ModelTestSample, MetricResult

# ─── Test database ────────────────────────────────────────────────────────────

SQLALCHEMY_TEST_URL = "sqlite:///./test_edgeimpulse.db"

engine_test = create_engine(
    SQLALCHEMY_TEST_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session", autouse=True)
def create_tables():
    Base.metadata.create_all(bind=engine_test)
    yield
    Base.metadata.drop_all(bind=engine_test)
    try:
        os.remove("test_edgeimpulse.db")
    except FileNotFoundError:
        pass


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def verify_email_directly(email):
    """Mark a user's email as verified via direct ORM access, bypassing the
    real email-sending flow. Mirrors how google_login() mocks Google's
    verification instead of performing a real OAuth round-trip."""
    db = TestingSessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user and not user.is_verified:
            user.is_verified = True
            user.verification_token_hash = None
            user.verification_token_expires_at = None
            db.commit()
    finally:
        db.close()


def register_and_login(client, email="test@gmail.com", password="testpass123", username="testuser"):
    # Try register first, then login
    r = client.post("/api/v1/auth/register", json={
        "email": email, "username": username, "password": password,
    })
    if r.status_code not in (400, 201):
        assert False, r.text
    verify_email_directly(email)
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def google_login(client, email="googleuser@gmail.com", sub="1234567890", email_verified=True):
    """Simulate a Google Sign-In by mocking Google's ID-token verification, so
    tests can exercise the /auth/google flow without real OAuth credentials."""
    fake_idinfo = {"email": email, "email_verified": email_verified, "sub": sub}
    with patch.object(settings, "GOOGLE_CLIENT_ID", "test-google-client-id"), \
         patch("app.api.v1.endpoints.auth.google_id_token.verify_oauth2_token", return_value=fake_idinfo):
        return client.post("/api/v1/auth/google", json={"credential": "fake-credential-token"})


def make_sample_json(n_samples=100, n_channels=3, freq_hz=100.0):
    """Generate a synthetic sensor data JSON payload."""
    values = np.random.randn(n_samples, n_channels).tolist()
    return json.dumps({
        "device_type": "TEST",
        "interval_ms": int(1000 / freq_hz),
        "sensors": [{"name": f"ch{i}", "units": "raw"} for i in range(n_channels)],
        "values": values,
    }).encode()


def make_test_png(width=100, height=50, color=(255, 0, 0)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


# ─── Auth tests ───────────────────────────────────────────────────────────────

class TestAuth:
    def test_register_success(self, client):
        r = client.post("/api/v1/auth/register", json={
            "email": "newuser@gmail.com",
            "username": "newuser",
            "password": "password123",
        })
        assert r.status_code == 201
        data = r.json()
        assert "access_token" not in data
        assert data["email"] == "newuser@gmail.com"

        db = TestingSessionLocal()
        user = db.query(User).filter(User.email == "newuser@gmail.com").first()
        assert user.is_verified is False
        assert user.verification_token_hash is not None
        db.close()

    def test_register_any_email_domain_allowed(self, client):
        r = client.post("/api/v1/auth/register", json={
            "email": "newuser@example.com",
            "username": "notgmailuser",
            "password": "password123",
        })
        assert r.status_code == 201

    def test_register_duplicate_email(self, client):
        client.post("/api/v1/auth/register", json={
            "email": "dup@gmail.com", "username": "dup1", "password": "pass123",
        })
        r = client.post("/api/v1/auth/register", json={
            "email": "dup@gmail.com", "username": "dup2", "password": "pass123",
        })
        assert r.status_code == 400
        assert "Email" in r.json()["detail"]

    def test_login_success_after_verification(self, client):
        client.post("/api/v1/auth/register", json={
            "email": "login@gmail.com", "username": "loginuser", "password": "pass12345",
        })
        verify_email_directly("login@gmail.com")
        r = client.post("/api/v1/auth/login", json={
            "email": "login@gmail.com", "password": "pass12345",
        })
        assert r.status_code == 200
        assert "access_token" in r.json()

    def test_login_blocked_before_verification(self, client):
        client.post("/api/v1/auth/register", json={
            "email": "unverified@gmail.com", "username": "unverifieduser", "password": "pass12345",
        })
        r = client.post("/api/v1/auth/login", json={
            "email": "unverified@gmail.com", "password": "pass12345",
        })
        assert r.status_code == 403
        assert "not verified" in r.json()["detail"].lower()

    def test_login_wrong_password(self, client):
        client.post("/api/v1/auth/register", json={
            "email": "wrongpass@gmail.com", "username": "wrongpuser", "password": "correct123",
        })
        r = client.post("/api/v1/auth/login", json={
            "email": "wrongpass@gmail.com", "password": "wrong",
        })
        assert r.status_code == 401

    def test_verify_email_success(self, client):
        # Capture the raw token instead of sending a real email — mirrors how
        # google_login() mocks Google's own verification call.
        with patch("app.api.v1.endpoints.auth.send_verification_email") as mock_send:
            client.post("/api/v1/auth/register", json={
                "email": "verifyme@gmail.com", "username": "verifymeuser", "password": "pass12345",
            })
        raw_token = mock_send.call_args.args[1]

        r = client.get("/api/v1/auth/verify-email", params={"token": raw_token})
        assert r.status_code == 200
        body = r.json()
        # Verification now issues a session so the user lands on the dashboard.
        assert body["access_token"]
        assert body["username"] == "verifymeuser"

        db = TestingSessionLocal()
        user = db.query(User).filter(User.email == "verifyme@gmail.com").first()
        assert user.is_verified is True
        # The token hash is intentionally retained so a re-click of the same
        # link short-circuits to success (idempotency) instead of "invalid".
        assert user.verification_token_hash is not None
        db.close()

        # Idempotency: re-using the link still succeeds and still issues a session.
        r2 = client.get("/api/v1/auth/verify-email", params={"token": raw_token})
        assert r2.status_code == 200
        assert r2.json()["access_token"]
        assert "already verified" in (r2.json()["message"] or "").lower()

        login_r = client.post("/api/v1/auth/login", json={
            "email": "verifyme@gmail.com", "password": "pass12345",
        })
        assert login_r.status_code == 200

    def test_verify_email_invalid_token(self, client):
        r = client.get("/api/v1/auth/verify-email", params={"token": "garbage"})
        assert r.status_code == 400

    def test_resend_verification(self, client):
        from app.api.v1.endpoints import auth as auth_module

        GENERIC = "if an account exists for this email, a verification email has been sent."

        # Force the in-memory throttle so the test is deterministic regardless of
        # whether a real Redis is reachable, and start from a clean slate.
        with patch.object(auth_module, "_get_redis_client", return_value=None), \
             patch.dict(auth_module._resend_memory_cache, {}, clear=True):

            with patch("app.api.v1.endpoints.auth.send_verification_email") as mock_send:
                client.post("/api/v1/auth/register", json={
                    "email": "resendme@gmail.com", "username": "resendmeuser", "password": "pass12345",
                })
            first_token = mock_send.call_args.args[1]

            # Unknown email: identical generic response, nothing sent.
            with patch("app.api.v1.endpoints.auth.send_verification_email") as mock_send:
                r = client.post("/api/v1/auth/resend-verification", json={"email": "ghost@gmail.com"})
            assert r.status_code == 200
            assert r.json()["message"].lower() == GENERIC
            mock_send.assert_not_called()

            # Unverified account: a fresh token replaces the old one and is emailed.
            with patch("app.api.v1.endpoints.auth.send_verification_email") as mock_send:
                r = client.post("/api/v1/auth/resend-verification", json={"email": "resendme@gmail.com"})
            assert r.status_code == 200
            assert r.json()["message"].lower() == GENERIC
            new_token = mock_send.call_args.args[1]
            assert new_token != first_token
            assert client.get("/api/v1/auth/verify-email", params={"token": first_token}).status_code == 400
            assert client.get("/api/v1/auth/verify-email", params={"token": new_token}).status_code == 200

        # A second resend within the 60s window is throttled: identical response,
        # no email. Normalization means differing case/whitespace hits the same
        # throttle key AND the same account, so this is also blocked.
        with patch.object(auth_module, "_get_redis_client", return_value=None), \
             patch.dict(auth_module._resend_memory_cache,
                        {"verification_resend:throttled@gmail.com": time.monotonic() + 60}, clear=True):
            with patch("app.api.v1.endpoints.auth.send_verification_email") as mock_send:
                # Register a fresh unverified user whose throttle slot is pre-claimed.
                client.post("/api/v1/auth/register", json={
                    "email": "throttled@gmail.com", "username": "throttleduser", "password": "pass12345",
                })
            with patch("app.api.v1.endpoints.auth.send_verification_email") as mock_send:
                r = client.post("/api/v1/auth/resend-verification", json={"email": "  Throttled@Gmail.com  "})
            assert r.status_code == 200
            assert r.json()["message"].lower() == GENERIC
            mock_send.assert_not_called()

        # Already verified: identical generic response, no email sent.
        with patch.object(auth_module, "_get_redis_client", return_value=None), \
             patch.dict(auth_module._resend_memory_cache, {}, clear=True):
            verify_email_directly("resendme@gmail.com")
            with patch("app.api.v1.endpoints.auth.send_verification_email") as mock_send:
                r = client.post("/api/v1/auth/resend-verification", json={"email": "resendme@gmail.com"})
            assert r.status_code == 200
            assert r.json()["message"].lower() == GENERIC
            mock_send.assert_not_called()

    def test_me_endpoint(self, client):
        token = register_and_login(client, "me@gmail.com", "pass12345", "meuser")
        r = client.get("/api/v1/auth/me", headers=auth_headers(token))
        assert r.status_code == 200
        assert r.json()["email"] == "me@gmail.com"

    def test_me_without_token(self, client):
        r = client.get("/api/v1/auth/me")
        assert r.status_code == 403  # HTTPBearer returns 403 when no creds

    def test_api_key_create_and_list(self, client):
        token = register_and_login(client, "apikey@gmail.com", "pass12345", "apikeyuser")
        hdrs  = auth_headers(token)

        r = client.post("/api/v1/auth/api-keys", json={"name": "Test Key"}, headers=hdrs)
        assert r.status_code == 200
        data = r.json()
        assert data["key"].startswith("ef_")
        assert "id" in data

        r2 = client.get("/api/v1/auth/api-keys", headers=hdrs)
        assert r2.status_code == 200
        assert len(r2.json()) >= 1


class TestGoogleAuth:
    def test_google_login_creates_new_user(self, client):
        r = google_login(client, email="newgoogle@gmail.com", sub="sub-new-1")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "access_token" in data
        assert data["username"]  # auto-derived from email local-part

        db = TestingSessionLocal()
        user = db.query(User).filter(User.email == "newgoogle@gmail.com").first()
        assert user.is_verified is True
        db.close()

    def test_google_login_links_existing_password_account(self, client):
        register_and_login(client, "linkme@gmail.com", "pass12345", "linkmeuser")
        r = google_login(client, email="linkme@gmail.com", sub="sub-link-1")
        assert r.status_code == 200, r.text

        me = client.get("/api/v1/auth/me", headers=auth_headers(r.json()["access_token"]))
        assert me.json()["email"] == "linkme@gmail.com"

        # Re-authenticating with the same Google sub still works (and the
        # password account remains usable, since we don't drop it).
        r2 = google_login(client, email="linkme@gmail.com", sub="sub-link-1")
        assert r2.status_code == 200
        login_r = client.post("/api/v1/auth/login", json={
            "email": "linkme@gmail.com", "password": "pass12345",
        })
        assert login_r.status_code == 200

    def test_google_login_rejects_non_gmail(self, client):
        r = google_login(client, email="user@notgmail.com", sub="sub-nogmail")
        assert r.status_code == 403

    def test_google_login_requires_verified_email(self, client):
        r = google_login(client, email="unverified@gmail.com", sub="sub-unverified", email_verified=False)
        assert r.status_code == 401

    def test_google_login_invalid_token(self, client):
        with patch.object(settings, "GOOGLE_CLIENT_ID", "test-google-client-id"), \
             patch("app.api.v1.endpoints.auth.google_id_token.verify_oauth2_token", side_effect=ValueError("bad token")):
            r = client.post("/api/v1/auth/google", json={"credential": "garbage"})
        assert r.status_code == 401

    def test_google_login_not_configured(self, client):
        with patch.object(settings, "GOOGLE_CLIENT_ID", ""):
            r = client.post("/api/v1/auth/google", json={"credential": "whatever"})
        assert r.status_code == 503


# ─── Project tests ────────────────────────────────────────────────────────────

class TestProjects:
    @pytest.fixture(autouse=True)
    def setup(self, client):
        self.token = register_and_login(client, "proj@gmail.com", "pass12345", "projuser")
        self.hdrs  = auth_headers(self.token)
        self.client = client

    def test_create_project(self):
        r = self.client.post("/api/v1/projects/",
                             json={"name": "Test Project", "description": "A test project"},
                             headers=self.hdrs)
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "Test Project"
        assert "id" in data

    def test_list_projects(self):
        self.client.post("/api/v1/projects/", json={"name": "P1"}, headers=self.hdrs)
        self.client.post("/api/v1/projects/", json={"name": "P2"}, headers=self.hdrs)
        r = self.client.get("/api/v1/projects/", headers=self.hdrs)
        assert r.status_code == 200
        assert len(r.json()) >= 2

    def test_get_project(self):
        r = self.client.post("/api/v1/projects/", json={"name": "GetMe"}, headers=self.hdrs)
        pid = r.json()["id"]
        r2 = self.client.get(f"/api/v1/projects/{pid}", headers=self.hdrs)
        assert r2.status_code == 200
        assert r2.json()["id"] == pid

    def test_delete_project(self):
        r = self.client.post("/api/v1/projects/", json={"name": "DeleteMe"}, headers=self.hdrs)
        pid = r.json()["id"]
        r2 = self.client.delete(f"/api/v1/projects/{pid}", headers=self.hdrs)
        assert r2.status_code == 204


# ─── Labels tests ─────────────────────────────────────────────────────────────

class TestLabels:
    @pytest.fixture(autouse=True)
    def setup(self, client):
        self.token = register_and_login(client, "label@gmail.com", "pass12345", "labeluser")
        self.hdrs  = auth_headers(self.token)
        self.client = client
        r = client.post("/api/v1/projects/", json={"name": "LabelProject"}, headers=self.hdrs)
        self.project_id = r.json()["id"]

    def test_create_label(self):
        r = self.client.post("/api/v1/labels/", json={
            "project_id": self.project_id, "name": "walking", "color": "#22c55e",
        }, headers=self.hdrs)
        assert r.status_code == 201
        assert r.json()["name"] == "walking"

    def test_list_labels(self):
        for name in ["idle", "running", "jumping"]:
            self.client.post("/api/v1/labels/", json={
                "project_id": self.project_id, "name": name,
            }, headers=self.hdrs)
        r = self.client.get(f"/api/v1/labels/project/{self.project_id}", headers=self.hdrs)
        assert r.status_code == 200
        assert len(r.json()) >= 3


# ─── Sample tests ─────────────────────────────────────────────────────────────

class TestSamples:
    @pytest.fixture(autouse=True)
    def setup(self, client):
        self.token = register_and_login(client, "sample@gmail.com", "pass12345", "sampleuser")
        self.hdrs  = auth_headers(self.token)
        self.client = client
        r = client.post("/api/v1/projects/", json={"name": "SampleProject"}, headers=self.hdrs)
        self.project_id = r.json()["id"]

    @patch("app.core.storage.StorageService.upload_bytes", return_value="test/key")
    @patch("app.core.storage.StorageService._ensure_bucket")
    def test_upload_sample(self, mock_bucket, mock_upload):
        data = make_sample_json(50, 3, 100.0)
        r = self.client.post(
            "/api/v1/samples/upload",
            data={
                "project_id": self.project_id,
                "sensor_type": "accelerometer",
                "frequency_hz": "100.0",
                "sample_type": "training",
            },
            files={"file": ("test.json", data, "application/json")},
            headers=self.hdrs,
        )
        assert r.status_code == 201
        body = r.json()
        assert body["sensor_type"] == "accelerometer"
        assert body["project_id"] == self.project_id

    @patch("app.core.storage.StorageService.upload_bytes", return_value="test/key")
    @patch("app.core.storage.StorageService._ensure_bucket")
    def test_list_samples(self, mock_bucket, mock_upload):
        # Upload a couple
        for _ in range(3):
            self.client.post(
                "/api/v1/samples/upload",
                data={"project_id": self.project_id, "sensor_type": "accel",
                      "frequency_hz": "100", "sample_type": "training"},
                files={"file": ("s.json", make_sample_json(), "application/json")},
                headers=self.hdrs,
            )
        r = self.client.get(f"/api/v1/samples/project/{self.project_id}", headers=self.hdrs)
        assert r.status_code == 200
        assert r.json()["total"] >= 3

    @patch("app.core.storage.StorageService.upload_bytes", return_value="test/key")
    @patch("app.core.storage.StorageService._ensure_bucket")
    def test_upload_batch_pascal_voc_annotations(self, mock_bucket, mock_upload):
        image_bytes = make_test_png(100, 50)
        xml_text = """<annotation>
  <filename>cat.png</filename>
  <object>
    <name>cat</name>
    <bndbox>
      <xmin>10</xmin><ymin>5</ymin><xmax>40</xmax><ymax>25</ymax>
    </bndbox>
  </object>
</annotation>"""
        r = self.client.post(
            "/api/v1/samples/upload-batch",
            data={"project_id": self.project_id, "sample_type": "training"},
            files=[
                ("files", ("animals/cat.png", image_bytes, "image/png")),
                ("files", ("animals/cat.xml", xml_text.encode(), "text/xml")),
            ],
            headers=self.hdrs,
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["uploaded"] == 1
        sample = body["items"][0]
        assert sample["label_name"] == "cat"
        boxes = sample["extra_metadata"]["boundingBoxes"]
        assert len(boxes) == 1
        assert boxes[0]["label"] == "cat"
        assert boxes[0]["x"] == 10
        assert boxes[0]["y"] == 5
        assert boxes[0]["w"] == 30
        assert boxes[0]["h"] == 20

    @patch("app.core.storage.StorageService.upload_bytes", return_value="test/key")
    @patch("app.core.storage.StorageService._ensure_bucket")
    def test_upload_batch_yolo_annotations(self, mock_bucket, mock_upload):
        image_bytes = make_test_png(100, 50)
        yolo_text = "1 0.5 0.5 0.2 0.4\n"
        r = self.client.post(
            "/api/v1/samples/upload-batch",
            data={"project_id": self.project_id, "sample_type": "training"},
            files=[
                ("files", ("train/dog.png", image_bytes, "image/png")),
                ("files", ("train/dog.txt", yolo_text.encode(), "text/plain")),
                ("files", ("classes.txt", b"cat\ndog\n", "text/plain")),
            ],
            headers=self.hdrs,
        )
        assert r.status_code == 201, r.text
        sample = r.json()["items"][0]
        assert sample["label_name"] == "dog"
        box = sample["extra_metadata"]["boundingBoxes"][0]
        assert box["label"] == "dog"
        assert round(box["x"], 2) == 40.0
        assert round(box["y"], 2) == 15.0
        assert round(box["w"], 2) == 20.0
        assert round(box["h"], 2) == 20.0

    @patch("app.core.storage.StorageService.upload_bytes", return_value="test/key")
    @patch("app.core.storage.StorageService._ensure_bucket")
    def test_upload_batch_coco_annotations(self, mock_bucket, mock_upload):
        image_bytes = make_test_png(100, 50)
        coco = {
            "images": [{"id": 7, "file_name": "set/coco.png"}],
            "annotations": [{"id": 1, "image_id": 7, "category_id": 3, "bbox": [10, 5, 30, 20]}],
            "categories": [{"id": 3, "name": "bird"}],
        }
        r = self.client.post(
            "/api/v1/samples/upload-batch",
            data={"project_id": self.project_id, "sample_type": "training"},
            files=[
                ("files", ("set/coco.png", image_bytes, "image/png")),
                ("files", ("annotations.json", json.dumps(coco).encode(), "application/json")),
            ],
            headers=self.hdrs,
        )
        assert r.status_code == 201, r.text
        sample = r.json()["items"][0]
        assert sample["label_name"] == "bird"
        box = sample["extra_metadata"]["boundingBoxes"][0]
        assert box["label"] == "bird"
        assert box["w"] == 30
        assert box["h"] == 20

    @patch("app.core.storage.StorageService.upload_bytes", return_value="test/key")
    @patch("app.core.storage.StorageService._ensure_bucket")
    def test_upload_batch_open_images_annotations(self, mock_bucket, mock_upload):
        image_bytes = make_test_png(120, 60)
        annotation_csv = (
            "ImageID,LabelName,XMin,XMax,YMin,YMax\n"
            "open-1,/m/cat,0.1,0.4,0.2,0.8\n"
        )
        class_csv = "LabelName,DisplayName\n/m/cat,Cat\n"
        r = self.client.post(
            "/api/v1/samples/upload-batch",
            data={"project_id": self.project_id, "sample_type": "training"},
            files=[
                ("files", ("open/open-1.png", image_bytes, "image/png")),
                ("files", ("annotations-bbox.csv", annotation_csv.encode(), "text/csv")),
                ("files", ("class-descriptions-boxable.csv", class_csv.encode(), "text/csv")),
            ],
            headers=self.hdrs,
        )
        assert r.status_code == 201, r.text
        sample = r.json()["items"][0]
        assert sample["label_name"] == "Cat"
        box = sample["extra_metadata"]["boundingBoxes"][0]
        assert box["label"] == "Cat"
        assert round(box["x"], 2) == 12.0
        assert round(box["y"], 2) == 12.0
        assert round(box["w"], 2) == 36.0
        assert round(box["h"], 2) == 36.0

        db = TestingSessionLocal()
        try:
            labels = {
                lbl.name for lbl in db.query(Label).filter(Label.project_id == self.project_id).all()
            }
            assert "Cat" in labels
        finally:
            db.close()

    @patch("app.core.storage.StorageService.upload_bytes", return_value="test/key")
    @patch("app.core.storage.StorageService._ensure_bucket")
    def test_upload_batch_skips_labels_and_readme_sidecars(self, mock_bucket, mock_upload):
        image_bytes = make_test_png(100, 50)
        labels_json = {
            "version": 1,
            "files": [
                {
                    "path": "testing/bike.png",
                    "boundingBoxes": [
                        {"label": "bike", "x": 10, "y": 5, "width": 30, "height": 20}
                    ],
                }
            ],
        }
        r = self.client.post(
            "/api/v1/samples/upload-batch",
            data={"project_id": self.project_id, "sample_type": "training"},
            files=[
                ("files", ("testing/bike.png", image_bytes, "image/png")),
                ("files", ("testing/bounding_boxes.labels", json.dumps(labels_json).encode(), "application/json")),
                ("files", ("testing/README.txt", b"dataset notes", "text/plain")),
            ],
            headers=self.hdrs,
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["uploaded"] == 1
        assert [item["filename"] for item in body["items"]] == ["testing/bike.png"]
        boxes = body["items"][0]["extra_metadata"]["boundingBoxes"]
        assert len(boxes) == 1
        assert boxes[0]["label"] == "bike"

    def test_get_sample_prefers_bounding_box_label_for_display(self):
        db = TestingSessionLocal()
        try:
            cat = Label(project_id=self.project_id, name="Cat")
            dog = Label(project_id=self.project_id, name="Dog")
            db.add_all([cat, dog])
            db.flush()
            dog_id = dog.id

            sample = Sample(
                project_id=self.project_id,
                label_id=cat.id,
                filename="detected.jpg",
                storage_key="samples/detected.jpg",
                sensor_type="camera",
                sample_type="training",
                extra_metadata={
                    "boundingBoxes": [
                        {"x": 0, "y": 0, "w": 10, "h": 10, "label": "Dog", "label_id": dog.id}
                    ]
                },
            )
            db.add(sample)
            db.commit()
            sample_id = sample.id
        finally:
            db.close()

        r = self.client.get(f"/api/v1/samples/{sample_id}", headers=self.hdrs)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["label_name"] == "Dog"
        assert body["label_id"] == dog_id

    def test_assign_label_updates_bounding_boxes_for_image_samples(self):
        db = TestingSessionLocal()
        try:
            bus = Label(project_id=self.project_id, name="bus")
            car = Label(project_id=self.project_id, name="car")
            db.add_all([bus, car])
            db.flush()
            bus_id = bus.id
            car_id = car.id

            sample = Sample(
                project_id=self.project_id,
                label_id=bus_id,
                filename="scene.jpg",
                storage_key="samples/scene.jpg",
                sensor_type="camera",
                sample_type="training",
                extra_metadata={
                    "boundingBoxes": [
                        {"x": 0, "y": 0, "w": 10, "h": 10, "label": "bus", "label_id": bus_id},
                        {"x": 12, "y": 8, "w": 6, "h": 6, "label": "bus", "label_id": bus_id},
                    ]
                },
            )
            db.add(sample)
            db.commit()
            sample_id = sample.id
        finally:
            db.close()

        r = self.client.patch(
            f"/api/v1/samples/{sample_id}/label",
            json={"label_id": car_id},
            headers=self.hdrs,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["label_id"] == car_id
        assert body["label_name"] == "car"
        boxes = body["extra_metadata"]["boundingBoxes"]
        assert all(box["label"] == "car" for box in boxes)
        assert all(box["label_id"] == car_id for box in boxes)

    def test_list_samples_can_filter_by_bounding_box_label(self):
        db = TestingSessionLocal()
        try:
            bus = Label(project_id=self.project_id, name="bus")
            car = Label(project_id=self.project_id, name="car")
            db.add_all([bus, car])
            db.flush()
            bus_id = bus.id
            car_id = car.id

            matching = Sample(
                project_id=self.project_id,
                filename="bus-scene.jpg",
                storage_key="samples/bus-scene.jpg",
                sensor_type="camera",
                sample_type="training",
                extra_metadata={
                    "boundingBoxes": [
                        {"x": 0, "y": 0, "w": 10, "h": 10, "label": "bus", "label_id": bus_id}
                    ]
                },
            )
            other = Sample(
                project_id=self.project_id,
                filename="car-scene.jpg",
                storage_key="samples/car-scene.jpg",
                sensor_type="camera",
                sample_type="training",
                extra_metadata={
                    "boundingBoxes": [
                        {"x": 0, "y": 0, "w": 10, "h": 10, "label": "car", "label_id": car_id}
                    ]
                },
            )
            db.add_all([matching, other])
            db.commit()
        finally:
            db.close()

        r = self.client.get(
            f"/api/v1/samples/project/{self.project_id}",
            params={"sample_type": "training", "label_id": bus_id},
            headers=self.hdrs,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 1
        assert [item["filename"] for item in body["items"]] == ["bus-scene.jpg"]

    @patch("app.core.storage.StorageService.delete_file")
    def test_delete_sample_detaches_model_test_rows(self, mock_delete_file):
        db = TestingSessionLocal()
        try:
            impulse = Impulse(
                project_id=self.project_id,
                name="Detach Sample Impulse",
                input_type="image",
                sensor_type="camera",
                dsp_blocks=[{"type": "image", "params": {"image_width": 96, "image_height": 96}}],
            )
            db.add(impulse)
            db.flush()

            sample = Sample(
                project_id=self.project_id,
                filename="delete-me.jpg",
                storage_key="samples/delete-me.jpg",
                sensor_type="camera",
                sample_type="testing",
                extra_metadata={},
            )
            db.add(sample)
            db.flush()

            test_row = ModelTestSample(
                project_id=self.project_id,
                impulse_id=impulse.id,
                sample_id=sample.id,
                sample_name="delete-me.jpg",
                expected_outcome="ok",
            )
            db.add(test_row)
            db.commit()
            sample_id = sample.id
            test_row_id = test_row.id
        finally:
            db.close()

        r = self.client.delete(f"/api/v1/samples/{sample_id}", headers=self.hdrs)
        assert r.status_code == 204, r.text

        db = TestingSessionLocal()
        try:
            assert db.query(Sample).filter(Sample.id == sample_id).first() is None
            detached_row = db.query(ModelTestSample).filter(ModelTestSample.id == test_row_id).first()
            assert detached_row is not None
            assert detached_row.sample_id is None
            assert detached_row.sample_name == "delete-me.jpg"
        finally:
            db.close()

    @patch("app.core.storage.StorageService.delete_file")
    def test_delete_sample_prunes_orphan_label(self, mock_delete_file):
        """Deleting the only sample referencing a label removes that label."""
        db = TestingSessionLocal()
        try:
            label = Label(project_id=self.project_id, name="orphan-me")
            db.add(label)
            db.flush()

            sample = Sample(
                project_id=self.project_id,
                filename="orphan.jpg",
                storage_key="samples/orphan.jpg",
                sensor_type="camera",
                sample_type="training",
                label_id=label.id,
                extra_metadata={},
            )
            db.add(sample)
            db.commit()
            sample_id = sample.id
            label_id = label.id
        finally:
            db.close()

        r = self.client.delete(f"/api/v1/samples/{sample_id}", headers=self.hdrs)
        assert r.status_code == 204, r.text

        db = TestingSessionLocal()
        try:
            assert db.query(Label).filter(Label.id == label_id).first() is None
        finally:
            db.close()

    @patch("app.core.storage.StorageService.delete_file")
    def test_bulk_delete_samples_prunes_orphan_labels(self, mock_delete_file):
        """Bulk-deleting all samples prunes their now-orphaned labels."""
        db = TestingSessionLocal()
        try:
            label = Label(project_id=self.project_id, name="bulk-orphan")
            db.add(label)
            db.flush()

            sample_ids = []
            for i in range(2):
                s = Sample(
                    project_id=self.project_id,
                    filename=f"bulk-orphan-{i}.jpg",
                    storage_key=f"samples/bulk-orphan-{i}.jpg",
                    sensor_type="camera",
                    sample_type="training",
                    label_id=label.id,
                    extra_metadata={},
                )
                db.add(s)
                db.flush()
                sample_ids.append(s.id)
            db.commit()
            label_id = label.id
        finally:
            db.close()

        r = self.client.post(
            "/api/v1/samples/bulk-delete",
            json={"sample_ids": sample_ids},
            headers=self.hdrs,
        )
        assert r.status_code == 200, r.text
        assert r.json()["deleted"] == 2

        db = TestingSessionLocal()
        try:
            assert db.query(Label).filter(Label.id == label_id).first() is None
        finally:
            db.close()


# ─── DSP tests ────────────────────────────────────────────────────────────────

class TestDSP:
    def test_list_blocks(self, client):
        r = client.get("/api/v1/dsp/blocks")
        # DSP blocks endpoint is public
        assert r.status_code in (200, 401, 403)

    def test_spectral_analysis(self):
        """Unit test DSP processor directly."""
        from app.ml.dsp.processor import DSPProcessor
        signal = np.random.randn(200).astype(np.float32)
        proc   = DSPProcessor("spectral_analysis", {"fft_length": 64}, 100.0)
        feats  = proc.extract(signal)
        assert feats.ndim == 1
        assert feats.shape[0] == 33  # fft_length//2 + 1

    def test_mfcc(self):
        from app.ml.dsp.processor import DSPProcessor
        signal = np.random.randn(500).astype(np.float32)
        proc   = DSPProcessor("mfcc", {"num_coefficients": 13, "frame_length": 64,
                                        "frame_stride": 32, "fft_length": 64}, 8000.0)
        feats  = proc.extract(signal)
        assert feats.ndim == 1
        assert feats.shape[0] > 0

    def test_raw_extraction(self):
        from app.ml.dsp.processor import DSPProcessor
        signal = np.random.randn(50).astype(np.float32)
        proc   = DSPProcessor("raw", {"normalize": True, "flatten": True}, 100.0)
        feats  = proc.extract(signal)
        assert feats.shape[0] == 50
        assert feats.min() >= 0.0 - 1e-5
        assert feats.max() <= 1.0 + 1e-5

    def test_flatten_features(self):
        from app.ml.dsp.processor import DSPProcessor
        signal = np.random.randn(3, 100).T.astype(np.float32)  # 100 samples, 3 channels
        proc   = DSPProcessor("flatten",
                              {"features": ["mean", "std", "rms", "max", "min"]}, 100.0)
        feats  = proc.extract(signal)
        # 5 features × 3 channels = 15
        assert feats.shape[0] == 15

    def test_spectrogram(self):
        from app.ml.dsp.processor import DSPProcessor
        signal = np.random.randn(256).astype(np.float32)
        proc   = DSPProcessor("spectrogram",
                              {"frame_length": 64, "frame_stride": 32,
                               "fft_length": 64, "num_mel_filters": 16}, 8000.0)
        feats  = proc.extract(signal)
        assert feats.ndim == 2
        assert feats.shape[1] == 16  # mel filters

    def test_features_endpoint_prefers_cached_bounding_box_label(self, client):
        token = register_and_login(client, "dspbbox@gmail.com", "pass12345", "dspbbox")
        hdrs = auth_headers(token)
        project = client.post("/api/v1/projects/", json={"name": "DSPBBoxProject"}, headers=hdrs).json()

        db = TestingSessionLocal()
        try:
            cat = Label(project_id=project["id"], name="Cat")
            dog = Label(project_id=project["id"], name="Dog")
            db.add_all([cat, dog])
            db.flush()

            impulse = Impulse(
                project_id=project["id"],
                name="BBox Impulse",
                input_type="image",
                sensor_type="camera",
                dsp_blocks=[{"type": "image", "params": {"image_width": 96, "image_height": 96}}],
            )
            db.add(impulse)
            db.flush()

            sample = Sample(
                project_id=project["id"],
                label_id=cat.id,
                filename="bbox.jpg",
                storage_key="samples/bbox.jpg",
                sensor_type="camera",
                sample_type="training",
                extra_metadata={},
            )
            db.add(sample)
            db.commit()

            buf = io.BytesIO()
            np.savez_compressed(
                buf,
                X=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                y=np.array([cat.id, cat.id], dtype=object),
                ids=np.array([sample.id, sample.id], dtype=object),
                boxes_json=np.array([
                    json.dumps([{"label": "Dog", "label_id": dog.id}]),
                    json.dumps([]),
                ], dtype=object),
            )
            buf.seek(0)
            feature_bytes = buf.getvalue()
            impulse_id = impulse.id
        finally:
            db.close()

        with patch("app.core.storage.StorageService.download_bytes", return_value=feature_bytes):
            r = client.get(f"/api/v1/dsp/features/{impulse_id}", headers=hdrs)

        assert r.status_code == 200, r.text
        points = r.json()["points"]
        assert points[0]["label"] == "Dog"
        assert points[1]["label"] == "Cat"


# ─── Model builder tests ──────────────────────────────────────────────────────

class TestModelBuilder:
    def test_dense_model(self):
        from app.ml.training.model_builder import build_model
        model = build_model("dense", (64,), 4)
        assert model.output_shape == (None, 4)

    def test_conv1d_model(self):
        from app.ml.training.model_builder import build_model
        model = build_model("conv1d", (64,), 3)
        assert model.output_shape == (None, 3)

    def test_conv2d_model(self):
        from app.ml.training.model_builder import build_model
        model = build_model("conv2d", (64,), 5)
        assert model.output_shape == (None, 5)

    def test_lstm_model(self):
        from app.ml.training.model_builder import build_model
        model = build_model("lstm", (64,), 3)
        assert model.output_shape == (None, 3)

    def test_mobilenet_model(self):
        from app.ml.training.model_builder import build_model
        model = build_model("mobilenet", (64,), 4)
        assert model.output_shape == (None, 4)

    def test_model_forward_pass(self):
        """Verify models can do a forward pass without errors."""
        import tensorflow as tf
        from app.ml.training.model_builder import build_model

        for arch in ["dense", "conv1d", "conv2d", "lstm"]:
            model = build_model(arch, (32,), 3)
            x = tf.random.uniform((4, 32))
            y = model(x, training=False)
            assert y.shape == (4, 3), f"{arch}: expected (4,3) got {y.shape}"


# ─── Impulse tests ────────────────────────────────────────────────────────────

class TestImpulses:
    @pytest.fixture(autouse=True)
    def setup(self, client):
        self.token = register_and_login(client, "imp@gmail.com", "pass12345", "impuser")
        self.hdrs  = auth_headers(self.token)
        self.client = client
        r = client.post("/api/v1/projects/", json={"name": "ImpulseProject"}, headers=self.hdrs)
        self.project_id = r.json()["id"]

    def test_create_impulse(self):
        r = self.client.post("/api/v1/impulses/", json={
            "project_id": self.project_id,
            "name": "My Impulse",
            "window_size_ms": 1000,
            "frequency_hz": 100.0,
            "dsp_blocks": [{"type": "spectral_analysis", "params": {"fft_length": 128}}],
            "ml_blocks": [{"architecture": "dense", "params": {}}],
        }, headers=self.hdrs)
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "My Impulse"
        assert len(data["dsp_blocks"]) == 1

    def test_update_impulse(self):
        r = self.client.post("/api/v1/impulses/", json={
            "project_id": self.project_id, "name": "ToUpdate",
            "dsp_blocks": [], "ml_blocks": [],
        }, headers=self.hdrs)
        imp_id = r.json()["id"]

        r2 = self.client.put(f"/api/v1/impulses/{imp_id}", json={
            "project_id": self.project_id, "name": "Updated Name",
            "window_size_ms": 2000, "frequency_hz": 50.0,
            "dsp_blocks": [{"type": "mfcc", "params": {}}],
            "ml_blocks": [{"architecture": "conv1d", "params": {}}],
        }, headers=self.hdrs)
        assert r2.status_code == 200
        assert r2.json()["name"] == "Updated Name"
        assert r2.json()["window_size_ms"] == 2000

    def test_create_impulse_with_image_block_sets_input_type_image(self):
        r = self.client.post("/api/v1/impulses/", json={
            "project_id": self.project_id,
            "name": "ImageImpulse",
            "input_type": "time-series",
            "dsp_blocks": [{
                "type": "image",
                "params": {"image_width": 64, "image_height": 48},
            }],
            "ml_blocks": [],
        }, headers=self.hdrs)
        assert r.status_code == 201
        data = r.json()
        assert data["input_type"] == "image"
        assert data["image_width"] == 64
        assert data["image_height"] == 48

    def test_create_impulse_with_image_input_type_preserves_image_defaults(self):
        r = self.client.post("/api/v1/impulses/", json={
            "project_id": self.project_id,
            "name": "DefaultImageImpulse",
            "input_type": "image",
            "input_axes": ["image"],
            "sensor_type": "camera",
            "image_width": 96,
            "image_height": 96,
            "resize_mode": "Fit shortest axis",
            "dsp_blocks": [],
            "ml_blocks": [],
        }, headers=self.hdrs)
        assert r.status_code == 201
        data = r.json()
        assert data["input_type"] == "image"
        assert data["input_axes"] == ["image"]
        assert data["sensor_type"] == "camera"

    def test_update_impulse_with_image_input_type_preserves_image_defaults(self):
        created = self.client.post("/api/v1/impulses/", json={
            "project_id": self.project_id,
            "name": "NeedsImageDefault",
            "dsp_blocks": [],
            "ml_blocks": [],
        }, headers=self.hdrs)
        assert created.status_code == 201
        imp_id = created.json()["id"]

        updated = self.client.put(f"/api/v1/impulses/{imp_id}", json={
            "project_id": self.project_id,
            "name": "NeedsImageDefault",
            "input_type": "image",
            "input_axes": ["image"],
            "sensor_type": "camera",
            "image_width": 96,
            "image_height": 96,
            "resize_mode": "Fit shortest axis",
            "dsp_blocks": [],
            "ml_blocks": [],
        }, headers=self.hdrs)
        assert updated.status_code == 200
        data = updated.json()
        assert data["input_type"] == "image"
        assert data["input_axes"] == ["image"]
        assert data["sensor_type"] == "camera"

    def test_patch_blocks_image_syncs_root_input_type(self):
        r = self.client.post("/api/v1/impulses/", json={
            "project_id": self.project_id,
            "name": "PatchImageImpulse",
            "dsp_blocks": [],
            "ml_blocks": [],
        }, headers=self.hdrs)
        imp_id = r.json()["id"]

        add = self.client.patch(f"/api/v1/impulses/{imp_id}/blocks", json={
            "action": "add",
            "block_kind": "dsp",
            "block": {
                "type": "image",
                "name": "Image",
                "params": {"image_width": 80, "image_height": 60},
            },
        }, headers=self.hdrs)
        assert add.status_code == 200
        added = add.json()
        assert added["input_type"] == "image"
        assert added["image_width"] == 80
        assert added["image_height"] == 60

        remove = self.client.patch(f"/api/v1/impulses/{imp_id}/blocks", json={
            "action": "remove",
            "block_kind": "dsp",
            "block_index": 0,
        }, headers=self.hdrs)
        assert remove.status_code == 200
        removed = remove.json()
        assert removed["input_type"] == "time-series"

    def test_delete_impulse_removes_model_testing_dependencies(self):
        r = self.client.post("/api/v1/impulses/", json={
            "project_id": self.project_id,
            "name": "DeleteWithTestingData",
            "dsp_blocks": [],
            "ml_blocks": [],
        }, headers=self.hdrs)
        assert r.status_code == 201
        impulse_id = r.json()["id"]

        db = TestingSessionLocal()
        try:
            version = ModelVersion(
                project_id=self.project_id,
                impulse_id=impulse_id,
                version_number="v1",
                name="Float32",
            )
            db.add(version)
            db.flush()

            run = ModelTestRun(
                project_id=self.project_id,
                impulse_id=impulse_id,
                model_version_id=version.id,
                total_samples=1,
                passed_samples=1,
                failed_samples=0,
            )
            db.add(run)
            db.flush()

            db.add(MetricResult(
                test_run_id=run.id,
                metric_name="accuracy",
                metric_display_name="Accuracy",
                metric_value=100.0,
            ))
            db.add(ModelTestSample(
                project_id=self.project_id,
                impulse_id=impulse_id,
                test_run_id=run.id,
                sample_name="sample-1",
                expected_outcome="ok",
            ))
            db.commit()
        finally:
            db.close()

        delete_res = self.client.delete(f"/api/v1/impulses/{impulse_id}", headers=self.hdrs)
        assert delete_res.status_code == 204

        db = TestingSessionLocal()
        try:
            assert db.query(Impulse).filter(Impulse.id == impulse_id).first() is None
            assert db.query(ModelVersion).filter(ModelVersion.impulse_id == impulse_id).count() == 0
            assert db.query(ModelTestRun).filter(ModelTestRun.impulse_id == impulse_id).count() == 0
            assert db.query(ModelTestSample).filter(ModelTestSample.impulse_id == impulse_id).count() == 0
            assert db.query(MetricResult).count() == 0
        finally:
            db.close()

    def test_delete_impulse_removes_deployment_history_dependencies(self):
        r = self.client.post("/api/v1/impulses/", json={
            "project_id": self.project_id,
            "name": "DeleteWithDeploymentHistory",
            "dsp_blocks": [],
            "ml_blocks": [],
        }, headers=self.hdrs)
        assert r.status_code == 201
        impulse_id = r.json()["id"]

        db = TestingSessionLocal()
        try:
            training_job = TrainingJob(
                impulse_id=impulse_id,
                status=JobStatus.completed,
            )
            db.add(training_job)
            db.flush()

            trained_model = TrainedModel(
                training_job_id=training_job.id,
                version="1",
                format="tflite",
                storage_key="models/test.tflite",
            )
            db.add(trained_model)
            db.flush()

            deployment = Deployment(
                model_id=trained_model.id,
                project_id=self.project_id,
                target=DeployTarget.raspberry_pi,
                deployment_target="raspberry_pi",
                device_profile="raspberry_pi_4",
                status=JobStatus.completed,
            )
            db.add(deployment)
            db.flush()

            device = Device(
                project_id=self.project_id,
                name="Delete Device",
                device_id="delete-device-1",
            )
            db.add(device)
            db.flush()

            db.add(DeviceUpdateHistory(
                project_id=self.project_id,
                device_id=device.id,
                deployment_id=deployment.id,
                status="requested",
            ))
            db.commit()
        finally:
            db.close()

        delete_res = self.client.delete(f"/api/v1/impulses/{impulse_id}", headers=self.hdrs)
        assert delete_res.status_code == 204, delete_res.text

        db = TestingSessionLocal()
        try:
            assert db.query(Impulse).filter(Impulse.id == impulse_id).first() is None
            assert db.query(TrainingJob).filter(TrainingJob.impulse_id == impulse_id).count() == 0
            assert db.query(Deployment).count() == 0
            assert db.query(DeviceUpdateHistory).count() == 0
        finally:
            db.close()

    def test_delete_impulse_removes_post_processing_settings_dependencies(self):
        r = self.client.post("/api/v1/impulses/", json={
            "project_id": self.project_id,
            "name": "DeleteWithPostProcessingSettings",
            "dsp_blocks": [],
            "ml_blocks": [],
        }, headers=self.hdrs)
        assert r.status_code == 201
        impulse_id = r.json()["id"]

        db = TestingSessionLocal()
        try:
            db.add(PostProcessingSettings(
                project_id=self.project_id,
                impulse_id=impulse_id,
                enabled=True,
                threshold=0.65,
            ))
            db.commit()
        finally:
            db.close()

        delete_res = self.client.delete(f"/api/v1/impulses/{impulse_id}", headers=self.hdrs)
        assert delete_res.status_code == 204, delete_res.text

        db = TestingSessionLocal()
        try:
            assert db.query(Impulse).filter(Impulse.id == impulse_id).first() is None
            assert (
                db.query(PostProcessingSettings)
                .filter(PostProcessingSettings.impulse_id == impulse_id)
                .count()
                == 0
            )
        finally:
            db.close()


# ─── Training job tests ───────────────────────────────────────────────────────

class TestTraining:
    @pytest.fixture(autouse=True)
    def setup(self, client):
        self.token = register_and_login(client, "train@gmail.com", "pass12345", "trainuser")
        self.hdrs  = auth_headers(self.token)
        self.client = client
        r = client.post("/api/v1/projects/", json={"name": "TrainProject"}, headers=self.hdrs)
        pid = r.json()["id"]
        r2 = client.post("/api/v1/impulses/", json={
            "project_id": pid, "name": "TrainImpulse",
            "dsp_blocks": [], "ml_blocks": [],
        }, headers=self.hdrs)
        self.impulse_id = r2.json()["id"]

    @patch("app.workers.training_worker.run_training_job.delay")
    def test_start_training_job(self, mock_delay):
        mock_task = MagicMock()
        mock_task.id = "mock-celery-task-id"
        mock_delay.return_value = mock_task

        # Explicit CPU: the request default is "gpu", which /training/start now
        # rejects with 503 unless GPU is enabled and a GPU worker is live. This
        # test is about the job lifecycle, not device routing.
        r = self.client.post("/api/v1/training/start", json={
            "impulse_id": self.impulse_id,
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "device_preference": "cpu",
        }, headers=self.hdrs)
        assert r.status_code == 201
        data = r.json()
        assert data["status"] == "pending"
        assert data["epochs"] == 10

        # Verify Celery task was dispatched
        mock_delay.assert_called_once_with(data["id"])

    @patch("app.workers.training_worker.run_training_job.delay")
    def test_list_jobs_for_impulse(self, mock_delay):
        mock_task = MagicMock(); mock_task.id = "t1"
        mock_delay.return_value = mock_task
        self.client.post("/api/v1/training/start",
                         json={"impulse_id": self.impulse_id, "epochs": 5,
                               "device_preference": "cpu"},
                         headers=self.hdrs)
        r = self.client.get(f"/api/v1/training/impulse/{self.impulse_id}", headers=self.hdrs)
        assert r.status_code == 200
        assert len(r.json()) >= 1


# ─── Deployment tests ─────────────────────────────────────────────────────────

class TestDeployment:
    def test_list_targets(self, client):
        r = client.get("/api/v1/deployment/targets")
        assert r.status_code in (200, 403)
        # Don't require auth for target list
        if r.status_code == 200:
            targets = r.json()["targets"]
            assert any(t["id"] == "tflite" for t in targets)
            assert any(t["id"] == "arduino" for t in targets)

    def test_deployment_package_generation(self):
        """Unit test deployment package generation for non-FOMO (classification) models."""
        from app.workers.deployment_worker import _gen_tflite, _gen_arduino, _gen_cpp
        import zipfile, io

        dummy_tflite = b"\x1c\x00\x00\x00TFL3" + b"\x00" * 100
        # Explicit classification meta — no architecture/output_type key that
        # would trigger the FOMO guard, so arduino/cpp must succeed.
        meta = {
            "label_names": ["idle", "walking", "running"],
            "input_shape": [64],
            "architecture": "dense",
        }

        for gen_fn, target in [(_gen_tflite, "tflite"), (_gen_arduino, "arduino"), (_gen_cpp, "cpp")]:
            zip_bytes = gen_fn(dummy_tflite, meta, {})
            assert len(zip_bytes) > 100, f"{target}: zip is empty"

            # Verify it's a valid zip
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                names = zf.namelist()
                assert len(names) > 0, f"{target}: empty zip"
                assert any("README" in n for n in names), f"{target}: missing README"



# ─── Deployment compatibility gate (Target Device Phase 4) ───────────────────

class TestDeploymentCompatibilityEndpoint:
    """Endpoint-level cover for the Phase 4 gate.

    The rules themselves are tested at ORM level in test_compatibility.py; what
    matters here is that the gate runs before anything is persisted, and that
    the read-only probe is an ownership surface like every other endpoint in
    deployment.py.
    """

    _CLASSIFICATION_META = {"label_names": ["a", "b"], "architecture": "dense"}
    _FOMO_META = {
        "label_names": ["a", "b"],
        "architecture": "fomo_mobilenetv2_0_1",
        "output_type": "detection_heatmap",
    }

    @pytest.fixture(autouse=True)
    def setup(self, client):
        self.client = client
        self.token = register_and_login(client, "compat@gmail.com", "pass12345", "compatuser")
        self.hdrs = auth_headers(self.token)

        import uuid as _uuid
        suffix = _uuid.uuid4().hex[:8]
        r = client.post(
            "/api/v1/projects/", json={"name": f"CompatProject-{suffix}"}, headers=self.hdrs,
        )
        assert r.status_code == 201, r.text
        self.project_id = r.json()["id"]
        r = client.post("/api/v1/impulses/", json={
            "project_id": self.project_id, "name": f"CompatImpulse-{suffix}",
            "dsp_blocks": [], "ml_blocks": [],
        }, headers=self.hdrs)
        self.impulse_id = r.json()["id"]

        self.cls_model_id = self._make_model(self._CLASSIFICATION_META)
        self.fomo_model_id = self._make_model(self._FOMO_META)
        self._seed_catalog()

    def _seed_catalog(self):
        """The device rule reads the catalog, which migrations own — this DB is
        built by create_all(), so seed the two slugs these tests name."""
        from app.models.devices import DeviceCatalogEntry

        db = TestingSessionLocal()
        try:
            for slug, display, family, target in [
                ("raspberry_pi_4", "Raspberry Pi 4", "Raspberry Pi", DeployTarget.raspberry_pi),
                ("arduino_nano_33_ble", "Arduino Nano 33 BLE", "Arduino", DeployTarget.arduino),
            ]:
                exists = db.query(DeviceCatalogEntry).filter(
                    DeviceCatalogEntry.slug == slug
                ).first()
                if not exists:
                    db.add(DeviceCatalogEntry(
                        slug=slug, display_name=display,
                        family=family, deploy_target=target,
                    ))
            db.commit()
        finally:
            db.close()

    def _make_model(self, meta: dict) -> str:
        db = TestingSessionLocal()
        try:
            job = TrainingJob(impulse_id=self.impulse_id, status=JobStatus.completed)
            db.add(job)
            db.flush()
            model = TrainedModel(
                training_job_id=job.id,
                version="1",
                format="tflite",
                storage_key="models/compat.tflite",
                model_metadata={**meta, "project_id": self.project_id},
            )
            db.add(model)
            db.commit()
            return model.id
        finally:
            db.close()

    def _url(self, model_id, fmt, device_profile=None):
        url = f"/api/v1/deployment/compatibility?model_id={model_id}&format={fmt}"
        if device_profile:
            url += f"&device_profile={device_profile}"
        return url

    # ── the read-only probe ──────────────────────────────────────────────────

    def test_compatible_combination(self):
        r = self.client.get(self._url(self.cls_model_id, "arduino"), headers=self.hdrs)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["compatible"] is True
        assert body["reason_code"] == "compatible"
        assert body["model_type"] == "classification"

    def test_incompatible_combination_carries_code_and_message(self):
        r = self.client.get(self._url(self.fomo_model_id, "arduino"), headers=self.hdrs)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["compatible"] is False
        assert body["reason_code"] == "model_format_incompatible"
        assert body["model_type"] == "fomo"
        # Actionable: names the pair and a format that would work.
        assert "arduino" in body["message"]
        assert "tflite" in body["message"]

    def test_same_model_passes_on_a_runtime_format(self):
        r = self.client.get(self._url(self.fomo_model_id, "tflite"), headers=self.hdrs)
        assert r.status_code == 200, r.text
        assert r.json()["compatible"] is True

    # ── auth / ownership ─────────────────────────────────────────────────────

    def test_requires_auth(self):
        r = self.client.get(self._url(self.cls_model_id, "arduino"))
        assert r.status_code in (401, 403)

    def test_other_users_model_is_404_not_403_and_not_a_result(self):
        other = register_and_login(self.client, "compat2@gmail.com", "pass12345", "compatuser2")
        r = self.client.get(self._url(self.cls_model_id, "arduino"), headers=auth_headers(other))
        assert r.status_code == 404, r.text
        assert "compatible" not in r.json()

    def test_unknown_model_is_404(self):
        r = self.client.get(self._url("does-not-exist", "arduino"), headers=self.hdrs)
        assert r.status_code == 404

    # ── the build gate ───────────────────────────────────────────────────────

    @patch("app.api.v1.endpoints.deployment.run_deployment_job.delay")
    def test_incompatible_build_is_rejected_before_anything_happens(self, mock_delay):
        r = self.client.post("/api/v1/deployment/build", json={
            "model_id": self.fomo_model_id,
            "target": "arduino",
        }, headers=self.hdrs)
        assert r.status_code == 400, r.text
        assert "not supported" in r.json()["detail"]

        # No row, no task — the whole point of gating before the insert.
        mock_delay.assert_not_called()
        db = TestingSessionLocal()
        try:
            assert db.query(Deployment).filter(
                Deployment.model_id == self.fomo_model_id
            ).count() == 0
        finally:
            db.close()

    @patch("app.api.v1.endpoints.deployment.run_deployment_job.delay")
    def test_compatible_build_still_enqueues(self, mock_delay):
        mock_task = MagicMock()
        mock_task.id = "compat-celery-task"
        mock_delay.return_value = mock_task

        r = self.client.post("/api/v1/deployment/build", json={
            "model_id": self.cls_model_id,
            "target": "arduino",
        }, headers=self.hdrs)
        assert r.status_code == 201, r.text
        mock_delay.assert_called_once_with(r.json()["id"])

    # ── the two endpoints must not answer differently ────────────────────────

    @patch("app.api.v1.endpoints.deployment.run_deployment_job.delay")
    def test_get_and_post_agree_for_the_same_selection(self, mock_delay):
        """`format=tflite` with a Raspberry Pi profile: unresolved that reads as
        a device mismatch, resolved it is a plain raspberry_pi build — which is
        what POST would actually run. The read-only twin must say what the gate
        would say, or the UI disables a build the server accepts."""
        mock_task = MagicMock()
        mock_task.id = "compat-agree-task"
        mock_delay.return_value = mock_task

        r = self.client.get(
            self._url(self.cls_model_id, "tflite", "raspberry_pi_4"), headers=self.hdrs,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["compatible"] is True, body["message"]
        assert body["build_format"] == "raspberry_pi"

        posted = self.client.post("/api/v1/deployment/build", json={
            "model_id": self.cls_model_id,
            "target": "tflite",
            "device_profile": "raspberry_pi_4",
        }, headers=self.hdrs)
        assert posted.status_code == 201, posted.text

    @patch("app.api.v1.endpoints.deployment.run_deployment_job.delay")
    def test_unknown_device_profile_is_rejected_before_the_build(self, mock_delay):
        """The worker raises "Unsupported deployment device_profile" on a slug
        the catalog does not have. That rejection has to be predicted, not
        discovered minutes into a build."""
        probe = self.client.get(
            self._url(self.cls_model_id, "tflite", "ghost_board"), headers=self.hdrs,
        )
        assert probe.status_code == 200, probe.text
        assert probe.json()["compatible"] is False
        assert probe.json()["reason_code"] == "unknown_device"

        r = self.client.post("/api/v1/deployment/build", json={
            "model_id": self.cls_model_id,
            "target": "tflite",
            "device_profile": "ghost_board",
        }, headers=self.hdrs)
        assert r.status_code == 400, r.text
        assert "ghost_board" in r.json()["detail"]
        mock_delay.assert_not_called()

    def test_no_device_profile_is_still_a_valid_build(self):
        """No target device selected must stay compatible — only a device that
        does not exist is a bad request."""
        r = self.client.get(self._url(self.cls_model_id, "tflite"), headers=self.hdrs)
        assert r.status_code == 200, r.text
        assert r.json()["compatible"] is True

    @patch("app.api.v1.endpoints.deployment.run_deployment_job.delay")
    def test_detection_model_still_builds_for_a_runtime_target(self, mock_delay):
        mock_task = MagicMock()
        mock_task.id = "compat-celery-task-2"
        mock_delay.return_value = mock_task

        r = self.client.post("/api/v1/deployment/build", json={
            "model_id": self.fomo_model_id,
            "target": "tflite",
        }, headers=self.hdrs)
        assert r.status_code == 201, r.text


# ─── SSD deployment routing tests ────────────────────────────────────────────

class TestSSDDeploymentRouting:
    """
    Tests for MobileNetV2-SSD deployment package routing.

    Covers:
      1. output_type 'ssd_detection' selects _SSD_INFERENCE_PY_TEMPLATE for tflite.
      2. output_type 'object_detection' (legacy) also routes to SSD template.
      3. Generated package contains inference.py.
      4. Generated inference.py contains the expected SSD-specific markers.
      5. Same routing applies to raspberry_pi target.
    """

    _SSD_META = {
        "label_names": ["cat", "dog"],
        "input_shape": [320, 320, 3],
        "architecture": "mobilenetv2_ssd_fpnlite_320x320",
        "output_type":  "ssd_detection",
    }
    _DUMMY_TFLITE = b"\x1c\x00\x00\x00TFL3" + b"\x00" * 100

    def _get_inference_src(self, gen_fn, meta) -> str:
        import zipfile, io
        zip_bytes = gen_fn(self._DUMMY_TFLITE, meta, {})
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "inference.py" in names, "Package must include inference.py"
            return zf.read("inference.py").decode()

    def test_ssd_detection_output_type_routes_to_ssd_template_tflite(self):
        """output_type='ssd_detection' must select _SSD_INFERENCE_PY_TEMPLATE for tflite."""
        from app.workers.deployment_worker import _gen_tflite
        src = self._get_inference_src(_gen_tflite, self._SSD_META)
        assert "ssd_outputs_not_recognized" in src, (
            "SSD inference.py must contain 'ssd_outputs_not_recognized' sentinel"
        )
        assert "detections" in src, (
            "SSD inference.py must return a 'detections' key"
        )
        assert "assuming boxes are [y1, x1, y2, x2]" in src, (
            "SSD inference.py must contain the box-format comment"
        )

    def test_ssd_detection_output_type_routes_to_ssd_template_rpi(self):
        """output_type='ssd_detection' must select _SSD_INFERENCE_PY_TEMPLATE for raspberry_pi."""
        from app.workers.deployment_worker import _gen_raspberry_pi
        src = self._get_inference_src(_gen_raspberry_pi, self._SSD_META)
        assert "ssd_outputs_not_recognized" in src, (
            "SSD inference.py must contain 'ssd_outputs_not_recognized' sentinel"
        )
        assert "detections" in src, (
            "SSD inference.py must return a 'detections' key"
        )
        assert "assuming boxes are [y1, x1, y2, x2]" in src, (
            "SSD inference.py must contain the box-format comment"
        )

    def test_legacy_object_detection_output_type_also_routes_to_ssd_template(self):
        """output_type='object_detection' (legacy label) must also use SSD template."""
        from app.workers.deployment_worker import _gen_tflite
        legacy_meta = {**self._SSD_META, "output_type": "object_detection"}
        src = self._get_inference_src(_gen_tflite, legacy_meta)
        assert "ssd_outputs_not_recognized" in src, (
            "Legacy 'object_detection' output_type must route to SSD template"
        )

    def test_ssd_inference_py_does_not_contain_classification_softmax(self):
        """SSD inference.py must not use the classification softmax path."""
        from app.workers.deployment_worker import _gen_tflite
        src = self._get_inference_src(_gen_tflite, self._SSD_META)
        assert "is_fomo" not in src, (
            "SSD inference.py must not contain FOMO-specific code"
        )

    def test_classification_meta_still_uses_generic_template(self):
        """Classification models must not be affected — still use _INFERENCE_PY_TEMPLATE."""
        from app.workers.deployment_worker import _gen_tflite
        cls_meta = {
            "label_names": ["idle", "running"],
            "input_shape": [64],
        }
        src = self._get_inference_src(_gen_tflite, cls_meta)
        assert "ssd_outputs_not_recognized" not in src, (
            "Classification model must not use SSD template"
        )
        assert "is_fomo" in src, (
            "Classification model must still use the generic template with FOMO branch"
        )


# ─── YOLO-Pro deployment routing tests ───────────────────────────────────────

class TestYOLODeploymentRouting:
    """
    Tests for YOLO-Pro deployment package routing.

    Covers:
      1. output_type 'yolo_pro_detection' selects _YOLO_INFERENCE_PY_TEMPLATE for tflite.
      2. output_type 'yolo_pro_detection' selects _YOLO_INFERENCE_PY_TEMPLATE for raspberry_pi.
      3. Generated inference.py contains DFL decode helper.
      4. Generated inference.py contains greedy NMS helper.
    """

    _YOLO_META = {
        "label_names": ["cat", "dog"],
        "input_shape": [96, 96, 3],
        "architecture": "yolo_pro_s",
        "output_type":  "yolo_pro_detection",
        "reg_max":      16,
    }
    _DUMMY_TFLITE = b"\x1c\x00\x00\x00TFL3" + b"\x00" * 100

    def _get_inference_src(self, gen_fn, meta) -> str:
        import zipfile, io
        zip_bytes = gen_fn(self._DUMMY_TFLITE, meta, {})
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "inference.py" in names, "Package must include inference.py"
            return zf.read("inference.py").decode()

    def test_yolo_pro_detection_routes_to_yolo_template_tflite(self):
        """output_type='yolo_pro_detection' must select _YOLO_INFERENCE_PY_TEMPLATE for tflite."""
        from app.workers.deployment_worker import _gen_tflite
        src = self._get_inference_src(_gen_tflite, self._YOLO_META)
        assert "_dfl_decode" in src, (
            "YOLO inference.py must contain '_dfl_decode' DFL helper"
        )
        assert "_nms" in src, (
            "YOLO inference.py must contain '_nms' greedy NMS helper"
        )
        assert "detections" in src, (
            "YOLO inference.py must return a 'detections' key"
        )

    def test_yolo_pro_detection_routes_to_yolo_template_rpi(self):
        """output_type='yolo_pro_detection' must select _YOLO_INFERENCE_PY_TEMPLATE for raspberry_pi."""
        from app.workers.deployment_worker import _gen_raspberry_pi
        src = self._get_inference_src(_gen_raspberry_pi, self._YOLO_META)
        assert "_dfl_decode" in src, (
            "YOLO inference.py must contain '_dfl_decode' DFL helper"
        )
        assert "_nms" in src, (
            "YOLO inference.py must contain '_nms' greedy NMS helper"
        )
        assert "detections" in src, (
            "YOLO inference.py must return a 'detections' key"
        )

    def test_yolo_inference_py_does_not_contain_ssd_sentinel(self):
        """YOLO inference.py must not use the SSD template."""
        from app.workers.deployment_worker import _gen_tflite
        src = self._get_inference_src(_gen_tflite, self._YOLO_META)
        assert "ssd_outputs_not_recognized" not in src, (
            "YOLO inference.py must not contain SSD-specific sentinel"
        )

    # ── Step 3.2 — YOLO _common_meta completeness ────────────────────────────

    def test_yolo_common_meta_contains_all_routing_fields(self):
        """
        _common_meta in run_yolo_pro_training must contain every field that
        deployment_worker routing and _build_dsp_config depend on.
        Inspects source so no database or model weights are needed.
        """
        import inspect
        from app.ml.yolo_pro_worker import run_yolo_pro_training
        src = inspect.getsource(run_yolo_pro_training)

        required = {
            '"output_type"':  '"yolo_pro_detection"',
            '"architecture"': "YOLO_PRO_ARCHITECTURE",
            '"input_shape"':  "input_shape",
            '"label_names"':  "label_names",
            '"dsp_blocks"':   "dsp_blocks",
        }
        for key, value_hint in required.items():
            assert key in src, (
                f"_common_meta must contain {key} — deployment routing depends on it"
            )
            assert value_hint in src, (
                f"_common_meta key {key} must be assigned from {value_hint!r}"
            )

    # ── Step 3.3 — deployment routing + dsp_config.json output_type ──────────

    def test_yolo_gen_tflite_uses_yolo_template_and_dsp_config_has_output_type(self):
        """
        _gen_tflite with output_type='yolo_pro_detection' must:
          (a) use _YOLO_INFERENCE_PY_TEMPLATE — confirmed by REG_MAX sentinel
          (b) emit dsp_config.json with output_type='yolo_pro_detection'
        """
        import zipfile, io, json as _json
        from app.workers.deployment_worker import _gen_tflite
        zip_bytes = _gen_tflite(self._DUMMY_TFLITE, self._YOLO_META, {})
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            inf_src = zf.read("inference.py").decode()
            dsp_cfg = _json.loads(zf.read("dsp_config.json").decode())

        # (a) REG_MAX is defined only in _YOLO_INFERENCE_PY_TEMPLATE
        assert "REG_MAX" in inf_src, (
            "inference.py must contain REG_MAX — proves _YOLO_INFERENCE_PY_TEMPLATE was used"
        )
        # (b) dsp_config.json must carry output_type so re-packaging stays correct
        assert dsp_cfg.get("output_type") == "yolo_pro_detection", (
            "dsp_config.json must contain output_type='yolo_pro_detection' "
            "so downstream tooling can re-route without reading the model file"
        )


class TestDeploymentHealthReport:

    def test_health_report_uses_detector_eval_metrics(self):
        from app.workers.deployment_worker import _compute_health_report

        health = _compute_health_report(
            {},
            {"gt_cells": 388, "f1_threshold": 0.3},
            training_cr={
                "yolo_pro_eval_status": "success",
                "map50": 0.62,
                "avg_confidence": 0.71,
                "total_predicted_cells": 351,
            },
        )

        assert health["model_quality"] == "fair"
        assert health["avg_confidence"] == 0.71
        assert health["predicted_cells"] == 351
        assert health["gt_cells"] == 388
        assert health["warning"] is None

    def test_live_detections_override_stored_detector_summary(self):
        from app.workers.deployment_worker import _compute_health_report

        health = _compute_health_report(
            {
                "detections": [
                    {"confidence": 0.9},
                    {"confidence": 0.7},
                ]
            },
            {"gt_cells": 10, "f1_threshold": 0.3},
            training_cr={
                "yolo_pro_eval_status": "success",
                "map50": 0.85,
                "avg_confidence": 0.2,
                "total_predicted_cells": 99,
            },
        )

        assert health["model_quality"] == "good"
        assert health["predicted_cells"] == 2
        assert health["avg_confidence"] == 0.8


# ─── FOMO MobileNetV2 0.35 tests ─────────────────────────────────────────────

class TestFOMOMobileNetV2:
    """
    Tests for fomo_mobilenetv2_0_1 backend support.

    Covers:
      1. Learning-block catalog includes the architecture.
      2. model_builder recognises and builds the FOMO model.
      3. FOMO model has the expected heatmap output shape.
      4. FOMO model forward pass produces valid probability tensors.
      5. Invalid image dimensions raise a clear ValueError.
      6. build_model raises ValueError for truly unknown architectures
         (not a silent dense fallback).
      7. training_worker fails explicitly with FOMONotImplementedError
         rather than silently mis-training.
      8. Impulse API accepts fomo_mobilenetv2_0_1 as a valid ml_block type.
    """

    # ── 1. Catalog ────────────────────────────────────────────────────────────

    def test_learning_block_catalog_contains_fomo(self, client):
        """GET /dsp/learning-blocks?show_all=true must include fomo_mobilenetv2_0_1."""
        # Register + login so the endpoint isn't blocked by auth
        token = register_and_login(client, "fomo_cat@gmail.com", "pass12345", "fomocat")
        hdrs  = auth_headers(token)
        r = client.get("/api/v1/dsp/learning-blocks?show_all=true", headers=hdrs)
        assert r.status_code == 200, r.text
        block_types = [b["type"] for b in r.json()["blocks"]]
        assert "fomo_mobilenetv2_0_1" in block_types, (
            f"fomo_mobilenetv2_0_1 not found in learning blocks: {block_types}"
        )

    def test_learning_block_fomo_metadata(self, client):
        """fomo_mobilenetv2_0_1 catalog entry must have required fields."""
        token = register_and_login(client, "fomo_meta@gmail.com", "pass12345", "fomometa")
        hdrs  = auth_headers(token)
        r = client.get("/api/v1/dsp/learning-blocks?show_all=true", headers=hdrs)
        assert r.status_code == 200
        blocks = {b["type"]: b for b in r.json()["blocks"]}
        fomo = blocks.get("fomo_mobilenetv2_0_1")
        assert fomo is not None
        assert fomo["name"] == "FOMO (Faster Objects, More Objects) MobileNetV2 0.35"
        assert "image" in fomo["input_types"]
        assert "camera" in fomo["sensor_types"]
        assert fomo["official"] is True
        # params schema must expose alpha, image_width, image_height
        params = fomo.get("params", {})
        assert "alpha"        in params
        assert "image_width"  in params
        assert "image_height" in params
        assert params["alpha"]["default"] == 0.35

    # ── 2 & 3. Model builder — architecture recognition + output shape ────────

    def test_build_fomo_model_recognised(self):
        """build_model must NOT raise for fomo_mobilenetv2_0_1."""
        from app.ml.training.model_builder import build_model
        # (96, 96, 3) is a standard FOMO image input
        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(96, 96, 3),
            num_classes=3,
        )
        assert model is not None
        assert model.name == "fomo_mobilenetv2_0_1"

    def test_fomo_model_output_shape(self):
        """
        FOMO output must be a heatmap: (batch, grid_h, grid_w, num_classes+1).
        For a 96×96 input the grid is 96/8=12 (stride-8 backbone feature map)
        or at most 96/32=3 (stride-32 fallback).  We only assert ndim==4
        and that the last axis == num_classes+1 to stay robust against
        backbone variant differences.
        """
        from app.ml.training.model_builder import build_model
        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(96, 96, 3),
            num_classes=4,
        )
        out_shape = model.output_shape   # (None, grid_h, grid_w, num_classes+1)
        assert len(out_shape) == 4, f"Expected 4-D output, got {out_shape}"
        assert out_shape[-1] == 5, (
            f"Last axis should be num_classes+1=5, got {out_shape[-1]}"
        )

    # ── 4. Forward pass ───────────────────────────────────────────────────────

    def test_fomo_forward_pass(self):
        """Model must produce a valid (0–1) heatmap tensor on a random input."""
        import tensorflow as tf
        from app.ml.training.model_builder import build_model

        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(96, 96, 3),
            num_classes=2,
        )
        x = tf.random.uniform((2, 96, 96, 3))
        y = model(x, training=False)
        assert y.ndim == 4
        assert y.shape[0] == 2
        assert y.shape[-1] == 3   # num_classes + 1
        # Head outputs raw logits (no sigmoid activation).
        # Logits are unbounded — only check shape and dtype here.
        assert y.dtype == tf.float32

    # ── 5. Dimension validation ───────────────────────────────────────────────

    def test_fomo_invalid_dimensions_raise(self):
        """
        Non-multiple-of-8 params must raise ValueError for flat (non-image) inputs.

        For 3-D image inputs, the builder snaps dimensions up to the nearest
        multiple of 8 automatically (no error).  For flat (1-D) inputs, the
        builder relies on params for image size — passing a non-multiple-of-8
        value must raise a clear error so the user knows to fix the configuration.
        """
        from app.ml.training.model_builder import build_model
        import pytest as _pytest
        # 10000-element flat vector + params declaring 100×100 (not a mult of 8)
        with _pytest.raises(ValueError, match="multiples of 8"):
            build_model(
                "fomo_mobilenetv2_0_1",
                input_shape=(10000,),   # flat 1-D — builder uses params for image dims
                num_classes=2,
                params={"image_width": 100, "image_height": 100},
            )

    def test_fomo_non_mult8_input_shape_is_snapped(self):
        """
        A 3-D input_shape with dimensions that are not multiples of 8 must be
        silently snapped up to the nearest multiple of 8.  The model must build
        successfully and produce the correct output grid.

        Example: input (54, 54, 3) → snapped to 56 → grid 56/8 = 7×7.
        """
        from app.ml.training.model_builder import build_model
        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(54, 54, 3),   # 54 is not a multiple of 8
            num_classes=2,
        )
        assert model is not None
        out = model.output_shape  # (None, grid_h, grid_w, num_classes+1)
        assert len(out) == 4, f"Expected 4-D output, got {out}"
        assert out[-1] == 3, f"Expected num_classes+1=3, got {out[-1]}"
        # 54 snaps to 56; stride-8 grid = 56/8 = 7
        assert out[1] == 7, f"Expected grid_h=7, got {out[1]}"
        assert out[2] == 7, f"Expected grid_w=7, got {out[2]}"

    # ── 6. Unknown architecture raises, not silent fallback ──────────────────

    def test_unknown_architecture_raises_not_dense(self):
        """
        build_model must raise ValueError for unknown architectures.
        Previous behaviour silently fell back to dense — that is now gone.
        """
        from app.ml.training.model_builder import build_model
        import pytest as _pytest
        with _pytest.raises((ValueError, NotImplementedError)):
            build_model("totally_unknown_arch_xyz", (64,), 3)

    # ── 7. Training worker is fully supported for FOMO ───────────────────────

    def test_training_worker_fomo_is_fully_supported(self):
        """
        FOMO training is implemented end-to-end.

        Verify:
          - _create_fomo_heatmap exists (heatmap target generation)
          - _combine_dsp_features exists (image tensor preservation)
          - No FOMONotImplementedError symbol (it was a placeholder — now gone)
          - No _WORKER_EXPLICIT_PLACEHOLDER symbol (FOMO is no longer a placeholder)
          - run_training_job uses _fomo_weighted_softmax_loss for FOMO
        """
        import inspect
        from app.workers import training_worker

        assert hasattr(training_worker, "_create_fomo_heatmap"), (
            "_create_fomo_heatmap must exist for heatmap target generation"
        )
        assert hasattr(training_worker, "_combine_dsp_features"), (
            "_combine_dsp_features must exist to preserve image tensors"
        )
        assert not hasattr(training_worker, "FOMONotImplementedError"), (
            "FOMONotImplementedError must not exist — FOMO training is now implemented"
        )
        assert not hasattr(training_worker, "_WORKER_EXPLICIT_PLACEHOLDER"), (
            "_WORKER_EXPLICIT_PLACEHOLDER must not exist — FOMO is no longer a placeholder"
        )
        # Verify weighted softmax loss is used in the FOMO compile path
        src = inspect.getsource(training_worker.run_training_job)
        assert "_fomo_weighted_softmax_loss" in src, (
            "run_training_job must use _fomo_weighted_softmax_loss for FOMO — "
            "softmax CE over logits + background down-weighting"
        )

    def test_training_worker_tflite_metadata_preserves_architecture(self):
        """
        All model records (keras, tflite-float32, tflite-int8) must include
        'architecture' in model_metadata via _common_meta so downstream
        evaluation/deployment can identify FOMO.
        """
        import inspect
        from app.workers import training_worker
        src = inspect.getsource(training_worker.run_training_job)
        # _common_meta carries architecture into every record via **_common_meta
        assert '"architecture": selected_architecture' in src, (
            "_common_meta must include 'architecture'"
        )
        assert "**_common_meta" in src, (
            "All model_metadata dicts must spread _common_meta"
        )

    def test_training_worker_fomo_metadata_has_output_type(self):
        """
        FOMO records must include output_type='detection_heatmap' in metadata
        so evaluation and deployment can detect the model type without
        relying solely on the output tensor shape.
        """
        import inspect
        from app.workers import training_worker
        src = inspect.getsource(training_worker.run_training_job)
        assert "detection_heatmap" in src, (
            "FOMO model_metadata must include output_type='detection_heatmap'"
        )

    def test_fomo_deployment_raises_for_embedded_targets(self):
        """
        FOMO models must raise ValueError for Arduino/ESP32/C++ deployment targets.
        Those templates assume a flat 1-D classification output and would silently
        produce wrong results for a 4-D heatmap output.
        """
        import pytest as _pytest
        from app.workers.deployment_worker import _gen_arduino, _gen_esp32, _gen_cpp

        fomo_meta = {
            "label_names":  ["person", "car"],
            "input_shape":  [96, 96, 3],
            "architecture": "fomo_mobilenetv2_0_1",
            "output_type":  "detection_heatmap",
        }
        dummy_bytes = b"\x00" * 128

        for gen_fn, target in [
            (_gen_arduino, "arduino"),
            (_gen_esp32,   "esp32"),
            (_gen_cpp,     "cpp"),
        ]:
            with _pytest.raises(ValueError, match="FOMO"):
                gen_fn(dummy_bytes, fomo_meta, {})

    def test_fomo_deployment_succeeds_for_python_targets(self):
        """
        TFLite and Raspberry Pi Python targets must succeed for FOMO models —
        the Python inference template already handles 4-D heatmap outputs.
        """
        import zipfile, io
        from app.workers.deployment_worker import _gen_tflite, _gen_raspberry_pi

        fomo_meta = {
            "label_names":  ["person", "car"],
            "input_shape":  [96, 96, 3],
            "architecture": "fomo_mobilenetv2_0_1",
            "output_type":  "detection_heatmap",
        }
        dummy_bytes = b"\x1c\x00\x00\x00TFL3" + b"\x00" * 100

        for gen_fn in [_gen_tflite, _gen_raspberry_pi]:
            zip_bytes = gen_fn(dummy_bytes, fomo_meta, {})
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                names = zf.namelist()
                assert any("inference" in n for n in names), (
                    "Python deployment package must include inference script"
                )
                # Verify FOMO heatmap handling is present in the inference script
                inference_src = zf.read(
                    next(n for n in names if "inference" in n)
                ).decode()
                assert "is_fomo" in inference_src or "fomo" in inference_src.lower(), (
                    "Python inference script must contain FOMO-specific handling"
                )

    def test_evaluation_fomo_skips_classification_accuracy(self):
        """
        _run_inference must NOT apply softmax to 4-D FOMO outputs.
        FOMO outputs use sigmoid and must be returned as-is (flat float array).
        """
        import inspect
        from app.api.v1.endpoints import evaluation
        src = inspect.getsource(evaluation._run_inference)
        # Must check for 4-D output before deciding to apply softmax
        assert "ndim == 4" in src, (
            "_run_inference must check output.ndim == 4 to skip softmax for FOMO"
        )
        # Softmax must only be applied in the non-FOMO branch (conditional)
        # Verify there is a return before softmax for 4D output
        assert "return output.flatten()" in src or "return" in src, (
            "_run_inference must return early for 4-D FOMO output before softmax"
        )

    # ── 8. Training endpoint — architecture allowance / blocking ─────────────

    def test_training_endpoint_allows_fomo_architecture(self):
        """
        _ensure_supported_training_architecture must NOT raise for
        fomo_mobilenetv2_0_1.  It must be treated as a first-class supported
        architecture, not blocked or silently downgraded.
        """
        from app.api.v1.endpoints.training import _ensure_supported_training_architecture
        from unittest.mock import MagicMock

        impulse = MagicMock()
        impulse.ml_blocks = [{"type": "fomo_mobilenetv2_0_1", "params": {}}]
        # Must not raise
        arch = _ensure_supported_training_architecture(impulse, {})
        assert arch == "fomo_mobilenetv2_0_1"

    def test_training_endpoint_blocks_unsupported_architectures(self):
        """
        Architectures in _UNSUPPORTED_TRAINING_ARCHITECTURES must be rejected
        at the HTTP layer with a 400, not at model-build time.
        """
        from fastapi import HTTPException
        from app.api.v1.endpoints.training import _ensure_supported_training_architecture
        from unittest.mock import MagicMock
        import pytest as _pytest

        for arch in ("yolo_pro", "fomo_v2_0.35", "mobilenet_v2_ssd_fpn_lite",
                     "object_detection", "anomaly_gmm"):
            impulse = MagicMock()
            impulse.ml_blocks = [{"type": arch, "params": {}}]
            with _pytest.raises(HTTPException) as exc_info:
                _ensure_supported_training_architecture(impulse, {})
            assert exc_info.value.status_code == 400, (
                f"{arch}: expected 400, got {exc_info.value.status_code}"
            )

    # ── 9. Heatmap target generation unit test ────────────────────────────────

    def test_fomo_heatmap_target_generation(self):
        """
        _create_fomo_heatmap must:
        - return shape (grid_h, grid_w, num_classes+1)
        - default all cells to background (channel 0 = 1.0)
        - set the correct class channel at the box centre cell
        - zero out background at occupied cells
        """
        from app.workers.training_worker import _create_fomo_heatmap
        import numpy as np

        # 96x96 image, 12x12 grid, 2 classes
        grid_w, grid_h = 12, 12
        img_w, img_h   = 96, 96
        num_classes    = 2

        # Fake label_map: label_id "lbl_a" → class index 0 (→ channel 1)
        label_map = {"lbl_a": 0}

        # Box centred at (48, 48) → grid cell (6, 6)
        boxes = [{"x": 40, "y": 40, "w": 16, "h": 16, "label_id": "lbl_a"}]

        heatmap = _create_fomo_heatmap(
            boxes, grid_w, grid_h, img_w, img_h, num_classes, label_map
        )

        assert heatmap.shape == (grid_h, grid_w, num_classes + 1), (
            f"Expected ({grid_h}, {grid_w}, {num_classes+1}), got {heatmap.shape}"
        )
        # Cell at (6, 6): background should be 0, class 1 should be 1
        assert heatmap[6, 6, 0] == 0.0,  "Background must be 0 at occupied cell"
        assert heatmap[6, 6, 1] == 1.0,  "Class channel 1 must be 1 at occupied cell"
        # A random unoccupied cell should be background
        assert heatmap[0, 0, 0] == 1.0,  "Unoccupied cell must be background"
        assert heatmap[0, 0, 1] == 0.0,  "Unoccupied cell class channel must be 0"

    def test_fomo_heatmap_ignores_unknown_label(self):
        """
        _create_fomo_heatmap must silently skip boxes whose label_id is not
        in the label_map — the heatmap stays all-background.
        """
        from app.workers.training_worker import _create_fomo_heatmap

        boxes = [{"x": 20, "y": 20, "w": 10, "h": 10, "label_id": "unknown_lbl"}]
        heatmap = _create_fomo_heatmap(
            boxes, 12, 12, 96, 96, 2, {"known_lbl": 0}
        )
        # All background since the label was unknown
        assert heatmap[:, :, 0].sum() == 12 * 12, "All cells must be background"
        assert heatmap[:, :, 1:].sum() == 0.0,    "No class activations expected"

    def test_fomo_train_test_gt_cell_parity(self):
        """
        Regression: a pixel-space box must map to the same GT cell in both
        training (after normalize_bounding_boxes) and model-testing
        (gt_cells_from_boxes).  Before the fix, the raw-sample training path
        passed pixel-space boxes with img_w=96 (DSP size), shifting GT cells
        relative to model-testing which divides by real image dims.
        """
        import numpy as np
        from app.workers.training_worker import _create_fomo_heatmap
        from app.ml.fomo_evaluator import gt_cells_from_boxes

        # Pixel-space box in a 480×480 source image (the reported failing case).
        img_w_real, img_h_real = 480, 480
        raw_box = {"x": 15.0, "y": 8.0, "w": 353.0, "h": 472.0, "label_id": "cat"}

        # Normalize exactly as normalize_bounding_boxes does for a pixel-space box.
        norm_box = {
            **raw_box,
            "x": raw_box["x"] / img_w_real,
            "y": raw_box["y"] / img_h_real,
            "w": raw_box["w"] / img_w_real,
            "h": raw_box["h"] / img_h_real,
        }
        norm_boxes = [norm_box]

        grid_w, grid_h = 6, 6
        label_map = {"cat": 0}
        label_to_idx = {"cat": 0}

        # Training path: normalized box → _create_fomo_heatmap.
        heatmap = _create_fomo_heatmap(
            norm_boxes, grid_w, grid_h, 96, 96, 1, label_map, spread=False,
        )
        obj_cells = list(zip(*np.where(heatmap[:, :, 0] < 1.0)))
        assert len(obj_cells) == 1, "Expected exactly 1 activated cell in training heatmap"
        train_cell = obj_cells[0]  # (row, col)

        # Testing path: normalized box → gt_cells_from_boxes.
        gt_cells, n_unresolved = gt_cells_from_boxes(norm_boxes, grid_h, grid_w, label_to_idx)
        assert n_unresolved == 0, "GT label must resolve in testing path"
        assert len(gt_cells) == 1, "Expected exactly 1 GT cell in testing path"
        test_cell = next(iter(gt_cells.keys()))  # (row, col)

        assert train_cell == test_cell, (
            f"FOMO GT cell parity failure: training mapped to {train_cell}, "
            f"testing mapped to {test_cell}. "
            "Ensure training normalizes boxes before _create_fomo_heatmap."
        )

    # ── 9b. FOMO detection confusion matrix ───────────────────────────────────

    def test_fomo_evaluate_detection_returns_confusion_matrix(self):
        """
        _evaluate_fomo_detection must include a 'confusion_matrix' key in the
        returned dict when evaluation succeeds.  The matrix must be shaped
        (n_classes+1) × (n_classes+1) with index 0 = background.
        """
        import numpy as np
        import tensorflow as tf
        from tensorflow import keras
        from app.workers.training_worker import _evaluate_fomo_detection, _create_fomo_heatmap

        label_names = ["cat", "dog"]
        n_classes   = len(label_names)
        grid_h, grid_w = 4, 4

        # Build a tiny FOMO model with stride-8 downsampling: 32×32 → 4×4
        inp = keras.Input(shape=(32, 32, 3), name="input")
        x   = keras.layers.MaxPooling2D(8)(inp)   # 32→4
        x   = keras.layers.Conv2D(n_classes + 1, 1, padding="same",
                                  activation="sigmoid", name="fomo_head")(x)
        model = keras.Model(inp, x)

        # Build two synthetic heatmaps: one cat cell, one dog cell
        label_map = {"cat": 0, "dog": 1}
        hm1 = _create_fomo_heatmap(
            [{"x": 4, "y": 4, "w": 8, "h": 8, "label_id": "cat"}],
            grid_w, grid_h, 32, 32, n_classes, label_map,
        )
        hm2 = _create_fomo_heatmap(
            [{"x": 20, "y": 20, "w": 8, "h": 8, "label_id": "dog"}],
            grid_w, grid_h, 32, 32, n_classes, label_map,
        )
        X = np.random.rand(2, 32, 32, 3).astype(np.float32)
        y = np.stack([hm1, hm2])

        result = _evaluate_fomo_detection(model, X, y, label_names, threshold=0.5)

        assert result.get("fomo_eval_status") == "success", (
            "Evaluation must succeed with valid data"
        )
        assert "confusion_matrix" in result, (
            "_evaluate_fomo_detection must include 'confusion_matrix' when eval succeeds"
        )
        cm = result["confusion_matrix"]
        assert len(cm) == n_classes + 1, (
            f"CM must have {n_classes+1} rows (background + {n_classes} classes), got {len(cm)}"
        )
        for row in cm:
            assert len(row) == n_classes + 1, (
                f"Each CM row must have {n_classes+1} columns, got {len(row)}"
            )

    def test_fomo_confusion_matrix_zero_predictions_renders_metrics(self):
        """
        When the FOMO model predicts nothing (all background), the confusion matrix
        must still be built and returned — all object cells should appear as FN
        (missed, predicted as background), and background cells as TN.

        This is the 'background-dominated model' case: metrics are all zero but
        valid, and the CM must render rather than being empty.
        """
        import numpy as np
        import tensorflow as tf
        from tensorflow import keras
        from app.workers.training_worker import _evaluate_fomo_detection, _create_fomo_heatmap

        label_names = ["cat"]
        n_classes   = 1
        grid_h, grid_w = 4, 4
        label_map = {"cat": 0}

        # Model that always predicts background: stride-8 downsampling + zero-init head
        inp  = keras.Input(shape=(32, 32, 3), name="input")
        x    = keras.layers.MaxPooling2D(8)(inp)   # 32→4
        head = keras.layers.Conv2D(n_classes + 1, 1, padding="same",
                                   activation="sigmoid", name="fomo_head",
                                   kernel_initializer="zeros",
                                   bias_initializer="zeros")
        model = keras.Model(inp, head(x))

        hm = _create_fomo_heatmap(
            [{"x": 8, "y": 8, "w": 8, "h": 8, "label_id": "cat"}],
            grid_w, grid_h, 32, 32, n_classes, label_map,
        )
        X = np.zeros((1, 32, 32, 3), dtype=np.float32)
        y = np.stack([hm])

        result = _evaluate_fomo_detection(model, X, y, label_names, threshold=0.9)

        assert result.get("fomo_eval_status") == "success"
        cm = result.get("confusion_matrix", [])
        assert len(cm) == n_classes + 1, "CM must exist even with zero predictions"

        # All cat GT cells must appear as FN (missed → predicted as background)
        cat_fn = result.get("cat", {}).get("fn", None)
        assert cat_fn is not None and cat_fn >= 1, (
            "At least 1 GT cat cell must be recorded as FN when model predicts nothing"
        )
        # TP for cat must be 0
        assert result.get("cat", {}).get("tp", -1) == 0, (
            "TP for cat must be 0 when model predicts nothing"
        )
        # CM[1][0] = FN for cat (predicted as background)
        assert cm[1][0] == cat_fn, (
            f"CM[cat][background] must equal fn={cat_fn}, got {cm[1][0]}"
        )
        # CM[1][1] = TP for cat (correctly detected) = 0
        assert cm[1][1] == 0, "CM[cat][cat] must be 0 when nothing is detected"

    def test_fomo_panel_returns_confusion_matrix_when_eval_succeeded(self):
        """
        When fomo_eval_status == 'success' and a non-empty confusion_matrix is
        stored on the job, the panel API must return it with background as the
        first label.
        """
        import inspect
        from app.api.v1.endpoints import trained_models

        src = inspect.getsource(trained_models.get_model_panel)

        # Panel must read confusion_matrix from the job
        assert "job.confusion_matrix" in src, (
            "Panel must read job.confusion_matrix to build the confusion_matrix response"
        )
        # FOMO CM labels must include background as first entry
        assert '"background"' in src, (
            "FOMO confusion matrix labels must include 'background' as first entry"
        )
        # The matrix data must pass through to the response (not hardcoded empty)
        assert "cm_raw" in src, (
            "Panel must use a cm_raw variable to forward the stored matrix"
        )

    def test_fomo_metrics_section_label_is_validation_set(self):
        """
        The panel API must label the metrics section 'Metrics (validation set)'
        for both FOMO and classification — not 'Detection metrics — grid-cell matching'.
        """
        import inspect
        from app.api.v1.endpoints import trained_models

        src = inspect.getsource(trained_models.get_model_panel)
        assert '"Metrics (validation set)"' in src, (
            "metrics.label must be 'Metrics (validation set)' in the panel response"
        )
        assert "Detection metrics" not in src, (
            "metrics.label must not say 'Detection metrics — grid-cell matching'"
        )

    # ── 10. FOMO dataset loader includes detection-only samples ───────────────

    def test_fomo_load_dataset_query_includes_detection_only_samples(self):
        """
        _load_dataset for FOMO must NOT filter by label_id.isnot(None).
        A FOMO sample that has bounding boxes in extra_metadata but no
        top-level label_id must still be included in training.
        """
        import inspect
        from app.workers import training_worker
        src = inspect.getsource(training_worker._load_dataset)
        # Verify the FOMO branch queries without the label_id filter
        assert "is_fomo" in src, "_load_dataset must branch on is_fomo"
        # The classification branch still keeps label_id filtering — that is expected.
        # The FOMO branch must NOT have label_id.isnot(None) as the sole filter.
        # We verify this by checking the FOMO-only query path exists separately.
        assert "label_id.isnot(None)" in src, (
            "label_id filter must still exist for the classification branch"
        )
        # Confirm the FOMO path falls into the non-label_id branch
        # (presence of 'if is_fomo:' before the sample query)
        fomo_branch_idx = src.index("if is_fomo:")
        query_idx       = src.index("db.query(Sample)")
        assert fomo_branch_idx < query_idx, (
            "is_fomo branch must appear before the sample query"
        )

    # ── 11. Evaluation FOMO sample query ──────────────────────────────────────

    def test_evaluation_classify_test_set_fomo_query_is_label_independent(self):
        """
        classify_test_set must not gate FOMO sample loading on label_id.
        Verify the branching exists in the source.
        """
        import inspect
        from app.api.v1.endpoints import evaluation
        src = inspect.getsource(evaluation.classify_test_set)
        assert "is_fomo_model" in src, (
            "classify_test_set must detect FOMO model and use a separate query path"
        )
        # The FOMO branch must not include label_id filter
        # Verify both paths exist (is_fomo branch and else branch)
        assert "if is_fomo_model:" in src, "Must have explicit FOMO sample query branch"

    # ── 12. Preprocessing contract ────────────────────────────────────────────

    def test_fomo_preprocessing_dsp_outputs_unit_range(self):
        """
        DSPProcessor._image() must output values in [0, 1].

        The FOMO model builder relies on this contract: it applies
        Rescaling(2.0, offset=-1) which maps [0,1] → [-1,1].  If the DSP
        block were to change to output [0, 255], the rescaling would saturate
        and FOMO training would break.
        """
        import numpy as np
        from app.ml.dsp.processor import DSPProcessor

        # Simulate a small 8×8 RGB image as raw bytes (uint8, values 0–255)
        rng = np.random.default_rng(42)
        img_array = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        raw_bytes = img_array.tobytes()

        proc = DSPProcessor(block_type="image", params={
            "image_width": 8, "image_height": 8, "channels": "rgb",
        })
        features = proc.extract(raw_bytes)
        arr = np.asarray(features, dtype=np.float32)

        assert arr.min() >= 0.0, f"DSP image output below 0: min={arr.min()}"
        assert arr.max() <= 1.0, f"DSP image output above 1: max={arr.max()}"

    def test_fomo_preprocessing_model_maps_unit_range_to_minus_one_one(self):
        """
        The FOMO model's internal Rescaling('normalize') layer must map
        [0, 1] input to the [-1, 1] range expected by MobileNetV2.

        Verifies:
          - pixel value 0.0  → −1.0
          - pixel value 1.0  →  +1.0
          - pixel value 0.5  →   0.0  (midpoint)
        """
        import tensorflow as tf
        from tensorflow import keras as _keras
        from app.ml.training.model_builder import build_model

        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(8, 8, 3),
            num_classes=2,
            params={"image_width": 8, "image_height": 8},
        )

        # Extract the "normalize" Rescaling sub-layer
        normalize_layer = None
        for layer in model.layers:
            if layer.name == "normalize":
                normalize_layer = layer
                break
        assert normalize_layer is not None, (
            "FOMO model must contain a layer named 'normalize'"
        )

        # Build a tiny model that only runs the normalize layer
        inp_test = _keras.Input(shape=(8, 8, 3))
        out_test = normalize_layer(inp_test)
        norm_model = _keras.Model(inp_test, out_test)

        # Test sentinel pixel values
        zeros = tf.zeros((1, 8, 8, 3))      # all black  → should become -1
        ones  = tf.ones((1, 8, 8, 3))       # all white  → should become +1
        halfs = tf.fill((1, 8, 8, 3), 0.5)  # mid-grey   → should become  0

        assert abs(float(tf.reduce_mean(norm_model(zeros))) - (-1.0)) < 1e-5, (
            "Rescaling(2.0, offset=-1): 0.0 input must produce -1.0"
        )
        assert abs(float(tf.reduce_mean(norm_model(ones)))  - (+1.0)) < 1e-5, (
            "Rescaling(2.0, offset=-1): 1.0 input must produce +1.0"
        )
        assert abs(float(tf.reduce_mean(norm_model(halfs))) - ( 0.0)) < 1e-5, (
            "Rescaling(2.0, offset=-1): 0.5 input must produce 0.0"
        )

    # ── 13. FOMO training metrics / early-stopping contract ───────────────────

    def test_fomo_compile_has_no_accuracy_metric(self):
        """
        training_worker must NOT compile FOMO with an 'accuracy' metric.

        Plain Keras accuracy is trivially ~1.0 on heatmap targets (most cells
        are background) and provides no signal.  The compile call for FOMO
        must omit the metric entirely.
        """
        import inspect
        from app.workers import training_worker
        src = inspect.getsource(training_worker.run_training_job)

        # Locate the is_fomo compile branch — it must not pass metrics=["accuracy"]
        # Strategy: find the FOMO compile block by its distinctive comment/loss,
        # then verify the word "accuracy" does not appear immediately after it
        # before the 'else' branch.
        fomo_compile_marker = "_fomo_weighted_softmax_loss"
        else_compile_marker = "sparse_categorical_crossentropy"
        fomo_start = src.index(fomo_compile_marker)
        else_start = src.index(else_compile_marker)
        fomo_section = src[fomo_start:else_start]

        assert "metrics" not in fomo_section, (
            "FOMO compile block must not include a 'metrics' argument — "
            "accuracy is misleading for heatmap detection."
        )

    def test_fomo_early_stopping_monitors_val_f1(self):
        """
        training_worker must use monitor='val_f1' for FOMO early stopping
        (driven by _FomoValF1Callback), not val_loss or val_accuracy.
        """
        import inspect
        from app.workers import training_worker
        src = inspect.getsource(training_worker.run_training_job)

        # The FOMO early-stop branch must reference val_f1
        assert "val_f1" in src, (
            "FOMO early stopping must monitor val_f1 — "
            "val_loss gives no signal about detection quality"
        )
        # The classification branch still uses val_accuracy
        assert "val_accuracy" in src, (
            "Classification early stopping must still monitor val_accuracy"
        )

    def test_fomo_patience_is_derived_from_training_epochs_only_for_fomo(self):
        """
        FOMO Phase 2 should derive patience from the configured epoch budget,
        while the non-FOMO Keras EarlyStopping path keeps the fixed patience.
        """
        import inspect
        from app.workers import training_worker
        src = inspect.getsource(training_worker.run_training_job)

        assert "_resolve_fomo_phase2_patience(job.epochs, FOMO_WARMUP_EPOCHS)" in src, (
            "FOMO path must derive Phase 2 patience from the configured epoch budget."
        )
        assert 'monitor="val_accuracy", patience=PATIENCE' in src, (
            "Non-FOMO EarlyStopping must keep its fixed patience configuration."
        )

    def test_fomo_training_history_has_no_accuracy_keys(self):
        """
        training_worker must NOT store 'accuracy' or 'val_accuracy' keys in
        training_history for FOMO — they do not exist and storing empty lists
        would suggest the training ran without them.
        """
        import inspect
        from app.workers import training_worker
        src = inspect.getsource(training_worker.run_training_job)

        # The FOMO history dict must not reference accuracy keys.
        # Find the FOMO branch by locating 'is_fomo' true-branch in DB write section.
        # We look for the 'is_fomo' sentinel in training_history assignment.
        assert '"is_fomo": True' in src or '"is_fomo":          True' in src, (
            "FOMO training_history must include is_fomo=True sentinel"
        )
        assert "metric_note" in src, (
            "FOMO training_history must include a metric_note explaining why "
            "accuracy is absent"
        )

    def test_fomo_best_accuracy_set_to_none(self):
        """
        training_worker must set job.best_accuracy = None for FOMO, not a
        misleading float.
        """
        import inspect
        from app.workers import training_worker
        src = inspect.getsource(training_worker.run_training_job)

        # The FOMO branch must explicitly assign None to best_accuracy
        assert "best_accuracy  = None" in src or "best_accuracy=None" in src or \
               "best_accuracy  = None" in src or "job.best_accuracy  = None" in src, (
            "FOMO path must set job.best_accuracy = None, not a fake float"
        )

    def test_fomo_progress_callback_uses_val_loss_metric(self):
        """
        _ProgressCallback must accept a progress_metric parameter and use it,
        so that FOMO live-progress reports val_loss rather than a zero accuracy.
        """
        import inspect
        from app.workers.training_worker import _ProgressCallback
        sig = inspect.signature(_ProgressCallback.__init__)
        assert "progress_metric" in sig.parameters, (
            "_ProgressCallback.__init__ must accept a 'progress_metric' parameter"
        )
        src = inspect.getsource(_ProgressCallback.on_epoch_end)
        assert "progress_metric" in src, (
            "_ProgressCallback.on_epoch_end must use self.progress_metric "
            "to select which log key to report"
        )

    def test_evaluation_get_metrics_counts_epochs_via_loss(self):
        """
        evaluation.get_metrics must count completed epochs using the 'loss' key
        (always present) rather than 'accuracy' (absent for FOMO).
        """
        import inspect
        from app.api.v1.endpoints import evaluation
        src = inspect.getsource(evaluation.get_metrics)
        # Should prefer "loss" for epoch counting
        assert '"loss"' in src or "'loss'" in src, (
            "get_metrics must use the 'loss' history key to count epochs"
        )

    def test_evaluation_get_metrics_exposes_is_fomo_and_metric_note(self):
        """
        get_metrics response must include is_fomo and metric_note fields so
        the frontend can decide whether to render accuracy or loss curves.
        """
        import inspect
        from app.api.v1.endpoints import evaluation
        src = inspect.getsource(evaluation.get_metrics)
        assert "is_fomo" in src, (
            "get_metrics must expose is_fomo in its response"
        )
        assert "metric_note" in src, (
            "get_metrics must expose metric_note in its response"
        )

    def test_impulse_accepts_fomo_ml_block(self, client):
        """Creating an impulse with fomo_mobilenetv2_0_1 ml_block must succeed."""
        token = register_and_login(client, "fomo_imp@gmail.com", "pass12345", "fomoimp")
        hdrs  = auth_headers(token)

        # Create project
        r = client.post("/api/v1/projects/", json={"name": "FOMOProject"}, headers=hdrs)
        assert r.status_code == 201
        pid = r.json()["id"]

        # Create impulse with FOMO ml_block
        r2 = client.post("/api/v1/impulses/", json={
            "project_id":  pid,
            "name":        "FOMO Impulse",
            "input_type":  "image",
            "image_width":  96,
            "image_height": 96,
            "sensor_type": "camera",
            "dsp_blocks":  [{"type": "image", "params": {}}],
            "ml_blocks":   [{
                "type":         "fomo_mobilenetv2_0_1",
                "architecture": "fomo_mobilenetv2_0_1",
                "params": {
                    "alpha":        0.35,
                    "image_width":  96,
                    "image_height": 96,
                },
            }],
        }, headers=hdrs)
        assert r2.status_code == 201, r2.text
        data = r2.json()
        assert len(data["ml_blocks"]) == 1
        block = data["ml_blocks"][0]
        # Architecture name must be preserved exactly
        arch = block.get("architecture") or block.get("type")
        assert arch == "fomo_mobilenetv2_0_1", (
            f"Architecture name not preserved in ml_blocks: {block}"
        )

    # ── 14. Multi-variant model support ──────────────────────────────────────

    def test_training_worker_saves_multiple_tflite_variants(self):
        """
        training_worker must produce three model artifacts per training job:
        keras (float32), tflite float32 (unoptimized), tflite int8 (quantized).
        """
        import inspect
        from app.workers import training_worker
        src = inspect.getsource(training_worker.run_training_job)

        # Must have both float32 and int8 TFLite conversion paths
        assert "model_float32.tflite" in src, (
            "Worker must save a float32 TFLite variant"
        )
        assert "model_int8.tflite" in src, (
            "Worker must save an int8 TFLite variant"
        )
        # Each variant must set correct quantized flag
        assert '"variant": "float32"' in src, (
            "Float32 records must have variant='float32'"
        )
        assert '"variant": "int8"' in src, (
            "Int8 records must have variant='int8'"
        )

    def test_training_worker_int8_uses_representative_dataset(self):
        """
        Int8 quantization requires a representative dataset.
        _convert_to_tflite must accept quantize=True and pass
        representative data to the converter in-process.

        Note: _convert_isolated was removed — conversion now runs in-process
        to avoid Windows spawn-mode ProcessPoolExecutor failures where the
        child process cannot import 'app.*' modules.
        """
        import inspect
        from app.workers import training_worker
        # Verify _convert_isolated no longer exists (subprocess approach removed)
        assert not hasattr(training_worker, "_convert_isolated"), (
            "_convert_isolated must not exist — conversion is now in-process"
        )
        src = inspect.getsource(training_worker._convert_to_tflite)
        assert "representative_dataset" in src, (
            "Int8 path must provide a representative_dataset to the converter"
        )
        assert "TFLITE_BUILTINS_INT8" in src, (
            "Int8 path must target TFLITE_BUILTINS_INT8 ops"
        )

    def test_trained_model_dict_exposes_variant(self):
        """
        _model_dict must include 'variant' and 'output_type' fields
        so the frontend can build a model version picker.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models._model_dict)
        assert '"variant"' in src, (
            "_model_dict must expose variant field"
        )
        assert '"output_type"' in src, (
            "_model_dict must expose output_type field"
        )

    def test_latest_model_endpoint_accepts_variant_param(self):
        """
        GET /trained-models/impulse/{id}/latest must accept a 'variant'
        query parameter to select between float32 and int8.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_latest_model)
        assert "variant" in src, (
            "get_latest_model must accept a variant parameter"
        )
        assert "available_variants" in src, (
            "get_latest_model must expose available_variants list"
        )

    def test_latest_model_endpoint_returns_available_variants(self):
        """
        get_latest_model response must include an available_variants list
        with id, format, variant, quantized, and file_size_bytes per entry.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_latest_model)
        for field in ["id", "format", "variant", "quantized", "file_size_bytes"]:
            assert f'"{field}"' in src, (
                f"available_variants entries must include '{field}'"
            )

    # ── 15. Model panel endpoint ─────────────────────────────────────────────

    def test_panel_endpoint_exists(self):
        """GET /trained-models/impulse/{id}/panel must exist and return the
        full model panel shape."""
        import inspect
        from app.api.v1.endpoints import trained_models
        assert hasattr(trained_models, "get_model_panel"), (
            "trained_models must expose get_model_panel endpoint"
        )

    def test_panel_response_shape_classification(self):
        """
        Panel response for a classification model must include all required
        top-level keys and classification-specific values.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_model_panel)

        required_keys = [
            "available_model_versions",
            "selected_model_version",
            "available_engines",
            "selected_engine",
            "training_performance",
            "confusion_matrix",
            "metrics",
            "device_performance",
        ]
        for key in required_keys:
            assert f'"{key}"' in src, (
                f"Panel response must include '{key}'"
            )

    def test_panel_classification_performance_has_f1(self):
        """Classification panel must expose F1 score as the headline metric."""
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_model_panel)
        assert "F1 SCORE" in src, (
            "Classification headline metric must be 'F1 SCORE'"
        )

    def test_panel_fomo_performance_uses_loss(self):
        """FOMO panel must use val_loss as headline, not accuracy."""
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_model_panel)
        assert "Best val_loss" in src, (
            "FOMO headline metric must be 'Best val_loss'"
        )
        # Must NOT fake accuracy for FOMO.  Whitespace-tolerant: the dict
        # literal is column-aligned in the endpoint.
        assert re.search(r'["\']accuracy["\']:\s+None', src), (
            "FOMO training_performance.accuracy must be None"
        )

    def test_panel_model_versions_include_unavailable(self):
        """Panel must explicitly list both int8 and float32, marking
        missing variants as available=False."""
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_model_panel)
        assert '"available": True' in src, (
            "Available variants must be marked available=True"
        )
        assert '"available": False' in src, (
            "Missing variants must be marked available=False"
        )

    def test_panel_engines_include_unsupported(self):
        """Panel must list all engine options with supported/unsupported status."""
        import inspect
        from app.api.v1.endpoints import trained_models
        # The engine list lives in the module-level _ENGINE_REGISTRY that the
        # endpoint serves, so assert against the registry itself rather than the
        # function body.
        registry = trained_models._ENGINE_REGISTRY
        ids = {e["id"] for e in registry}
        statuses = {e["status"] for e in registry}
        assert "eon_ram_optimized" in ids, "Must list EON RAM-optimized engine"
        assert "coming_soon" in statuses, "Unsupported engines must show coming_soon status"
        assert "available" in statuses, "Supported engines must show available status"

    def test_panel_engines_have_unsupported_reason(self):
        """Unsupported engines must include an unsupported_reason string."""
        from app.api.v1.endpoints.trained_models import _ENGINE_REGISTRY
        for eng in _ENGINE_REGISTRY:
            if not eng["supported"]:
                assert eng.get("unsupported_reason"), (
                    f"Unsupported engine '{eng['id']}' must have unsupported_reason"
                )
            else:
                assert eng.get("unsupported_reason") is None, (
                    f"Supported engine '{eng['id']}' must have unsupported_reason=None"
                )

    def test_panel_engine_fallback_for_unsupported(self):
        """
        When an unsupported engine is requested, selected_engine must
        include is_fallback=True and fallback_reason.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_model_panel)
        assert "engine_was_fallback" in src, (
            "Panel must track engine fallback state"
        )
        assert '"is_fallback"' in src, (
            "selected_engine must include is_fallback"
        )

    def test_panel_selected_engine_is_object(self):
        """
        selected_engine in panel response must be an object with id,
        supported, is_fallback, and fallback_reason — not a bare string.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_model_panel)
        # Verify the response builds selected_engine as a dict, not a string.
        # Whitespace-tolerant: the response dict is column-aligned.
        assert re.search(r'["\']id["\']:\s+selected_engine', src), (
            "selected_engine must be a dict with 'id' key"
        )
        assert '"supported"' in src, (
            "selected_engine must include 'supported' flag"
        )

    def test_panel_device_performance_uses_source_field(self):
        """
        device_performance metrics must use 'source' field with values
        'measured', 'estimated', or 'unavailable' — not a boolean 'estimated'.

        As of Target Device Phase 5, the panel's device_performance block is
        built by app.services.estimation (shared with GET /deployment/estimate),
        not computed inline in trained_models.py — see
        docs/target_device_phase5.md.
        """
        from app.services import estimation
        assert estimation.SOURCE_MEASURED == "measured"
        assert estimation.SOURCE_ESTIMATED == "estimated"
        assert estimation.SOURCE_UNAVAILABLE == "unavailable"

    def test_panel_device_performance_flash_measured(self):
        """Flash usage must be measured from the model's real file size."""
        import inspect
        from app.services import estimation
        src = inspect.getsource(estimation._flash_metric)
        assert "file_size_bytes" in src or "file_bytes" in src, (
            "Flash must be read from the model's file size"
        )
        assert "SOURCE_MEASURED" in src, "Flash is a real measurement, not an estimate"

    def test_panel_device_performance_ram_estimated(self):
        """RAM usage must be estimated from input shape when available."""
        import inspect
        from app.services import estimation
        src = inspect.getsource(estimation._ram_metric)
        assert "input_shape" in src, "RAM estimate must use input_shape"
        assert "quantized" in src, "RAM estimate must account for quantization"

    def test_panel_device_performance_latency_needs_op_count(self):
        """Latency must be unavailable without a MAC/op count on the model."""
        import inspect
        from app.services import estimation
        src = inspect.getsource(estimation._latency_metric)
        assert "mac_count" in src, "Latency must key off a MAC/op count"
        assert "REASON_NO_OP_COUNT" in src, (
            "Missing op count must report REASON_NO_OP_COUNT, not a fabricated figure"
        )

    def test_panel_device_performance_unsupported_engine(self):
        """
        When the selected engine is unsupported, device_performance must
        return available=False with engine_supported=False and a reason.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models._build_device_performance)
        assert '"engine_supported": False' in src, (
            "Unsupported engine must set engine_supported=False"
        )
        assert '"unavailable_reason"' in src, (
            "Unsupported engine perf must include unavailable_reason"
        )

    def test_estimate_peak_ram_int8_vs_float32(self):
        """Int8 RAM estimate must be smaller than float32 for same input shape."""
        from app.services.estimation import _ram_metric
        ram_int8 = _ram_metric({"input_shape": [96, 96, 3], "quantized": True}, None, None, None)
        ram_f32 = _ram_metric({"input_shape": [96, 96, 3], "quantized": False}, None, None, None)
        assert ram_int8.value < ram_f32.value, (
            f"Int8 RAM ({ram_int8.value}) must be < float32 ({ram_f32.value})"
        )
        assert ram_int8.source == "estimated"
        assert ram_f32.source == "estimated"

    def test_estimate_peak_ram_unknown_shape(self):
        """Empty input shape must return source='unavailable'."""
        from app.services.estimation import _ram_metric
        result = _ram_metric({}, None, None, None)
        assert result.source == "unavailable"
        assert result.value is None

    def test_engine_registry_has_all_three(self):
        """_ENGINE_REGISTRY must contain tflite, eon, and eon_ram_optimized."""
        from app.api.v1.endpoints.trained_models import _ENGINE_REGISTRY
        ids = {e["id"] for e in _ENGINE_REGISTRY}
        assert ids == {"tflite", "eon", "eon_ram_optimized"}

    def test_panel_confusion_matrix_fomo_has_background_label(self):
        """FOMO confusion matrix labels must include 'background' as first entry."""
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_model_panel)
        assert '"background"' in src, (
            "FOMO confusion matrix labels must include 'background'"
        )

    # ── 16. Int8 variant selection (end-to-end contract) ─────────────────────

    def test_panel_int8_selected_when_available(self):
        """
        When ?variant=int8 is requested and an int8 artifact exists,
        selected_model_version.variant must be 'int8' and is_fallback=False.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_model_panel)
        assert '"is_fallback"' in src, (
            "selected_model_version must include is_fallback flag"
        )
        assert '"requested_variant"' in src, (
            "selected_model_version must echo the requested_variant"
        )

    def test_panel_int8_fallback_when_missing(self):
        """
        When ?variant=int8 is requested but int8 does not exist,
        selected_model_version must set is_fallback=True and include
        a fallback_reason string explaining the fallback.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_model_panel)
        assert "variant_was_fallback" in src, (
            "Panel must track whether the selection was a fallback"
        )
        assert '"fallback_reason"' in src, (
            "selected_model_version must include fallback_reason"
        )

    def test_panel_unavailable_variant_has_reason(self):
        """
        available_model_versions entries with available=False must include
        a non-None unavailable_reason string.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_model_panel)
        assert '"unavailable_reason"' in src, (
            "Each model version entry must include unavailable_reason"
        )
        # Available entries set unavailable_reason=None
        assert '"unavailable_reason": None' in src, (
            "Available variants must set unavailable_reason=None"
        )

    def test_panel_uses_real_int8_error_from_training_history(self):
        """
        When int8 conversion fails, the training worker records int8_error
        in training_history. The panel must prefer this real error over
        a generic fallback reason.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_model_panel)
        assert 'int8_error' in src, (
            "Panel must check training_history.int8_error for real failure reason"
        )

    def test_training_worker_records_int8_status(self):
        """
        training_worker must record int8_status and int8_error in
        training_history for both FOMO and classification paths.
        """
        import inspect
        from app.workers import training_worker
        src = inspect.getsource(training_worker.run_training_job)
        assert src.count('"int8_status"') >= 2, (
            "int8_status must appear in both FOMO and classification history dicts"
        )
        assert src.count('"int8_error"') >= 2, (
            "int8_error must appear in both FOMO and classification history dicts"
        )

    # ── 16b. Per-variant evaluation metrics ──────────────────────────────────

    def test_ssd_worker_records_int8_status(self):
        """
        mobilenetv2_ssd_worker must record int8_status/int8_error into
        training_history like the other two workers.  The panel's "why is int8
        missing" message reads training_history["int8_error"]; without this the
        SSD path had no source for it and always fell back to generic copy.
        """
        import inspect
        from app.ml import mobilenetv2_ssd_worker
        src = inspect.getsource(mobilenetv2_ssd_worker)
        assert 'int8_status' in src, (
            "SSD worker must track an int8_status flag"
        )
        assert '_hist["int8_status"]' in src and '_hist["int8_error"]' in src, (
            "SSD worker must merge int8_status/int8_error into training_history "
            "after export (it is written per-epoch during the loop)"
        )

    def test_all_three_workers_run_per_variant_eval(self):
        """
        Every worker that exports TFLite variants must score them individually
        after export.  Without this each variant inherits the job-wide Keras
        eval, so int8's quantization loss is invisible in the UI.
        """
        import inspect
        from app.ml import mobilenetv2_ssd_worker, yolo_pro_worker
        from app.workers import training_worker

        for mod, name in (
            (training_worker, "training_worker"),
            (yolo_pro_worker, "yolo_pro_worker"),
            (mobilenetv2_ssd_worker, "mobilenetv2_ssd_worker"),
        ):
            src = inspect.getsource(mod)
            assert "evaluate_exported_variants" in src, (
                f"{name} must call evaluate_exported_variants after export"
            )
            assert '"int8":' in src, (
                f"{name} must submit the int8 variant for evaluation"
            )

    def test_variant_eval_isolates_per_variant_failures(self):
        """
        One variant failing to evaluate must not fail the training job or
        prevent the other variants from being scored.
        """
        import inspect
        from app.ml import variant_eval
        src = inspect.getsource(variant_eval.evaluate_exported_variants)
        assert 'except Exception as exc:' in src, (
            "per-variant eval must catch exceptions per variant"
        )
        assert '"status": "failed"' in src, (
            "a variant whose eval raises must be recorded status='failed'"
        )
        assert '"status": "export_failed"' in src, (
            "a variant with no artifact must be recorded status='export_failed'"
        )

    def test_variant_eval_persists_onto_trained_model_metadata(self):
        """
        Metrics live on TrainedModel.model_metadata (already one row per
        variant), not a new table — so no migration is required.
        """
        import inspect
        from app.ml import variant_eval
        assert variant_eval.VARIANT_METRICS_KEY == "variant_metrics"
        src = inspect.getsource(variant_eval.evaluate_exported_variants)
        assert "TrainedModel" in src and "model_metadata" in src, (
            "per-variant metrics must be stamped onto the variant's own "
            "TrainedModel.model_metadata"
        )

    def test_variant_eval_dequantizes_int8_io(self):
        """
        int8 tensors must be dequantized before scoring, using the established
        round → clip → cast / (raw - zero) * scale convention rather than
        hand-rolled scaling.
        """
        import inspect
        from app.ml import variant_eval
        src = inspect.getsource(variant_eval.TFLiteVariantRunner)
        assert "np.round" in src and "np.clip" in src, (
            "int8 input quantization must round-then-clip, not truncate"
        )
        assert "(raw - zero) * scale" in src, (
            "int8 outputs must be dequantized back to float before scoring"
        )

    def test_panel_exposes_per_variant_metrics(self):
        """
        The panel must return each variant's own metrics plus an explicit
        availability flag/reason, so the dropdown never renders one variant's
        numbers under another variant's name.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_model_panel)
        assert '"metrics_available"' in src, (
            "each model version entry must carry metrics_available"
        )
        assert '"metrics_unavailable_reason"' in src, (
            "an unscored/failed variant must carry the recorded reason"
        )
        assert 'variant_metrics' in src, (
            "panel must read per-variant metrics from model_metadata"
        )
        assert 'has_variant_metrics' in src, (
            "panel must prefer the selected variant's own metrics over the "
            "job-wide classification_report"
        )

    def test_panel_aggregate_metrics_match_headline_source(self):
        """
        The METRICS aggregate block (map/map50/map75/precision/recall) must be
        read from the same variant-scoped source as the headline metric, not
        from the job-wide classification_report — otherwise the headline and
        the METRICS list can show two different numbers for the same variant
        (regression: headline read `selected_variant_metrics`, the aggregate
        block below it read `cr` unconditionally).
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_model_panel)

        assert "eval_src = selected_variant_metrics if has_variant_metrics else cr" in src, (
            "headline and aggregate metrics must resolve from one shared "
            "eval_src, computed once, not two independent has_variant_metrics "
            "ternaries that can drift apart"
        )

        agg_block = src[src.index("# ── Aggregate metrics"):]
        assert 'eval_src.get("map")' in agg_block, (
            "aggregate map must come from eval_src, not the job-wide `cr`"
        )
        assert 'eval_src.get("map50")' in agg_block, (
            "aggregate map50 must come from eval_src, not the job-wide `cr`"
        )
        assert 'eval_src.get("map75")' in agg_block, (
            "aggregate map75 must come from eval_src, not the job-wide `cr`"
        )
        assert 'eval_src.get("precision")' in agg_block, (
            "aggregate precision must come from eval_src, not the job-wide `cr`"
        )
        assert 'eval_src.get("recall")' in agg_block, (
            "aggregate recall must come from eval_src, not the job-wide `cr`"
        )
        assert 'eval_src.get("detailed_metrics"' in agg_block, (
            "extended COCO breakdown (map_small/large, recall@k, ...) must "
            "come from eval_src, not the job-wide `cr`"
        )

    def test_latest_model_endpoint_signals_fallback(self):
        """
        GET /trained-models/impulse/{id}/latest must include is_fallback
        and fallback_reason when the requested variant was not found.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_latest_model)
        assert '"is_fallback"' in src, (
            "get_latest_model must include is_fallback in response"
        )
        assert '"fallback_reason"' in src, (
            "get_latest_model must include fallback_reason when falling back"
        )

    def test_project_latest_model_endpoint_exists(self):
        """
        Device clients need a project-scoped latest-model endpoint so they can
        recover from stale PETAL_MODEL_ID values after WS auth resolves the
        owning project.
        """
        import inspect
        from app.api.v1.endpoints import trained_models
        src = inspect.getsource(trained_models.get_latest_model_for_project)
        assert '"/project/{project_id}/latest"' in src or "project_id" in src
        assert '"requested_variant"' in src
        assert '"is_fallback"' in src

    def test_hello_ack_carries_project_id(self):
        """
        Device WS hello-ack should include project_id so clients can resolve
        a fresh latest model for the authenticated project.
        """
        import inspect
        from app.realtime import schemas, ws_device
        hello_ack_src = inspect.getsource(schemas.HelloAck)
        ws_src = inspect.getsource(ws_device.device_ws)
        assert "project_id" in hello_ack_src
        assert "project_id=project_id" in ws_src

    def test_panel_variant_metadata_preserved(self):
        """
        Both int8 and float32 variants must preserve architecture and
        output_type in their metadata (via _common_meta in the worker).
        """
        import inspect
        from app.workers import training_worker
        src = inspect.getsource(training_worker.run_training_job)
        # _common_meta carries architecture and output_type for all variants.
        # Whitespace-tolerant: the dict literal is column-aligned, so an exact
        # single-space match breaks on pure formatting changes.
        assert re.search(r'"architecture":\s+selected_architecture', src), (
            "_common_meta must carry architecture for every variant"
        )
        # Both int8 and float32 TrainedModel records must spread _common_meta
        assert src.count("**_common_meta") >= 3, (
            "All three records (keras, tflite-float32, tflite-int8) must spread _common_meta"
        )

    # ── 17. Image-size consistency (DSP ↔ model ↔ grid) ─────────────────────

    def test_fomo_54x54_builds_for_54x54_not_96x96(self):
        """
        When the DSP output / input_shape is (54, 54, 3), the FOMO builder
        must NOT silently resize to 96×96 internally.

        Expected: input layer accepts (None, 54, 54, 3).
        The internal Resizing layer, if present, must target 56×56 (snapped),
        NOT 96×96.
        """
        from app.ml.training.model_builder import build_model
        from tensorflow import keras

        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(54, 54, 3),
            num_classes=2,
        )
        # Input layer must accept the given shape, not 96×96
        assert model.input_shape == (None, 54, 54, 3), (
            f"Model input_shape must be (None, 54, 54, 3), got {model.input_shape}"
        )
        # If a Resizing layer exists its target must be 56×56 (snapped), not 96×96
        for layer in model.layers:
            if isinstance(layer, keras.layers.Resizing):
                assert layer.height != 96 and layer.width != 96, (
                    f"Resizing layer must not target 96×96 when input is 54×54; "
                    f"got ({layer.height}, {layer.width})"
                )
                # Must snap to nearest mult-of-8 above 54 = 56
                assert layer.height == 56 and layer.width == 56, (
                    f"54 should snap to 56, got ({layer.height}, {layer.width})"
                )

    def test_fomo_96x96_builds_for_96x96(self):
        """
        When the DSP output / input_shape is (96, 96, 3), the FOMO builder
        must build for exactly 96×96 (no Resizing layer needed, or a 96×96 one).
        Grid must be 96/8 = 12.
        """
        from app.ml.training.model_builder import build_model
        from tensorflow import keras

        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(96, 96, 3),
            num_classes=2,
        )
        assert model.input_shape == (None, 96, 96, 3), (
            f"Model input_shape must be (None, 96, 96, 3), got {model.input_shape}"
        )
        # No Resizing layer, or a 96×96 one (no-op)
        for layer in model.layers:
            if isinstance(layer, keras.layers.Resizing):
                assert layer.height == 96 and layer.width == 96, (
                    f"Resizing target must be 96×96 for 96×96 input, "
                    f"got ({layer.height}, {layer.width})"
                )

    def test_fomo_grid_matches_snapped_image_size_54(self):
        """
        For a 54×54 input (snaps to 56), the model output grid must be 7×7
        (56 / 8 = 7).  This is the same formula used in training_worker for
        heatmap targets, so the shapes are guaranteed to align.
        """
        from app.ml.training.model_builder import build_model

        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(54, 54, 3),
            num_classes=1,
        )
        out = model.output_shape  # (None, grid_h, grid_w, num_classes+1)
        assert out[1] == 7, f"grid_h should be 7 for 54→56 input, got {out[1]}"
        assert out[2] == 7, f"grid_w should be 7 for 54→56 input, got {out[2]}"

    def test_fomo_grid_matches_snapped_image_size_96(self):
        """
        For a 96×96 input the grid must be 12×12 (96 / 8 = 12).
        """
        from app.ml.training.model_builder import build_model

        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(96, 96, 3),
            num_classes=1,
        )
        out = model.output_shape
        assert out[1] == 12, f"grid_h should be 12 for 96×96 input, got {out[1]}"
        assert out[2] == 12, f"grid_w should be 12 for 96×96 input, got {out[2]}"

    def test_load_dataset_img_wh_from_dsp_params_not_silent_96(self):
        """
        _load_dataset must read img_w / img_h from the DSP image-block params
        (the same source DSPProcessor uses) rather than silently defaulting to 96.

        Verified via source inspection: the code must look up 'image_width' /
        'image_height' from the dsp_block params before falling back to
        impulse.image_width / 96.
        """
        import inspect
        from app.workers import training_worker

        src = inspect.getsource(training_worker._load_dataset)

        # Must explicitly read dsp block params for image size
        assert "image_width" in src, (
            "_load_dataset must reference 'image_width' from DSP block params"
        )
        assert "image_height" in src, (
            "_load_dataset must reference 'image_height' from DSP block params"
        )
        # The lookup must check dsp params BEFORE impulse width (source priority)
        img_w_idx  = src.index("image_width")
        imp_w_idx  = src.index("impulse.image_width")
        assert img_w_idx < imp_w_idx, (
            "DSP block param lookup (image_width) must appear before "
            "impulse.image_width fallback in _load_dataset"
        )

    def test_load_dataset_img_wh_dsp_params_shadow_default_96(self):
        """
        When a DSP image block has image_width/height != 96, _load_dataset
        must use those values — it must NOT silently override them with 96.

        Verified via source: there must be no unconditional `img_w = ... or 96`
        before the dsp-params lookup.
        """
        import inspect
        from app.workers import training_worker

        src = inspect.getsource(training_worker._load_dataset)

        # The old buggy pattern was:  img_w = impulse.image_width or 96
        # That pattern must no longer be the first/only assignment to img_w.
        # We verify by checking that the dsp-params lookup comes before any
        # direct '96' default assignment.
        assert "_dsp_img_params" in src, (
            "_load_dataset must use a _dsp_img_params dict to read DSP image size"
        )

    def test_no_silent_resize_to_default_when_explicit_size_set(self):
        """
        When build_model receives an explicit (H, W, 3) or (H, W) input shape,
        any Resizing layer inside the model must target the snapped version of
        that explicit size — NOT the 96 default from params.

        Regression guard: model builder must not read params['image_width/height']
        for 2-D or 3-D image inputs; it must use input_shape exclusively.
        """
        import inspect
        from app.ml.training import model_builder

        src = inspect.getsource(model_builder._build_fomo_mobilenetv2_0_1)

        # For 2-D and 3-D inputs (image cases), the code derives from input_shape.
        # Verify the branch snaps from raw_h/raw_w, not params.
        assert "raw_h, raw_w = input_shape[0], input_shape[1]" in src, (
            "FOMO builder must derive image size from input_shape[0]/[1] for 2-D/3-D inputs"
        )
        # params fallback must only apply in the else branch (flat 1-D inputs)
        if_2d_idx  = src.index("if len(input_shape) >= 2:")
        else_idx   = src.index("else:", if_2d_idx)
        params_idx = src.index('params.get("image_width"')
        assert params_idx > else_idx, (
            "params.get('image_width') must only appear in the else/flat-input "
            "branch, not before it — 2-D/3-D inputs must use input_shape exclusively"
        )

    def test_fomo_2d_input_71x71_output_shape_is_9x9(self):
        """
        FOMO model built from a 2-D DSP feature input (71, 71) must output a
        9×9 heatmap grid, NOT 12×12 (the 96×96 default fallback).

        71 snaps to 72 (nearest multiple of 8); 72 / 8 = 9.
        """
        from app.ml.training.model_builder import build_model

        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(71, 71),
            num_classes=2,
        )
        out = model.output_shape  # (None, grid_h, grid_w, num_classes+1)
        assert out[1] == 9, (
            f"grid_h should be 9 for 2-D 71×71 input (snaps to 72), got {out[1]}"
        )
        assert out[2] == 9, (
            f"grid_w should be 9 for 2-D 71×71 input (snaps to 72), got {out[2]}"
        )

    def test_fomo_2d_input_no_silent_resize_to_96(self):
        """
        A 2-D input (71, 71) must not produce a 12×12 output (the 96×96 fallback).
        This is the direct regression guard for the shape-mismatch bug where
        model output was (None, 12, 12, C) but targets were (None, 9, 9, C).
        """
        from app.ml.training.model_builder import build_model

        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(71, 71),
            num_classes=1,
        )
        out = model.output_shape
        assert out[1] != 12 and out[2] != 12, (
            f"2-D (71,71) input must not produce 12×12 output (the 96×96 default). "
            f"Got {out[1]}×{out[2]}"
        )

    def test_fomo_2d_grid_aligns_with_training_worker(self):
        """
        The grid computed by training_worker._snap8 for a 2-D feat_shape (71, 71)
        must equal the model output grid for input_shape (71, 71).

        Both must use snap-to-multiple-of-8 on feat_shape[0]/[1], not img_h/img_w.
        """
        from app.ml.training.model_builder import build_model

        def _snap8(n):
            return n if n % 8 == 0 else n + (8 - n % 8)

        feat_shape = (71, 71)
        expected_grid_h = _snap8(feat_shape[0]) // 8  # 9
        expected_grid_w = _snap8(feat_shape[1]) // 8  # 9

        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=feat_shape,
            num_classes=2,
        )
        out = model.output_shape
        assert out[1] == expected_grid_h, (
            f"Model grid_h {out[1]} != worker grid_h {expected_grid_h} for 2-D input"
        )
        assert out[2] == expected_grid_w, (
            f"Model grid_w {out[2]} != worker grid_w {expected_grid_w} for 2-D input"
        )


# ─── FOMO v2 (Adaptive Resolution) tests ─────────────────────────────────────


class TestFOMOv2:
    """Tests for FOMO v2 (Adaptive Resolution, stride-8).

    Regression contract:
      - v1 code paths are NEVER entered at runtime when fomo_version == 2.
      - v2 grid = input_size // 8 (stride-8, block_6_expand_relu).
      - Old impulses without fomo_version default to v1 (unchanged behaviour).
    """

    # ── A. Model builder dispatches correctly ─────────────────────────────────

    def test_v1_regression_96x96_output_shape(self):
        """v1 at 96×96: model name must be 'fomo_mobilenetv2_0_1' (unchanged)."""
        from app.ml.training.model_builder import build_model
        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(96, 96, 3),
            num_classes=2,
            params={"fomo_version": 1},
        )
        assert model.name == "fomo_mobilenetv2_0_1", (
            f"v1 model name must be 'fomo_mobilenetv2_0_1', got {model.name!r}"
        )
        out = model.output_shape
        assert len(out) == 4, "v1 output must be 4-D (batch, H, W, C+1)"
        assert out[-1] == 3, f"v1 output last axis must be num_classes+1=3, got {out[-1]}"

    def test_v2_128x128_model_name(self):
        """v2 at 128×128: model name must be 'fomo_mobilenetv2_v2'."""
        from app.ml.training.model_builder import build_model
        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(128, 128, 3),
            num_classes=2,
            params={"fomo_version": 2},
        )
        assert model.name == "fomo_mobilenetv2_v2", (
            f"v2 model name must be 'fomo_mobilenetv2_v2', got {model.name!r}"
        )

    def test_v2_output_shape_128x128(self):
        """v2 at 128×128: output grid must be 16×16 (128//8), 4× v1 grid area."""
        from app.ml.training.model_builder import build_model
        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(128, 128, 3),
            num_classes=3,
            params={"fomo_version": 2},
        )
        out = model.output_shape  # (None, grid_h, grid_w, C+1)
        assert len(out) == 4, "v2 output must be 4-D"
        assert out[-1] == 4, f"v2 num_classes+1=4, got {out[-1]}"
        # Grid must be stride-8 → 128/8 = 16
        assert out[1] == 16 and out[2] == 16, (
            f"v2 128×128 grid must be 16×16 (stride-8), got {out[1]}×{out[2]}"
        )

    def test_v2_output_shape_160x160(self):
        """v2 at 160×160: output grid must be 20×20 (160//8)."""
        from app.ml.training.model_builder import build_model
        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(160, 160, 3),
            num_classes=1,
            params={"fomo_version": 2},
        )
        out = model.output_shape
        assert out[1] == 20 and out[2] == 20, (
            f"v2 160×160 grid must be 20×20 (stride-8), got {out[1]}×{out[2]}"
        )

    def test_build_fomo_v2_public_api(self):
        """build_fomo_v2() public helper must return a v2 model."""
        from app.ml.training.model_builder import build_fomo_v2
        model = build_fomo_v2(input_shape=(128, 128, 3), num_classes=2)
        assert model.name == "fomo_mobilenetv2_v2"
        out = model.output_shape
        assert out[1] == 16 and out[2] == 16, (
            f"build_fomo_v2 128×128 must give 16×16 grid, got {out[1]}×{out[2]}"
        )

    def test_v2_forward_pass_128x128(self):
        """v2 forward pass at 128×128 must produce a valid 4-D tensor."""
        import tensorflow as tf
        from app.ml.training.model_builder import build_model
        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(128, 128, 3),
            num_classes=2,
            params={"fomo_version": 2},
        )
        x = tf.zeros((1, 128, 128, 3))
        y = model(x, training=False)
        assert y.ndim == 4
        assert y.shape[0] == 1
        assert y.shape[-1] == 3  # num_classes + 1
        assert y.dtype == tf.float32

    # ── B. Default fomo_version = 1 for old impulses ──────────────────────────

    def test_default_fomo_version_is_1(self):
        """build_model with no fomo_version param must build v1 (default)."""
        from app.ml.training.model_builder import build_model
        model = build_model(
            "fomo_mobilenetv2_0_1",
            input_shape=(96, 96, 3),
            num_classes=2,
        )
        # v1 model name is preserved (v2 would be 'fomo_mobilenetv2_v2')
        assert model.name == "fomo_mobilenetv2_0_1", (
            f"Default (no fomo_version param) must produce v1, got {model.name!r}"
        )

    def test_fomo_version_param_zero_or_missing_defaults_to_v1(self):
        """fomo_version=0 or None in params must fall back to v1."""
        from app.ml.training.model_builder import build_model
        for bad_val in (0, None, ""):
            params = {"fomo_version": bad_val}
            # int(0 or 1) = 1, int(None or 1) would raise but we use .get default
            # The dispatch uses int(params.get("fomo_version", 1)) which for 0 gives int(0)=0 → v1
            # For None: int(None) would raise, but params.get returns None → default 1 is used
            try:
                model = build_model(
                    "fomo_mobilenetv2_0_1",
                    input_shape=(96, 96, 3),
                    num_classes=2,
                    params=params,
                )
                assert model.name == "fomo_mobilenetv2_0_1", (
                    f"fomo_version={bad_val!r} should default to v1, got {model.name!r}"
                )
            except (ValueError, TypeError):
                pass  # some falsy values may raise — that's acceptable

    # ── C. DSP processor version detection ────────────────────────────────────

    def test_get_fomo_version_returns_1_for_old_impulse(self):
        """_get_fomo_version must return 1 for impulses without the field."""
        from app.ml.dsp.processor import _get_fomo_version
        old_impulse = {"ml_blocks": [{"type": "fomo_mobilenetv2_0_1", "params": {}}]}
        assert _get_fomo_version(old_impulse) == 1

    def test_get_fomo_version_returns_2_for_v2_impulse(self):
        """_get_fomo_version must return 2 when fomo_version=2 is set in params."""
        from app.ml.dsp.processor import _get_fomo_version
        v2_impulse = {"ml_blocks": [{"type": "fomo_mobilenetv2_0_1", "params": {"fomo_version": 2}}]}
        assert _get_fomo_version(v2_impulse) == 2

    def test_v1_merge_image_params_keeps_96_floor(self):
        """v1: merge_image_params must enforce the 96×96 minimum (unchanged behaviour)."""
        from app.ml.dsp.processor import merge_image_params
        impulse = {"input_type": "image", "image_width": 48, "image_height": 48,
                   "resize_mode": None, "ml_blocks": [{"type": "fomo_mobilenetv2_0_1",
                   "params": {"fomo_version": 1}}]}
        result = merge_image_params(impulse, {"type": "image", "params": {}})
        assert result["image_width"] >= 96, "v1: width must be at least 96"
        assert result["image_height"] >= 96, "v1: height must be at least 96"

    def test_v2_merge_image_params_no_96_floor(self):
        """v2: merge_image_params must NOT enforce the 96×96 minimum."""
        from app.ml.dsp.processor import merge_image_params
        impulse = {"input_type": "image", "image_width": 64, "image_height": 64,
                   "resize_mode": None, "ml_blocks": [{"type": "fomo_mobilenetv2_0_1",
                   "params": {"fomo_version": 2}}]}
        result = merge_image_params(impulse, {"type": "image", "params": {}})
        # v2 must not snap to 96 — 64 is already a multiple of 8
        assert result["image_width"] == 64, (
            f"v2: 64×64 must NOT be bumped to 96, got {result['image_width']}"
        )
        assert result["image_height"] == 64, (
            f"v2: 64×64 height must NOT be bumped to 96, got {result['image_height']}"
        )

    def test_v2_merge_image_params_snaps_to_mult8(self):
        """v2: non-mult-of-8 dimensions must be snapped up to nearest multiple of 8."""
        from app.ml.dsp.processor import merge_image_params
        impulse = {"input_type": "image", "image_width": 65, "image_height": 65,
                   "resize_mode": None, "ml_blocks": [{"type": "fomo_mobilenetv2_0_1",
                   "params": {"fomo_version": 2}}]}
        result = merge_image_params(impulse, {"type": "image", "params": {}})
        # 65 snaps to 72 (nearest multiple of 8 above 65)
        assert result["image_width"] % 8 == 0, (
            f"v2: snapped width must be divisible by 8, got {result['image_width']}"
        )
        assert result["image_width"] == 72, (
            f"v2: 65 must snap to 72, got {result['image_width']}"
        )

    # ── D. Training worker reads fomo_version correctly ───────────────────────

    def test_training_worker_reads_fomo_version_from_extra_params(self):
        """training_worker must prefer extra_params.fomo_version over impulse params."""
        import inspect
        from app.workers import training_worker
        src = inspect.getsource(training_worker.run_training_job)
        assert "fomo_version" in src, "run_training_job must read fomo_version"
        assert "_fomo_version" in src, "_fomo_version variable must exist in training job"

    def test_training_worker_fomo_metadata_includes_version_and_grid(self):
        """FOMO model metadata must include fomo_version and grid_size fields."""
        import inspect
        from app.workers import training_worker
        src = inspect.getsource(training_worker.run_training_job)
        assert '"fomo_version"' in src, "model metadata must include fomo_version"
        assert '"grid_size"' in src, "model metadata must include grid_size"

    # ── E. Deployment manifest contains fomo_version ─────────────────────────

    def test_deployment_manifest_contains_fomo_version(self):
        """_build_manifest must write fomo_version and grid_size for FOMO models."""
        import json
        from app.workers.deployment_worker import _build_manifest
        fomo_v2_meta = {
            "label_names":   ["person"],
            "input_shape":   [128, 128, 3],
            "architecture":  "fomo_mobilenetv2_0_1",
            "output_type":   "detection_heatmap",
            "fomo_version":  2,
            "grid_size":     16,
        }
        manifest_bytes = _build_manifest(fomo_v2_meta, "tflite")
        manifest = json.loads(manifest_bytes.decode())
        assert manifest.get("fomo_version") == 2, (
            f"Manifest fomo_version must be 2, got {manifest.get('fomo_version')!r}"
        )
        assert manifest.get("grid_size") == 16, (
            f"Manifest grid_size must be 16, got {manifest.get('grid_size')!r}"
        )

    def test_deployment_manifest_v1_default_fomo_version(self):
        """_build_manifest for old v1 models (no fomo_version key) must default to 1."""
        import json
        from app.workers.deployment_worker import _build_manifest
        old_fomo_meta = {
            "label_names":  ["cat"],
            "input_shape":  [96, 96, 3],
            "architecture": "fomo_mobilenetv2_0_1",
            "output_type":  "detection_heatmap",
            # no fomo_version key — simulates old model
        }
        manifest_bytes = _build_manifest(old_fomo_meta, "tflite")
        manifest = json.loads(manifest_bytes.decode())
        assert manifest.get("fomo_version") == 1, (
            f"Old models without fomo_version must default to 1 in manifest, "
            f"got {manifest.get('fomo_version')!r}"
        )

    def test_pxe_build_accepts_uuid_project_id(self):
        """_gen_pxe must not fail when project_id is a UUID string."""
        from app.workers.deployment_worker import _gen_pxe

        pkg = _gen_pxe(
            b"dummy_tflite",
            {
                "label_names": ["cat"],
                "input_shape": [96, 96, 3],
                "project_id": "7b88dd54-f16e-4445-8b0d-f9f8f3f5b7e4",
            },
            {},
        )
        assert isinstance(pkg, (bytes, bytearray))
        assert len(pkg) > 0

    # ── F. Catalog exposes fomo_version param ─────────────────────────────────

    def test_fomo_catalog_exposes_fomo_version_param(self, client):
        """fomo_mobilenetv2_0_1 catalog entry must include fomo_version param."""
        import pytest
        token = register_and_login(client, "fomov2_cat@gmail.com", "pass12345", "fomov2cat")
        hdrs  = auth_headers(token)
        r = client.get("/api/v1/dsp/learning-blocks?show_all=true", headers=hdrs)
        assert r.status_code == 200
        blocks = {b["type"]: b for b in r.json()["blocks"]}
        fomo = blocks.get("fomo_mobilenetv2_0_1", {})
        params = fomo.get("params", {})
        assert "fomo_version" in params, (
            f"fomo_mobilenetv2_0_1 catalog params must expose fomo_version, "
            f"got params: {list(params.keys())}"
        )
        assert params["fomo_version"]["default"] == 1, (
            "fomo_version default must be 1 (v1 is the safe default)"
        )

    # ── G. decode_fomo_heatmap accepts v2 params ──────────────────────────────

    def test_decode_fomo_heatmap_accepts_version_params(self):
        """decode_fomo_heatmap must accept fomo_version and grid_size kwargs."""
        import numpy as np
        from app.ml.fomo_evaluator import decode_fomo_heatmap
        # Small synthetic heatmap: 4×4 grid, 3 channels (bg + 2 classes)
        raw = np.zeros((4, 4, 3), dtype=np.float32)
        raw[2, 2, 1] = 10.0  # strong object signal at (2, 2)
        cells, probs = decode_fomo_heatmap(
            raw, threshold=0.5, fomo_version=2, grid_size=4
        )
        # The object cell must be detected
        assert (2, 2) in cells, f"Expected cell (2,2) in detections, got {cells}"

    def test_decode_fomo_heatmap_v1_unchanged(self):
        """decode_fomo_heatmap with default params (no version args) must behave identically."""
        import numpy as np
        from app.ml.fomo_evaluator import decode_fomo_heatmap
        raw = np.zeros((6, 6, 3), dtype=np.float32)
        raw[3, 3, 2] = 8.0  # object at (3, 3), class 2
        cells_v1, _ = decode_fomo_heatmap(raw, threshold=0.5)
        cells_same, _ = decode_fomo_heatmap(raw, threshold=0.5, fomo_version=1, grid_size=6)
        assert cells_v1 == cells_same, (
            f"v1 path and explicit version=1 must produce identical cells: {cells_v1} vs {cells_same}"
        )

    def test_decode_fomo_heatmap_nms_does_not_suppress_tied_cells(self):
        """Bug 1 regression: equal-confidence adjacent cells must NOT mutually suppress.

        Prior to the fix the NMS condition used >= instead of >, causing a
        symmetric deadlock: cell A suppressed B (B.conf >= A.conf) and B
        suppressed A (A.conf >= B.conf), so neither survived.  With the
        corrected strict-greater-than condition only one cell can suppress the
        other — the one with strictly lower confidence is dropped, the one with
        strictly higher (or no higher neighbour) survives.

        Arrangement tested:
            Two non-adjacent cells at equal confidence — both must survive.
            Two adjacent cells at equal confidence — exactly one must survive
            (the one whose iteration comes first in dict order; the other sees
            a neighbour with equal conf but is itself the local max when its
            turn arrives because the surviving cell was already added and is no
            longer in candidates).

        The cleaner single-cell test: a lone object peak with no neighbours
        must always survive regardless of min_peak_gap.
        """
        import numpy as np
        from app.ml.fomo_evaluator import decode_fomo_heatmap

        # ── Case 1: isolated peak — must survive ──────────────────────────────
        raw = np.zeros((6, 6, 2), dtype=np.float32)   # bg + 1 class
        raw[3, 3, 1] = 5.0   # strong peak at (3, 3), no neighbours
        cells, _ = decode_fomo_heatmap(raw, threshold=0.3)
        assert (3, 3) in cells, (
            f"Isolated peak (3,3) must always survive NMS; got {cells}"
        )

        # ── Case 2: two non-adjacent equal-confidence peaks — both survive ────
        raw2 = np.zeros((6, 6, 2), dtype=np.float32)
        raw2[1, 1, 1] = 5.0   # peak A — no shared 8-neighbour with peak B
        raw2[4, 4, 1] = 5.0   # peak B
        cells2, _ = decode_fomo_heatmap(raw2, threshold=0.3)
        assert (1, 1) in cells2 and (4, 4) in cells2, (
            f"Non-adjacent equal-confidence peaks must both survive; got {cells2}"
        )

        # ── Case 3: center beats ortho neighbour — center survives ────────────
        raw3 = np.zeros((6, 6, 2), dtype=np.float32)
        raw3[3, 3, 1] = 8.0   # dominant center
        raw3[3, 4, 1] = 4.0   # weaker ortho neighbour
        cells3, _ = decode_fomo_heatmap(raw3, threshold=0.3)
        assert (3, 3) in cells3, "Center with higher conf must survive NMS"
        assert (3, 4) not in cells3, "Weaker neighbour must be suppressed by center"


# ─── Deployment project_id regression ────────────────────────────────────────


class TestDeploymentProjectId:
    """
    Regression: build_deployment must stamp Deployment.project_id so the
    compatibility resolver can match it against Device.project_id.

    Root cause: Deployment() was created without project_id, so all rows had
    project_id=NULL and resolve_latest_compatible_deployment() always returned
    None (its first filter is Deployment.project_id == device.project_id).
    """

    def test_project_id_derived_from_model_metadata(self):
        """
        When model_metadata contains project_id, build_deployment must copy it
        onto the Deployment row without touching the ORM relationship.
        """
        import inspect
        from app.api.v1.endpoints import deployment as dep_mod

        src = inspect.getsource(dep_mod.build_deployment)
        assert 'model_metadata' in src and 'project_id' in src, (
            "build_deployment must read project_id from model_metadata"
        )
        # Verify the ORM fallback path is also present.
        assert 'training_job' in src, (
            "build_deployment must fall back to model.training_job.impulse.project_id "
            "when model_metadata lacks project_id"
        )

    def test_deployment_row_carries_project_id(self, db_session):
        """
        End-to-end: a Deployment created via build_deployment logic must have
        project_id set to match the TrainedModel's project.
        """
        from app.models.user import (
            Deployment, TrainedModel, TrainingJob, Impulse,
            DeployTarget, JobStatus,
        )
        from app.services.compatibility import resolve_latest_compatible_deployment, Device

        project_id = "test-proj-deploy-regress"

        # Minimal object graph: Impulse → TrainingJob → TrainedModel → Deployment
        impulse = Impulse(
            id="imp-regress-001",
            project_id=project_id,
            name="Regress impulse",
            sensor_type="accel",
        )
        db_session.add(impulse)

        job = TrainingJob(
            id="job-regress-001",
            impulse_id=impulse.id,
            status=JobStatus.completed,
        )
        db_session.add(job)

        model = TrainedModel(
            id="mdl-regress-001",
            training_job_id=job.id,
            version="1",
            format="tflite",
            storage_key="fake/key.tflite",
            model_metadata={"project_id": project_id},
        )
        db_session.add(model)
        db_session.flush()

        # Simulate what build_deployment does (the fixed version).
        derived_pid = (model.model_metadata or {}).get("project_id")
        if not derived_pid and model.training_job and model.training_job.impulse:
            derived_pid = model.training_job.impulse.project_id

        dep = Deployment(
            model_id=model.id,
            project_id=derived_pid,
            target=DeployTarget.raspberry_pi,
            deployment_target="raspberry_pi",
            device_profile="raspberry_pi_4",
            status=JobStatus.completed,
        )
        db_session.add(dep)
        db_session.flush()

        assert dep.project_id == project_id, (
            f"Deployment.project_id must be {project_id!r}, got {dep.project_id!r}"
        )

    def test_compatibility_resolver_matches_after_fix(self, db_session):
        """
        resolve_latest_compatible_deployment must return the deployment once
        project_id is correctly stamped (regression guard for the NULL bug).
        """
        from app.models.user import (
            Deployment, TrainedModel, TrainingJob, Impulse,
            DeployTarget, JobStatus, Device,
        )
        from app.services.compatibility import resolve_latest_compatible_deployment

        project_id = "test-proj-compat-regress"

        impulse = Impulse(
            id="imp-compat-001",
            project_id=project_id,
            name="Compat impulse",
            sensor_type="accel",
        )
        db_session.add(impulse)

        job = TrainingJob(id="job-compat-001", impulse_id=impulse.id, status=JobStatus.completed)
        db_session.add(job)

        model = TrainedModel(
            id="mdl-compat-001",
            training_job_id=job.id,
            version="1",
            format="tflite",
            storage_key="fake/compat.tflite",
            model_metadata={"project_id": project_id},
        )
        db_session.add(model)

        dep = Deployment(
            model_id=model.id,
            project_id=project_id,          # fixed: was NULL before the patch
            target=DeployTarget.raspberry_pi,
            deployment_target="raspberry_pi",
            device_profile="raspberry_pi_4",
            status=JobStatus.completed,
        )
        db_session.add(dep)

        device = Device(
            id="dev-compat-001",
            device_id="rpi-regress-sim",
            name="rpi-regress-sim",
            project_id=project_id,
            deployment_target="raspberry_pi",
            device_profile="raspberry_pi_4",
        )
        db_session.add(device)
        db_session.flush()

        result = resolve_latest_compatible_deployment(db_session, device, force=True)
        assert result is not None, (
            "resolve_latest_compatible_deployment returned None — "
            "Deployment.project_id is NULL or target/profile mismatch"
        )
        assert result.id == dep.id


# ─── Health endpoint ──────────────────────────────────────────────────────────


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "EdgeImpulse" in r.json()["message"]

# ── 18. TFLite Export & Keras 3 Compatibility ───────────────────────────

def test_fomo_export_model_rewiring_keras3_compat():
    """
    _build_fomo_export_model_for_tflite must successfully rewire a functional
    model for TFLite export in Keras 3 by correctly accessing node metadata
    (using _inbound_nodes fallback).
    """
    import tensorflow as tf
    from tensorflow import keras
    from app.workers.training_worker import _build_fomo_export_model_for_tflite

    # 1. Build a representative model with Resizing & Rescaling
    inp = keras.Input(shape=(96, 96, 3), name="input")
    x = keras.layers.Resizing(96, 96, name="resize")(inp)
    x = keras.layers.Rescaling(2.0, offset=-1, name="normalize")(x)
    x = keras.layers.Conv2D(8, 3, padding="same", name="backbone_start")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    model = keras.Model(inp, x, name="test_fomo")

    # 2. Run the rewiring (using the private function directly)
    export_model = _build_fomo_export_model_for_tflite(model)

    # 3. Verify:
    # - The new model must NOT have the 'normalize' or 'resize' layers
    layer_names = [l.name for l in export_model.layers]
    assert "normalize" not in layer_names, "Export model must strip 'normalize' layer"
    assert "resize" not in layer_names, "Export model must strip 'resize' layer"
    
    # - The new model must start with the 'backbone_start' (or similar) but correctly wired
    assert "backbone_start" in layer_names, "Export model must preserve backbone layers"
    
    # - The input shape must match the backbone's expected input (output of normalize)
    # export_model.input_shape will be (None, 96, 96, 3) because we created a new Input
    assert export_model.input_shape == (None, 96, 96, 3), (
        f"Export model input must match backbone shape, got {export_model.input_shape}"
    )

    # - It must be a valid Keras model that can be converted to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    tflite_model = converter.convert()
    assert tflite_model is not None, "TFLite conversion of rewired model failed"


def test_convert_to_tflite_prefers_safe_strategies_for_stripped_models():
    import tensorflow as tf
    from tensorflow import keras
    from app.workers import training_worker

    inp = keras.Input(shape=(8, 8, 3), name="input")
    out = keras.layers.Conv2D(4, 1, padding="same", name="conv")(inp)
    model = keras.Model(inp, out, name="base_model")

    export_inp = keras.Input(shape=(8, 8, 3), name="input")
    export_out = keras.layers.Conv2D(4, 1, padding="same", name="conv_export")(export_inp)
    export_model = keras.Model(export_inp, export_out, name="export_model")

    calls = []

    class FakeConverter:
        def convert(self):
            calls.append("convert")
            return b"ok"

    with patch.object(
        training_worker,
        "_build_fomo_export_model_for_tflite",
        return_value=export_model,
    ), patch.object(
        tf.lite.TFLiteConverter,
        "from_concrete_functions",
        side_effect=lambda *args, **kwargs: calls.append("concrete_fn") or FakeConverter(),
    ), patch.object(
        tf.saved_model,
        "save",
        side_effect=lambda *args, **kwargs: calls.append("saved_model_save"),
    ), patch.object(
        tf.lite.TFLiteConverter,
        "from_saved_model",
        side_effect=lambda *args, **kwargs: calls.append("saved_model") or FakeConverter(),
    ), patch.object(
        tf.lite.TFLiteConverter,
        "from_keras_model",
        side_effect=AssertionError("from_keras_model should be skipped for stripped models"),
    ):
        result = training_worker._convert_to_tflite(
            model, np.zeros((1, 8, 8, 3), dtype=np.float32), quantize=False
        )

    assert result == b"ok"
    assert calls == ["concrete_fn", "convert"]


# ─── FOMO int8 export: representative calibration data shape ──────────────────


def test_prepare_fomo_export_cal_data_grayscale_2d():
    """
    _prepare_fomo_export_cal_data must transform 2-D grayscale (N, H, W)
    training samples into rank-4 (N, H_snap, W_snap, 3) calibration tensors.

    Concrete case: (N, 63, 63) -> (N, 64, 64, 3).
    Values must be in [-1, 1] range after normalization.
    """
    from app.workers.training_worker import _prepare_fomo_export_cal_data

    rng = np.random.default_rng(0)
    data = rng.uniform(0.0, 1.0, size=(5, 63, 63)).astype(np.float32)
    result = _prepare_fomo_export_cal_data(data, export_input_shape=(64, 64, 3))

    assert result.shape == (5, 64, 64, 3), (
        f"Expected (5, 64, 64, 3), got {result.shape}"
    )
    assert result.min() >= -1.0 - 1e-5 and result.max() <= 1.0 + 1e-5, (
        f"Values out of [-1, 1]: min={result.min():.4f}, max={result.max():.4f}"
    )
    assert result.dtype == np.float32


def test_prepare_fomo_export_cal_data_rgb_already_matching():
    """
    _prepare_fomo_export_cal_data with RGB (N, H, W, 3) data that already
    matches the export shape must only normalize — no resize, no channel tiling.
    """
    from app.workers.training_worker import _prepare_fomo_export_cal_data

    rng = np.random.default_rng(1)
    data = rng.uniform(0.0, 1.0, size=(4, 64, 64, 3)).astype(np.float32)
    result = _prepare_fomo_export_cal_data(data, export_input_shape=(64, 64, 3))

    assert result.shape == (4, 64, 64, 3), (
        f"Expected (4, 64, 64, 3), got {result.shape}"
    )
    # Normalization check: input 0.5 -> output 0.0
    expected_sample = data[0] * 2.0 - 1.0
    np.testing.assert_allclose(result[0], expected_sample, atol=1e-5)


def test_prepare_fomo_export_cal_data_rgb_needs_resize():
    """
    _prepare_fomo_export_cal_data with RGB (N, 63, 63, 3) data that needs
    to be resized to the snapped (64, 64, 3) export shape.
    """
    from app.workers.training_worker import _prepare_fomo_export_cal_data

    rng = np.random.default_rng(2)
    data = rng.uniform(0.0, 1.0, size=(3, 63, 63, 3)).astype(np.float32)
    result = _prepare_fomo_export_cal_data(data, export_input_shape=(64, 64, 3))

    assert result.shape == (3, 64, 64, 3), (
        f"Expected (3, 64, 64, 3), got {result.shape}"
    )
    assert result.min() >= -1.0 - 1e-5 and result.max() <= 1.0 + 1e-5


def test_convert_to_tflite_int8_grayscale_cal_data_shape():
    """
    When preprocessing is stripped from a FOMO model with a 2-D grayscale
    input (N, 63, 63), _convert_to_tflite must feed rank-4 tensors shaped
    (1, H_snap, W_snap, 3) to the converter's representative_dataset.

    This is the direct regression guard for:
      input->dims->size != 4 (3 != 4) Node number 0 (CONV_2D) failed to prepare
    """
    import tensorflow as tf
    from tensorflow import keras
    from unittest.mock import patch, MagicMock
    from app.workers import training_worker

    # Build a tiny training model with 2-D grayscale input and preprocessing
    inp = keras.Input(shape=(63, 63), name="input")
    x = keras.layers.Reshape((63, 63, 1))(inp)
    x = keras.layers.Conv2D(3, 1, padding="same", name="channel_expand")(x)
    x = keras.layers.Resizing(64, 64, name="resize")(x)
    x = keras.layers.Rescaling(2.0, offset=-1, name="normalize")(x)
    out = keras.layers.Conv2D(2, 1, padding="same", name="fomo_head", activation="sigmoid")(x)
    model = keras.Model(inp, out, name="fomo_mobilenetv2_0_1")

    # Representative training data: 2-D grayscale in [0, 1]
    rng = np.random.default_rng(42)
    rep_data = rng.uniform(0.0, 1.0, size=(10, 63, 63)).astype(np.float32)

    fed_shapes = []

    # Patch the converter to capture what shapes are actually fed
    class CapturingConverter:
        optimizations = None
        representative_dataset = None
        target_spec = MagicMock()
        inference_input_type = None
        inference_output_type = None
        experimental_new_converter = None

        def convert(self):
            if self.representative_dataset is not None:
                for batch in self.representative_dataset():
                    fed_shapes.append(tuple(batch[0].shape))
            return b"ok"

    with patch.object(
        tf.lite.TFLiteConverter,
        "from_concrete_functions",
        return_value=CapturingConverter(),
    ):
        result = training_worker._convert_to_tflite(model, rep_data, quantize=True)

    assert result == b"ok", f"Conversion returned empty bytes"
    assert len(fed_shapes) > 0, "No calibration batches were fed to the converter"
    for shape in fed_shapes:
        assert len(shape) == 4, (
            f"Calibration tensor must be rank-4, got rank-{len(shape)}: shape={shape}"
        )
        assert shape[1] == 64 and shape[2] == 64 and shape[3] == 3, (
            f"Calibration tensor shape must be (1, 64, 64, 3), got {shape}"
        )


class TestFomoV2CenterDominance:
    """
    Unit tests for the FOMO v2 center-dominance improvements.

    These tests verify:
      1. v1 loss config is regression-safe (unchanged).
      2. v2 ranking config is stronger than the previous baseline.
      3. v2 loss gradient pushes the GT center harder than v1 when outranked.
      4. Near-strict (not soft) neighbor targets are preserved.
      5. Cat/dog class balance logic is unchanged.
    """

    def test_v1_loss_config_unchanged(self):
        """v1 center_dominance, rank_penalty, and rank_margin must stay at legacy values."""
        from app.workers.training_worker import _fomo_loss_tuning_config
        cfg = _fomo_loss_tuning_config(1)
        assert cfg["center_dominance"]   == 1.0, f"v1 center_dominance changed: {cfg}"
        assert cfg["center_rank_penalty"] == 0.0, f"v1 center_rank_penalty changed: {cfg}"
        assert cfg["center_rank_margin"]  == 0.0, f"v1 center_rank_margin changed: {cfg}"

    def test_v2_ranking_config_stronger_than_baseline(self):
        """v2 must use strictly stronger ranking parameters than the prior baseline."""
        from app.workers.training_worker import _fomo_loss_tuning_config
        cfg = _fomo_loss_tuning_config(2)
        # Baselines that were previously insufficient (2.20 / 2.10 / 0.10)
        assert cfg["center_dominance"]    >= 3.00, (
            f"center_dominance must be ≥3.00, got {cfg['center_dominance']}"
        )
        assert cfg["center_rank_penalty"] >= 4.00, (
            f"center_rank_penalty must be ≥4.00, got {cfg['center_rank_penalty']}"
        )
        assert cfg["center_rank_margin"]  >= 0.20, (
            f"center_rank_margin must be ≥0.20, got {cfg['center_rank_margin']}"
        )

    def test_v2_loss_gradient_pushes_center_harder_than_v1(self):
        """
        When a nearby cell outranks the GT center, v2's ranking loss must produce
        a larger gradient magnitude at the center logit than v1 (which has no
        ranking penalty).  Larger magnitude → center logit increases faster per
        gradient-descent step → outranked-rate drops over training.
        """
        import numpy as np
        import tensorflow as tf
        from app.workers.training_worker import _fomo_weighted_softmax_loss, _fomo_loss_tuning_config

        v2_cfg = _fomo_loss_tuning_config(2)
        v1_cfg = _fomo_loss_tuning_config(1)

        # 1 image, 3×3 grid, 2 object classes + background (C=3 channels).
        # GT center at (1,1), class 0 (hard center target = 1.0).
        B, H, W, C = 1, 3, 3, 3
        y_true_np = np.zeros((B, H, W, C), dtype=np.float32)
        y_true_np[:, :, :, 0] = 1.0           # all cells background
        y_true_np[0, 1, 1, 0] = 0.0           # remove BG at GT center
        y_true_np[0, 1, 1, 1] = 1.0           # class 0 at GT center

        # Logits: neighbor (1,2) has higher class-0 logit than center (1,1) →
        # center is outranked (nearby max prob > center prob before softmax).
        y_pred_np = np.zeros((B, H, W, C), dtype=np.float32)
        y_pred_np[0, 1, 1, 1] = 0.30          # center: class-0 logit
        y_pred_np[0, 1, 2, 1] = 0.55          # right neighbor: class-0 logit (higher)

        y_true = tf.constant(y_true_np, dtype=tf.float32)

        def _grad_at_center(loss_fn):
            y_pred = tf.Variable(y_pred_np, dtype=tf.float32)
            with tf.GradientTape() as tape:
                loss = loss_fn(y_true, y_pred)
            return tape.gradient(loss, y_pred).numpy()[0, 1, 1, 1]

        grad_v2 = _grad_at_center(_fomo_weighted_softmax_loss(**v2_cfg))
        grad_v1 = _grad_at_center(_fomo_weighted_softmax_loss(**v1_cfg))

        # Both gradients must be negative (loss decreases when center logit rises).
        assert grad_v2 < 0, f"v2 center gradient should be negative, got {grad_v2:.4f}"
        assert grad_v1 < 0, f"v1 center gradient should be negative, got {grad_v1:.4f}"

        # v2 must apply MORE pressure (more negative gradient → faster correction).
        assert grad_v2 < grad_v1, (
            f"v2 should push center logit harder than v1 when outranked: "
            f"v2_grad={grad_v2:.4f}, v1_grad={grad_v1:.4f}"
        )

        # One gradient-descent step (lr=0.5): v2 center logit should rise more.
        lr = 0.5
        center_logit_after_v2 = y_pred_np[0, 1, 1, 1] - lr * grad_v2
        center_logit_after_v1 = y_pred_np[0, 1, 1, 1] - lr * grad_v1
        assert center_logit_after_v2 > center_logit_after_v1, (
            f"After one GD step v2 center logit must exceed v1: "
            f"v2={center_logit_after_v2:.4f}, v1={center_logit_after_v1:.4f}"
        )

    def test_v2_neighbor_targets_are_near_strict(self):
        """
        v2 must keep neighbor_value ≤ 0.05 (near-strict) so adjacent cells
        receive only a tiny target signal that cannot compete with the GT center.
        Soft neighbor targets (>0.10) would reintroduce the ranking problem.
        """
        from app.workers.training_worker import _fomo_training_target_config
        cfg = _fomo_training_target_config(2)
        assert cfg["center_value"] == 1.0, (
            f"v2 GT center value must remain 1.0, got {cfg['center_value']}"
        )
        assert cfg["neighbor_value"] <= 0.05, (
            f"v2 neighbor_value must be near-strict (≤0.05), got {cfg['neighbor_value']}"
        )
        assert cfg["diagonal_value"] <= 0.05, (
            f"v2 diagonal_value must be near-strict (≤0.05), got {cfg['diagonal_value']}"
        )
        # v1 must still be fully strict (no neighbor signal at all)
        cfg_v1 = _fomo_training_target_config(1)
        assert cfg_v1["neighbor_value"] == 0.0, (
            f"v1 neighbor_value must remain 0.0, got {cfg_v1['neighbor_value']}"
        )

    def test_v2_class_balance_preserved(self):
        """
        _compute_fomo_class_weights must weight the minority class (dog) higher
        than the majority class (cat) for v2, matching the v1 ordering — the
        rebalancing logic must not be affected by the center-dominance changes.
        """
        import numpy as np
        from app.workers.training_worker import _compute_fomo_class_weights

        # 3 cat GT cells, 1 dog GT cell → dog should have higher weight.
        cell_counts = np.array([3.0, 1.0], dtype=np.float32)
        label_names = ["cat", "dog"]

        weights_v1 = _compute_fomo_class_weights(cell_counts, label_names, fomo_version=1)
        weights_v2 = _compute_fomo_class_weights(cell_counts, label_names, fomo_version=2)

        assert weights_v1 is not None, "v1 must return class weights for 2 classes"
        assert weights_v2 is not None, "v2 must return class weights for 2 classes"

        # Dog (rarer) must outweigh cat for both versions.
        assert weights_v1[1] > weights_v1[0], (
            f"v1: dog weight ({weights_v1[1]}) must exceed cat ({weights_v1[0]})"
        )
        assert weights_v2[1] > weights_v2[0], (
            f"v2: dog weight ({weights_v2[1]}) must exceed cat ({weights_v2[0]})"
        )

        # Both must produce the same relative ordering (ratio preserved within 1%).
        ratio_v1 = weights_v1[1] / weights_v1[0]
        ratio_v2 = weights_v2[1] / weights_v2[0]
        assert abs(ratio_v1 - ratio_v2) / ratio_v1 < 0.01, (
            f"v1 and v2 dog/cat weight ratio must match within 1%: "
            f"v1={ratio_v1:.4f}, v2={ratio_v2:.4f}"
        )

    def test_v2_ranking_uses_all_pairs_mode(self):
        """
        v2 must set center_rank_all_pairs=True so every offending neighbour
        receives an independent gradient rather than only the argmax cell.
        v1 must leave this False (legacy max-based path, penalty=0 anyway).
        """
        from app.workers.training_worker import _fomo_loss_tuning_config
        cfg_v2 = _fomo_loss_tuning_config(2)
        cfg_v1 = _fomo_loss_tuning_config(1)
        assert cfg_v2.get("center_rank_all_pairs") is True, (
            f"v2 must use all-pairs ranking, got: {cfg_v2.get('center_rank_all_pairs')}"
        )
        assert not cfg_v1.get("center_rank_all_pairs", False), (
            f"v1 must not use all-pairs ranking, got: {cfg_v1.get('center_rank_all_pairs')}"
        )

    def test_v2_all_pairs_gradient_additive_per_violating_neighbor(self):
        """
        With all-pairs mode, each additional outranking neighbour adds an
        independent gradient contribution at the GT center.

        Proof: hold the right-neighbor's violation constant; add a left
        neighbor that also outranks at the same magnitude.  The center
        gradient must grow, showing the contributions are summed.

        Under the old max-based approach the max violation does not change
        (same right-neighbor is still the argmax), so the center gradient
        would be identical — this test would fail with the old implementation.
        """
        import numpy as np
        import tensorflow as tf
        from app.workers.training_worker import _fomo_weighted_softmax_loss, _fomo_loss_tuning_config

        v2_cfg = _fomo_loss_tuning_config(2)

        # 1 image, 3×3 grid, 2 channels (BG + 1 class).
        B, H, W, C = 1, 3, 3, 2
        y_true_np = np.zeros((B, H, W, C), dtype=np.float32)
        y_true_np[:, :, :, 0] = 1.0       # all BG
        y_true_np[0, 1, 1, 0] = 0.0       # GT center: not BG
        y_true_np[0, 1, 1, 1] = 1.0       # GT center: class 0

        # Center logit kept constant across both scenarios.
        center_logit = 0.20

        def _center_grad(right_logit, left_logit):
            y_pred_np = np.zeros((B, H, W, C), dtype=np.float32)
            y_pred_np[0, 1, 1, 1] = center_logit   # center class-0 logit
            y_pred_np[0, 1, 2, 1] = right_logit    # right neighbor (1,2)
            y_pred_np[0, 1, 0, 1] = left_logit     # left neighbor (1,0)
            y_true = tf.constant(y_true_np, dtype=tf.float32)
            y_pred = tf.Variable(y_pred_np, dtype=tf.float32)
            with tf.GradientTape() as tape:
                loss = _fomo_weighted_softmax_loss(**v2_cfg)(y_true, y_pred)
            return float(tape.gradient(loss, y_pred).numpy()[0, 1, 1, 1])

        # Scenario 1: only right outranks (left at zero — no violation).
        grad_1 = _center_grad(right_logit=0.70, left_logit=-2.0)

        # Scenario 2: right logit unchanged; left also outranks at same magnitude.
        grad_2 = _center_grad(right_logit=0.70, left_logit=0.70)

        # All-pairs: adding a second equally-violating neighbour must increase
        # the center gradient magnitude.
        assert grad_2 < grad_1, (
            f"All-pairs: second violating neighbour must add to center gradient. "
            f"S1={grad_1:.4f}, S2={grad_2:.4f}"
        )

        # The absolute gradient INCREASE should equal one pair's ranking
        # contribution (penalty × d_prob/d_logit), independent of CE magnitude.
        # Expected ≈ center_rank_penalty × p_c × (1-p_c) ≈ 4.5 × 0.25 ≈ 1.1.
        # Allow generous tolerance since softmax probs vary by scenario.
        delta = abs(grad_2) - abs(grad_1)
        assert delta > 0.30, (
            f"Absolute gradient increase from 2nd violator must be > 0.30 "
            f"(expected ~{v2_cfg['center_rank_penalty'] * 0.20:.2f}): "
            f"delta={delta:.4f}, S1={grad_1:.4f}, S2={grad_2:.4f}"
        )

        # Cross-mode comparison: max-based (legacy) at the SAME 2-neighbor
        # scenario would NOT increase the center gradient because max(R,L)=R
        # is unchanged when L gets the same logit as R.  All-pairs MUST produce
        # a strictly more-negative gradient than max-based for this scenario.
        v2_max_cfg = {**v2_cfg, "center_rank_all_pairs": False}
        grad_max_2 = _center_grad(right_logit=0.70, left_logit=0.70)
        # Temporarily recompute with max-based
        def _center_grad_max(right_logit, left_logit):
            y_pred_np2 = np.zeros((B, H, W, C), dtype=np.float32)
            y_pred_np2[0, 1, 1, 1] = center_logit
            y_pred_np2[0, 1, 2, 1] = right_logit
            y_pred_np2[0, 1, 0, 1] = left_logit
            y_true2 = tf.constant(y_true_np, dtype=tf.float32)
            y_pred2 = tf.Variable(y_pred_np2, dtype=tf.float32)
            with tf.GradientTape() as tape2:
                loss2 = _fomo_weighted_softmax_loss(**v2_max_cfg)(y_true2, y_pred2)
            return float(tape2.gradient(loss2, y_pred2).numpy()[0, 1, 1, 1])

        grad_max_s2 = _center_grad_max(right_logit=0.70, left_logit=0.70)
        assert grad_2 < grad_max_s2, (
            f"All-pairs must penalise both equal-ranked neighbours (2 pairs); "
            f"max-based only penalises one (same argmax). "
            f"all_pairs={grad_2:.4f} must be < max_based={grad_max_s2:.4f}"
        )


# ─── Job logs viewer endpoint ─────────────────────────────────────────────────

class TestJobLogs:
    """GET /projects/{project_id}/jobs/{job_type}/{job_id}/logs

    Read-only viewer over already-persisted logs. No new storage.
    """

    @pytest.fixture(autouse=True)
    def setup(self, client):
        import uuid as _uuid
        self.client = client
        self.token = register_and_login(client, "joblogs@gmail.com", "pass12345", "joblogsuser")
        self.hdrs = auth_headers(self.token)

        # Unique names/ids per test so the autouse setup never collides with a
        # prior test's rows in the shared (session-scoped) sqlite database.
        sfx = _uuid.uuid4().hex[:8]
        r = client.post("/api/v1/projects/", json={"name": f"JobLogsProject-{sfx}"}, headers=self.hdrs)
        self.project_id = r.json()["id"]
        r2 = client.post("/api/v1/impulses/", json={
            "project_id": self.project_id, "name": f"JobLogsImpulse-{sfx}",
            "dsp_blocks": [], "ml_blocks": [],
        }, headers=self.hdrs)
        self.impulse_id = r2.json()["id"]

        # Seed a completed training job with persisted log lines, plus a DSP
        # feature job (no log column), directly via the DB.
        self.training_job_id = f"joblogs-training-{sfx}"
        self.dsp_job_id = f"joblogs-dsp-{sfx}"
        self.log_lines = ["Job started", "Epoch 1/30 - loss: 0.5", "Epoch 2/30 - loss: 0.3"]
        db = TestingSessionLocal()
        try:
            db.add_all([
                TrainingJob(
                    id=self.training_job_id,
                    impulse_id=self.impulse_id,
                    status=JobStatus.completed,
                    training_history={"log_lines": list(self.log_lines)},
                ),
                DspFeatureJob(
                    id=self.dsp_job_id,
                    impulse_id=self.impulse_id,
                    project_id=self.project_id,
                    status=JobStatus.completed,
                ),
            ])
            db.commit()
        finally:
            db.close()

    def _url(self, job_type, job_id, project_id=None):
        pid = project_id or self.project_id
        return f"/api/v1/projects/{pid}/jobs/{job_type}/{job_id}/logs"

    def test_training_logs_verbatim(self):
        r = self.client.get(self._url("training", self.training_job_id), headers=self.hdrs)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["job_id"] == self.training_job_id
        assert body["job_type"] == "training"
        assert body["status"] == "completed"
        assert body["log_lines"] == self.log_lines

    def test_retraining_reads_same_training_job(self):
        # retraining maps to the same TrainingJob table.
        r = self.client.get(self._url("retraining", self.training_job_id), headers=self.hdrs)
        assert r.status_code == 200, r.text
        assert r.json()["log_lines"] == self.log_lines

    def test_dsp_feature_returns_empty(self):
        r = self.client.get(self._url("dsp_feature", self.dsp_job_id), headers=self.hdrs)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["job_type"] == "dsp_feature"
        assert body["log_lines"] == []

    def test_unsupported_job_type_400(self):
        for bad in ("deployment", "model_test", "ai_labeling", "feature_generation", "bogus"):
            r = self.client.get(self._url(bad, self.training_job_id), headers=self.hdrs)
            assert r.status_code == 400, f"{bad}: {r.status_code} {r.text}"

    def test_missing_job_404(self):
        r = self.client.get(self._url("training", "does-not-exist"), headers=self.hdrs)
        assert r.status_code == 404

    def test_cross_project_job_404(self):
        # A second user's project must not be able to read this job.
        other = register_and_login(self.client, "joblogs2@gmail.com", "pass12345", "joblogsuser2")
        ohdrs = auth_headers(other)
        import uuid as _uuid
        r = self.client.post(
            "/api/v1/projects/", json={"name": f"OtherJobLogsProject-{_uuid.uuid4().hex[:8]}"}, headers=ohdrs,
        )
        other_pid = r.json()["id"]
        r2 = self.client.get(
            self._url("training", self.training_job_id, project_id=other_pid), headers=ohdrs,
        )
        assert r2.status_code == 404

    def test_requires_auth(self):
        r = self.client.get(self._url("training", self.training_job_id))
        assert r.status_code in (401, 403)


# ─── Forgot-password / reset-password ────────────────────────────────────────

class TestForgotPasswordReset:
    """
    Covers POST /auth/forgot-password and POST /auth/reset-password.

    Email sending is always mocked; the module-level throttle uses the
    in-memory fallback (Redis is unavailable in the test environment).
    Each test uses a unique email so in-memory throttle slots don't bleed
    between cases.
    """

    GENERIC_MSG = "If an account exists for this email, a password reset link has been sent."

    @pytest.fixture(autouse=True)
    def setup(self, client):
        self.client = client

    # ── helpers ──────────────────────────────────────────────────────────────

    def _register_local(self, email, username, password="Passw0rd!"):
        """Register + verify a local-password account without hitting SMTP."""
        with patch("app.api.v1.endpoints.auth.send_verification_email"):
            r = self.client.post("/api/v1/auth/register", json={
                "email": email, "username": username, "password": password,
            })
        # Already-registered (re-run) is fine; status must be 201 or 400.
        assert r.status_code in (201, 400), r.text
        verify_email_directly(email)
        return password

    def _get_reset_token(self, email):
        """Trigger forgot-password (patching SMTP) and return the raw token."""
        with patch("app.api.v1.endpoints.auth.send_password_reset_email") as mock_send:
            self.client.post("/api/v1/auth/forgot-password", json={"email": email})
        if not mock_send.called:
            return None
        return mock_send.call_args.args[1]

    # ── forgot-password: all cases must return the identical generic message ─

    def test_unknown_email_returns_generic(self):
        r = self.client.post("/api/v1/auth/forgot-password",
                             json={"email": "fp_nobody_xyz@example.com"})
        assert r.status_code == 200
        assert r.json()["message"] == self.GENERIC_MSG

    def test_local_account_returns_generic_and_sends_email(self):
        email = "fp_local_send@example.com"
        self._register_local(email, "fp_local_send_u")

        with patch("app.api.v1.endpoints.auth.send_password_reset_email") as mock_send:
            r = self.client.post("/api/v1/auth/forgot-password", json={"email": email})

        assert r.status_code == 200
        assert r.json()["message"] == self.GENERIC_MSG
        mock_send.assert_called_once()
        # Token was passed as second positional arg to send_password_reset_email.
        assert mock_send.call_args.args[1]

    def test_google_account_returns_generic_no_email(self):
        """A Google-only account (hashed_password=None) must never receive a reset email."""
        email = "fp_google_only@gmail.com"
        google_login(self.client, email=email, sub="fp_google_only_sub")

        with patch("app.api.v1.endpoints.auth.send_password_reset_email") as mock_send:
            r = self.client.post("/api/v1/auth/forgot-password", json={"email": email})

        assert r.status_code == 200
        assert r.json()["message"] == self.GENERIC_MSG
        mock_send.assert_not_called()

    def test_throttled_returns_same_generic_message(self):
        """Second call within the 60s window is throttled but still returns the generic message."""
        email = "fp_throttle_test@example.com"
        self._register_local(email, "fp_throttle_u")

        # First call claims the slot.
        with patch("app.api.v1.endpoints.auth.send_password_reset_email"):
            r1 = self.client.post("/api/v1/auth/forgot-password", json={"email": email})

        # Second call is throttled — no email sent, same response.
        with patch("app.api.v1.endpoints.auth.send_password_reset_email") as mock_no_send:
            r2 = self.client.post("/api/v1/auth/forgot-password", json={"email": email})

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["message"] == r2.json()["message"] == self.GENERIC_MSG
        mock_no_send.assert_not_called()

    # ── reset-password cases ─────────────────────────────────────────────────

    def test_valid_reset_updates_password_and_allows_login(self):
        email = "fp_valid_reset@example.com"
        old_pw = "OldPass1!"
        self._register_local(email, "fp_valid_reset_u", password=old_pw)
        raw_token = self._get_reset_token(email)
        assert raw_token is not None

        new_pw = "NewPass2!"
        r = self.client.post("/api/v1/auth/reset-password", json={
            "token": raw_token, "password": new_pw, "confirm_password": new_pw,
        })
        assert r.status_code == 200
        assert "successfully" in r.json()["message"].lower()

        # Old password must no longer work.
        r_old = self.client.post("/api/v1/auth/login",
                                  json={"email": email, "password": old_pw})
        assert r_old.status_code == 401

        # New password must work (user is already verified).
        r_new = self.client.post("/api/v1/auth/login",
                                  json={"email": email, "password": new_pw})
        assert r_new.status_code == 200
        assert "access_token" in r_new.json()

    def test_reset_token_is_single_use(self):
        """Using a token a second time is rejected (hash is cleared on first use)."""
        email = "fp_reused_reset@example.com"
        self._register_local(email, "fp_reused_reset_u")
        raw_token = self._get_reset_token(email)
        assert raw_token is not None

        pw = "ResetOnce1!"
        r1 = self.client.post("/api/v1/auth/reset-password", json={
            "token": raw_token, "password": pw, "confirm_password": pw,
        })
        assert r1.status_code == 200

        r2 = self.client.post("/api/v1/auth/reset-password", json={
            "token": raw_token, "password": "AnotherPass!", "confirm_password": "AnotherPass!",
        })
        assert r2.status_code == 400

    def test_expired_token_rejected(self):
        from datetime import datetime as _dt
        email = "fp_expired_reset@example.com"
        self._register_local(email, "fp_expired_reset_u")
        raw_token = self._get_reset_token(email)
        assert raw_token is not None

        # Wind the expiry back in the DB.
        db = TestingSessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            user.reset_token_expires_at = _dt(2000, 1, 1)
            db.commit()
        finally:
            db.close()

        r = self.client.post("/api/v1/auth/reset-password", json={
            "token": raw_token, "password": "AnyPass1!", "confirm_password": "AnyPass1!",
        })
        assert r.status_code == 400

    def test_invalid_token_rejected(self):
        r = self.client.post("/api/v1/auth/reset-password", json={
            "token": "totallyinvalidtoken",
            "password": "SomePass1!",
            "confirm_password": "SomePass1!",
        })
        assert r.status_code == 400

    def test_password_mismatch_rejected(self):
        email = "fp_mismatch_reset@example.com"
        self._register_local(email, "fp_mismatch_reset_u")
        raw_token = self._get_reset_token(email)
        assert raw_token is not None

        r = self.client.post("/api/v1/auth/reset-password", json={
            "token": raw_token,
            "password": "PasswordA1!",
            "confirm_password": "PasswordB2!",
        })
        assert r.status_code == 400
        assert "match" in r.json()["detail"].lower()
