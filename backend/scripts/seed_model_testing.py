"""
Seed script — Model Testing data
=================================
Creates the minimum set of records required for the Model Testing page to
render with realistic data out of the box.

Seeded records
--------------
- 1 User       (demo@petaledge.io / demo1234)
- 1 Project    ("test")
- 1 Impulse    ("Impulse #7")
- 3 ModelVersions (int8 active, float32, EON)
- 7 ModelTestSamples (mixed pass/fail as per Edge Impulse screenshot)
- 1 ModelTestRun    (accuracy = 57.14 %)
- 3 MetricResults   (Precision / Recall / F1 non-background = 0.00)

Run
---
    cd backend
    python -m scripts.seed_model_testing

The script is idempotent: it checks for existing records by name/email before
inserting, so running it multiple times is safe.
"""
import sys
import os

# ── path bootstrap (run from repo root or backend/) ──────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_backend = os.path.dirname(_here)
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from datetime import datetime
from app.core.database import SessionLocal
from app.core.auth import get_password_hash
from app.models.user import User, Project, Impulse
from app.models.model_testing import (
    ModelVersion, ModelTestRun, MetricResult, ModelTestSample,
    TestResultStatus, QuantizationType,
)

# ─── Seed constants ───────────────────────────────────────────────────────────

SEED_USER_EMAIL    = "demo@petaledge.io"
SEED_USER_NAME     = "demo"
SEED_USER_PASSWORD = "demo1234"

SEED_PROJECT_NAME = "test"
SEED_IMPULSE_NAME = "Impulse #7"

SEED_SAMPLES = [
    # (sample_name, f1_score, result_status, expected_outcome)
    ("000005",   100.0, TestResultStatus.pass_,  "-"),
    ("000012",   100.0, TestResultStatus.pass_,  "-"),
    ("000001",     0.0, TestResultStatus.fail,   "-"),
    ("000019",   100.0, TestResultStatus.pass_,  "-"),
    ("000004_t", 100.0, TestResultStatus.pass_,  "-"),
    ("000015_t",   0.0, TestResultStatus.fail,   "-"),
    ("000012_t",   0.0, TestResultStatus.fail,   "-"),
]

SEED_ACCURACY = 57.14          # 4/7 pass ≈ 57.14 %

SEED_METRICS = [
    ("precision_non_background", "Precision (non-background)", 0.00),
    ("recall_non_background",    "Recall (non-background)",    0.00),
    ("f1_score_non_background",  "F1 Score (non-background)",  0.00),
]

SEED_MODEL_VERSIONS = [
    # (version_number, name, quantization_type, is_active)
    ("v1", "Quantized (int8)", QuantizationType.int8,    True),
    ("v2", "Float32",          QuantizationType.float32, False),
    ("v3", "EON Compiled",     QuantizationType.eon,     False),
]


# ─── Main ─────────────────────────────────────────────────────────────────────

def seed() -> None:
    db = SessionLocal()
    try:
        now = datetime.utcnow()

        # ── User ──────────────────────────────────────────────────────────────
        user = db.query(User).filter(User.email == SEED_USER_EMAIL).first()
        if not user:
            user = User(
                email           = SEED_USER_EMAIL,
                username        = SEED_USER_NAME,
                hashed_password = get_password_hash(SEED_USER_PASSWORD),
                role            = "developer",
                is_active       = True,
                created_at      = now,
                updated_at      = now,
            )
            db.add(user)
            db.flush()
            print(f"  [+] User created: {SEED_USER_EMAIL}")
        else:
            print(f"  [~] User exists:  {SEED_USER_EMAIL}")

        # ── Project ───────────────────────────────────────────────────────────
        project = (
            db.query(Project)
            .filter(Project.owner_id == user.id, Project.name == SEED_PROJECT_NAME)
            .first()
        )
        if not project:
            project = Project(
                owner_id   = user.id,
                name       = SEED_PROJECT_NAME,
                created_at = now,
                updated_at = now,
            )
            db.add(project)
            db.flush()
            print(f"  [+] Project created: {SEED_PROJECT_NAME}")
        else:
            print(f"  [~] Project exists:  {SEED_PROJECT_NAME}")

        # ── Impulse ───────────────────────────────────────────────────────────
        impulse = (
            db.query(Impulse)
            .filter(Impulse.project_id == project.id, Impulse.name == SEED_IMPULSE_NAME)
            .first()
        )
        if not impulse:
            impulse = Impulse(
                project_id  = project.id,
                name        = SEED_IMPULSE_NAME,
                input_type  = "image",
                image_width = 96,
                image_height= 96,
                created_at  = now,
                updated_at  = now,
            )
            db.add(impulse)
            db.flush()
            print(f"  [+] Impulse created: {SEED_IMPULSE_NAME}")
        else:
            print(f"  [~] Impulse exists:  {SEED_IMPULSE_NAME}")

        # ── ModelVersions ─────────────────────────────────────────────────────
        versions: dict[str, ModelVersion] = {}
        for vnum, vname, qtype, active in SEED_MODEL_VERSIONS:
            existing = (
                db.query(ModelVersion)
                .filter(
                    ModelVersion.impulse_id    == impulse.id,
                    ModelVersion.version_number == vnum,
                )
                .first()
            )
            if not existing:
                v = ModelVersion(
                    project_id       = project.id,
                    impulse_id       = impulse.id,
                    version_number   = vnum,
                    name             = vname,
                    quantization_type= qtype,
                    is_active        = active,
                    created_at       = now,
                )
                db.add(v)
                db.flush()
                versions[vnum] = v
                print(f"  [+] ModelVersion: {vname}")
            else:
                versions[vnum] = existing
                print(f"  [~] ModelVersion exists: {vname}")

        active_version = versions["v1"]

        # ── ModelTestRun ──────────────────────────────────────────────────────
        existing_run = (
            db.query(ModelTestRun)
            .filter(
                ModelTestRun.impulse_id       == impulse.id,
                ModelTestRun.model_version_id == active_version.id,
            )
            .first()
        )
        if not existing_run:
            passed = sum(1 for _, f1, _, _ in SEED_SAMPLES if f1 >= 50)
            failed = sum(1 for _, f1, _, _ in SEED_SAMPLES if f1 < 50)
            run = ModelTestRun(
                project_id       = project.id,
                impulse_id       = impulse.id,
                model_version_id = active_version.id,
                accuracy         = SEED_ACCURACY,
                total_samples    = len(SEED_SAMPLES),
                passed_samples   = passed,
                failed_samples   = failed,
                created_at       = now,
            )
            db.add(run)
            db.flush()
            print(f"  [+] ModelTestRun: accuracy={SEED_ACCURACY}%")

            # MetricResults
            for mname, mdisplay, mvalue in SEED_METRICS:
                db.add(MetricResult(
                    test_run_id         = run.id,
                    metric_name         = mname,
                    metric_display_name = mdisplay,
                    metric_value        = mvalue,
                    created_at          = now,
                ))
            print(f"  [+] MetricResults: {len(SEED_METRICS)} rows")
        else:
            run = existing_run
            print(f"  [~] ModelTestRun exists: accuracy={existing_run.accuracy}%")

        # ── ModelTestSamples ──────────────────────────────────────────────────
        for sname, f1, status, expected in SEED_SAMPLES:
            exists = (
                db.query(ModelTestSample)
                .filter(
                    ModelTestSample.impulse_id  == impulse.id,
                    ModelTestSample.sample_name == sname,
                )
                .first()
            )
            if not exists:
                db.add(ModelTestSample(
                    project_id       = project.id,
                    impulse_id       = impulse.id,
                    test_run_id      = run.id,
                    sample_name      = sname,
                    expected_outcome = expected,
                    f1_score         = f1,
                    result_status    = status,
                    created_at       = now,
                    updated_at       = now,
                ))
                print(f"  [+] TestSample: {sname:12s} f1={f1:5.1f}  {status.value}")
            else:
                print(f"  [~] TestSample exists: {sname}")

        db.commit()
        print("\n  Seed complete.")

    except Exception as exc:
        db.rollback()
        print(f"\n  ERROR: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("Seeding model testing data …\n")
    seed()
