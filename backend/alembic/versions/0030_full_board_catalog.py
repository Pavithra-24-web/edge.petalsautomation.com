"""Full board catalog — 53 boards, 21 families

Revision ID: 0030_full_board_catalog
Revises: 0029_project_target_device
Create Date: 2026-08-17 00:00:00.000000

Target Device Phase 1b (docs/target_device_phase1b.md). Grows the device
catalog seeded in Phase 1 (0028_device_catalog, 16 rows) to the scale a real
device picker needs: 53 more boards across 21 new vendor families, on top of
the 9 Phase 1 already had. No schema change — same 6-column identity shape,
same `deploytarget` enum, same read-only API.

Every row here resolves to one of the 7 existing deploy targets. Boards whose
silicon carries an accelerator this catalog can't exploit (an NPU, a DSP, a
GPU with no matching build format) get that stated in `accelerator_note`
rather than the board being dropped or the limit worked around — the read-only
API and the deployment worker (`_supported_device_profiles`, unchanged by this
migration) both surface it.

`display_name` never encodes a clock rate or memory figure — that's Phase 2
specification data, composed at render time so it can't drift out of sync
with a separately-stored spec. The one deliberate exception is the `Cortex`
family: those two rows are bare processor-class entries with no board around
them, so the clock rate *is* the identity, not an appended specification
(the same reasoning that keeps "(CPU)" vs "(DRP-AI3)" in names elsewhere in
this file — those distinguish rows for the same board, not describe one).

`downgrade()` removes exactly these 53 rows by slug, never truncates the
table — Phase 1's 16 rows (and the 5 legacy device-profile slugs among them)
must survive a downgrade untouched.
"""
import uuid
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa


revision = "0030_full_board_catalog"
down_revision = "0029_project_target_device"
branch_labels = None
depends_on = None


_RPI_NOTE = "runs via the generic Raspberry Pi Python package on CPU only."
_CPP_NOTE = "runs via the generic C++ build on the CPU only."
_TFLITE_NOTE = "runs the generic TFLite package on CPU only."

# (slug, display_name, family, deploy_target, accelerator_note)
_SEED_ENTRIES = [
    # ── Arduino ──────────────────────────────────────────────────────────
    ("arduino_nicla_vision",    "Arduino Nicla Vision",    "Arduino", "arduino", None),
    ("arduino_nicla_vision_m4", "Arduino Nicla Vision M4", "Arduino", "arduino", None),
    ("arduino_portenta_h7",     "Arduino Portenta H7",     "Arduino", "arduino", None),
    ("arduino_ventuno_q",       "Arduino VENTUNO Q",       "Arduino", "unoq",    None),

    # ── ESP32 ────────────────────────────────────────────────────────────
    ("espressif_esp_eye", "Espressif ESP-EYE", "ESP32", "esp32", None),

    # ── Nordic ───────────────────────────────────────────────────────────
    ("nordic_nrf5340_dk",  "Nordic nRF5340 DK",  "Nordic", "cpp", None),
    ("nordic_nrf54l15_dk", "Nordic nRF54L15 DK", "Nordic", "cpp", None),
    ("nordic_nrf9151_dk",  "Nordic nRF9151 DK",  "Nordic", "cpp", None),
    ("nordic_nrf9160_dk",  "Nordic nRF9160 DK",  "Nordic", "cpp", None),
    ("nordic_nrf9161_dk",  "Nordic nRF9161 DK",  "Nordic", "cpp", None),

    # ── NVIDIA Jetson ────────────────────────────────────────────────────
    (
        "jetson_orin_nx", "NVIDIA Jetson Orin NX", "NVIDIA Jetson", "raspberry_pi",
        f"CUDA / TensorRT not exploited; {_RPI_NOTE}",
    ),

    # ── STM32 ────────────────────────────────────────────────────────────
    ("st_iot_discovery_kit", "ST IoT Discovery Kit",   "STM32", "cpp", None),
    ("st_stm32n6",           "ST STM32N6",              "STM32", "cpp", None),
    (
        "st_stm32n6_neural_art", "ST STM32N6 (Neural-ART)", "STM32", "cpp",
        f"Neural-ART NPU not exploited; {_CPP_NOTE}",
    ),

    # ── Alif ─────────────────────────────────────────────────────────────
    (
        "alif_ensemble_e7_he", "Alif Ensemble E7 HE", "Alif", "cpp",
        f"Ethos-U55 NPU not exploited; {_CPP_NOTE}",
    ),
    (
        "alif_ensemble_e7_hp", "Alif Ensemble E7 HP", "Alif", "cpp",
        f"Ethos-U55 NPU not exploited; {_CPP_NOTE}",
    ),

    # ── Ambiq ────────────────────────────────────────────────────────────
    ("ambiq_apollo4_evb", "Ambiq Apollo4 EVB", "Ambiq", "cpp", None),
    ("ambiq_apollo5_evb", "Ambiq Apollo5 EVB", "Ambiq", "cpp", None),

    # ── BrainChip ────────────────────────────────────────────────────────
    (
        "brainchip_akd1000", "BrainChip AKD1000", "BrainChip", "cpp",
        f"Akida neuromorphic accelerator not exploited; {_CPP_NOTE}",
    ),
    (
        "brainchip_akd1500", "BrainChip AKD1500", "BrainChip", "cpp",
        f"Akida neuromorphic accelerator not exploited; {_CPP_NOTE}",
    ),

    # ── Cortex — bare processor-class entries, not products. The clock
    # rate distinguishes the entry itself, so it stays in display_name here
    # (see module docstring). ─────────────────────────────────────────────
    ("cortex_m4f_80mhz",  "Cortex-M4F 80MHz",  "Cortex", "cpp", None),
    ("cortex_m7_216mhz",  "Cortex-M7 216MHz",  "Cortex", "cpp", None),

    # ── Desktop ──────────────────────────────────────────────────────────
    ("macbook_pro_16_2020", "MacBook Pro 16\" 2020", "Desktop", "tflite", None),
    (
        "macbook_pro_16_2021", "MacBook Pro 16\" 2021", "Desktop", "tflite",
        f"Apple Neural Engine not exploited; {_TFLITE_NOTE}",
    ),

    # ── Digi ─────────────────────────────────────────────────────────────
    ("digi_connectcore_93_cpu", "Digi ConnectCore 93 (CPU)", "Digi", "raspberry_pi", None),
    (
        "digi_connectcore_93_npu", "Digi ConnectCore 93 (NPU)", "Digi", "raspberry_pi",
        f"Ethos-U65 NPU not exploited; {_RPI_NOTE}",
    ),

    # ── IMDT ─────────────────────────────────────────────────────────────
    ("imdt_v2h_cpu", "IMDT V2H (CPU)", "IMDT", "raspberry_pi", None),
    (
        "imdt_v2h_drpai", "IMDT V2H (RZ/V2H)", "IMDT", "raspberry_pi",
        f"DRP-AI3 not exploited; {_RPI_NOTE}",
    ),

    # ── Infineon ─────────────────────────────────────────────────────────
    ("infineon_psoc6_cy8c624",  "Infineon PSoC6 CY8C624",  "Infineon", "cpp", None),
    ("infineon_psoc6_cy8c6347", "Infineon PSoC6 CY8C6347", "Infineon", "cpp", None),

    # ── MemryX ───────────────────────────────────────────────────────────
    (
        "memryx_mx3", "MemryX MX3", "MemryX", "tflite",
        f"MX3 accelerator not exploited; {_TFLITE_NOTE}",
    ),

    # ── Microchip ────────────────────────────────────────────────────────
    ("microchip_sama7d65", "Microchip SAMA7D65 Evaluation Kit", "Microchip", "raspberry_pi", None),
    ("microchip_sama7g54", "Microchip SAMA7G54 Evaluation Kit", "Microchip", "raspberry_pi", None),

    # ── OpenMV ───────────────────────────────────────────────────────────
    ("openmv_cam_h7_plus", "OpenMV Cam H7 Plus", "OpenMV", "cpp", None),

    # ── Particle ─────────────────────────────────────────────────────────
    ("particle_boron", "Particle Boron", "Particle", "cpp", None),

    # ── Reloc ────────────────────────────────────────────────────────────
    ("reloc_brickml", "BrickML", "Reloc", "cpp", None),

    # ── Silex ────────────────────────────────────────────────────────────
    ("silex_ep_200q_evk", "Silex Technology EP-200Q-EVK", "Silex", "raspberry_pi", None),

    # ── Sony ─────────────────────────────────────────────────────────────
    ("sony_spresense", "Sony Spresense", "Sony", "cpp", None),

    # ── Synaptics ────────────────────────────────────────────────────────
    (
        "synaptics_ka10000", "Synaptics KA10000", "Synaptics", "cpp",
        f"On-chip NPU not exploited; {_CPP_NOTE}",
    ),

    # ── Qualcomm ─────────────────────────────────────────────────────────
    (
        "rubik_pi_3", "Rubik Pi 3", "Qualcomm", "raspberry_pi",
        f"Hexagon NPU not exploited; {_RPI_NOTE}",
    ),

    # ── Renesas ──────────────────────────────────────────────────────────
    ("renesas_rz_v2h_cpu", "Renesas RZ/V2H (CPU)", "Renesas", "raspberry_pi", None),
    ("renesas_rz_v2l_cpu", "Renesas RZ/V2L (CPU)", "Renesas", "raspberry_pi", None),
    (
        "renesas_rz_v2h_drpai3", "Renesas RZ/V2H (DRP-AI3)", "Renesas", "raspberry_pi",
        f"DRP-AI not exploited; {_RPI_NOTE}",
    ),
    (
        "renesas_rz_v2l_drpai", "Renesas RZ/V2L (DRP-AI)", "Renesas", "raspberry_pi",
        f"DRP-AI not exploited; {_RPI_NOTE}",
    ),

    # ── Seeed ────────────────────────────────────────────────────────────
    ("seeed_wio_terminal", "Seeed Studio Wio Terminal", "Seeed", "cpp", None),
    (
        "seeed_sensecap_a1101", "Seeed SenseCAP A1101", "Seeed", "cpp",
        f"HX6537-A ARC DSP not exploited; {_CPP_NOTE}",
    ),
    (
        "seeed_vision_ai_module", "Seeed Vision AI Module", "Seeed", "cpp",
        f"HX6537-A ARC DSP not exploited; {_CPP_NOTE}",
    ),

    # ── SiLabs ───────────────────────────────────────────────────────────
    ("silabs_thunderboard_sense_2", "SiLabs Thunderboard Sense 2", "SiLabs", "cpp", None),
    (
        "silabs_efr32mg24", "SiLabs EFR32MG24", "SiLabs", "cpp",
        f"MVP matrix accelerator not exploited; {_CPP_NOTE}",
    ),

    # ── Texas Instruments ────────────────────────────────────────────────
    ("ti_launchxl_cc1352p", "TI LAUNCHXL-CC1352P", "Texas Instruments", "cpp", None),
    (
        "ti_am62a", "TI AM62A", "Texas Instruments", "raspberry_pi",
        f"Deep Learning Accelerator not exploited; {_RPI_NOTE}",
    ),
    (
        "ti_am68a", "TI AM68A", "Texas Instruments", "raspberry_pi",
        f"Deep Learning Accelerator not exploited; {_RPI_NOTE}",
    ),
    (
        "ti_tda4vm", "TI TDA4VM", "Texas Instruments", "raspberry_pi",
        f"Matrix multiply accelerator (MMA) not exploited; {_RPI_NOTE}",
    ),
]

assert len(_SEED_ENTRIES) == 53, f"expected 53 new rows, got {len(_SEED_ENTRIES)}"


def upgrade() -> None:
    table = sa.table(
        "device_catalog_entries",
        sa.column("id", sa.String()),
        sa.column("slug", sa.String()),
        sa.column("display_name", sa.String()),
        sa.column("family", sa.String()),
        sa.column("deploy_target", sa.String()),
        sa.column("accelerator_note", sa.String()),
        sa.column("created_at", sa.DateTime()),
    )

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    op.bulk_insert(table, [
        {
            "id": str(uuid.uuid4()),
            "slug": slug,
            "display_name": display_name,
            "family": family,
            "deploy_target": deploy_target,
            "accelerator_note": accelerator_note,
            "created_at": now,
        }
        for slug, display_name, family, deploy_target, accelerator_note in _SEED_ENTRIES
    ])


def downgrade() -> None:
    device_catalog_entries = sa.table(
        "device_catalog_entries",
        sa.column("slug", sa.String()),
    )
    slugs = [slug for slug, *_ in _SEED_ENTRIES]
    op.execute(
        device_catalog_entries.delete().where(device_catalog_entries.c.slug.in_(slugs))
    )
