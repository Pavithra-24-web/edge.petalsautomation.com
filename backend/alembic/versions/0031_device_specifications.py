"""Device specifications — hardware characteristics per catalog entry

Revision ID: 0031_device_specifications
Revises: 0030_full_board_catalog
Create Date: 2026-08-17 00:00:00.000000

Target Device Phase 2 (docs/target_device_phase2.md). Attaches one
`device_specifications` row to every `device_catalog_entries` row seeded by
Phase 1 / 1b: the characteristics validation (Phase 4) and estimation
(Phase 5) will later compute against. Nothing reads these values yet.

Identity (`display_name`, `family`, `deploy_target`) is deliberately not
repeated here — it's composed from the catalog entry at read time so there is
exactly one owner of it. `vendor` is the one identity-adjacent field that IS
new: `family` is a product line ("STM32"), `vendor` is the company
("STMicroelectronics") — not the same thing.

**No value here is silently defaulted.** Every column is nullable except
`device_class`, and NULL means "unknown" — never zero, never an empty string,
never a figure copied from a similar-looking board. Where a board's exact
processor variant, RAM split, or clock isn't confidently sourceable from a
public datasheet, the field is left NULL rather than estimated; a plausible
invented number would be worse than an admitted gap, because Phase 5 will
compute against it as though it were real. Roughly a third of the 69 boards —
mostly Phase 1b's newer eval kits and accelerator-only chips (BrainChip,
MemryX, Synaptics, several Alif/Ambiq/Renesas/Digi/IMDT variants) — carry
partial specifications for exactly this reason. `docs/target_device_phase2_implementation.md`
lists which.

`has_ai_accelerator` is three-state on purpose (see docstring on the model in
`app/models/devices.py`): `True`/`False`/`NULL`. Where a Phase 1b
`accelerator_note` already documents a board's accelerator going unexploited,
that's a `True` here (the note describes the *build format* not using it, not
the hardware lacking it) with `ai_accelerator` naming it. Two chip families in
this catalog are modelled as sibling "(CPU)" / "(NPU-or-equivalent)" rows over
the *same physical silicon* (Digi ConnectCore 93, Renesas RZ/V2H, RZ/V2L,
IMDT V2H) — both siblings get `has_ai_accelerator = True`, because the chip
genuinely has the accelerator regardless of which row's note calls it out;
the difference between the two rows is which deploy target they resolve to,
not what's on the die. `st_stm32n6` (the plain row, as opposed to
`st_stm32n6_neural_art`) is left `NULL` rather than `False` for the same
reason — the underlying STM32N6 silicon always includes the Neural-ART NPU,
so asserting `False` would be a confident, false claim.

Fails loudly — raises rather than silently seeding a partial catalog — if any
catalog slug has no entry in `_SPECS`, or if `_SPECS` names a slug the catalog
doesn't have. A missing specification must break the migration, not leave a
board unspecified.
"""
import uuid
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa


revision = "0031_device_specifications"
down_revision = "0030_full_board_catalog"
branch_labels = None
depends_on = None


_DEVICE_CLASS_ENUM = sa.Enum(
    "microcontroller", "linux_sbc", "accelerator", "desktop", "mobile",
    name="deviceclass",
)


def _spec(device_class, **fields):
    row = {
        "vendor": None, "processor_family": None, "processor": None,
        "cpu_architecture": None, "clock_rate_mhz": None, "ram_kb": None,
        "rom_kb": None, "runtime_environment": None,
        "has_ai_accelerator": None, "ai_accelerator": None,
        "latency_budget_ms": None, "latency_budget_basis": None,
        "supported_precisions": None,
    }
    row["device_class"] = device_class
    row.update(fields)
    return row


_TFLITE_PREC = ["int8", "float32"]

# slug → specification fields. One entry per catalog row (69 total); the
# migration itself asserts this set matches the live catalog exactly.
_SPECS: dict[str, dict] = {

    # ── Phase 1 legacy slugs ────────────────────────────────────────────
    "generic_tflite": _spec(
        "desktop",
        runtime_environment="Python (TFLite runtime)",
        latency_budget_ms=200,
        latency_budget_basis="class default for a generic TFLite host (no specific hardware)",
        supported_precisions=_TFLITE_PREC,
    ),
    "arduino_nano_33_ble": _spec(
        "microcontroller", vendor="Arduino",
        processor_family="Cortex-M4F", processor="nRF52840", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=64, ram_kb=256, rom_kb=1024, runtime_environment="Arduino core",
        has_ai_accelerator=False,
        latency_budget_ms=100, latency_budget_basis="class default for Cortex-M4F at 64MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "esp32_devkit": _spec(
        "microcontroller", vendor="Espressif",
        processor_family="Xtensa LX6", processor="ESP32", cpu_architecture="Xtensa LX6 (dual-core)",
        clock_rate_mhz=240, ram_kb=520, rom_kb=4096, runtime_environment="ESP-IDF / Arduino core",
        has_ai_accelerator=False,
        latency_budget_ms=150, latency_budget_basis="class default for dual-core Xtensa LX6 at 240MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "raspberry_pi_4": _spec(
        "linux_sbc", vendor="Raspberry Pi Foundation",
        processor_family="Cortex-A72", processor="Broadcom BCM2711",
        cpu_architecture="ARMv8-A (quad-core, 64-bit)",
        clock_rate_mhz=1500, ram_kb=4194304, runtime_environment="Linux (Raspberry Pi OS, Debian-based)",
        has_ai_accelerator=False,
        latency_budget_ms=50, latency_budget_basis="class default for a 4-core Cortex-A72 Linux SBC",
        supported_precisions=_TFLITE_PREC,
    ),
    "unoq": _spec(
        "linux_sbc", vendor="Arduino",
        processor_family="Cortex-A53 (quad-core)", processor="Qualcomm Dragonwing QRB2210",
        cpu_architecture="ARMv8-A (quad-core, 64-bit) + STM32U585 (Cortex-M33) real-time coprocessor",
        clock_rate_mhz=2000, ram_kb=4194304, rom_kb=33554432,
        runtime_environment="Linux + real-time MCU coprocessor",
        has_ai_accelerator=False,
        latency_budget_ms=40,
        latency_budget_basis="class default for a quad-core Cortex-A53 Linux SBC",
        supported_precisions=_TFLITE_PREC,
    ),
    "esp32_s3": _spec(
        "microcontroller", vendor="Espressif",
        processor_family="Xtensa LX7", processor="ESP32-S3", cpu_architecture="Xtensa LX7 (dual-core)",
        clock_rate_mhz=240, ram_kb=512, rom_kb=8192, runtime_environment="ESP-IDF / Arduino core",
        has_ai_accelerator=False,
        latency_budget_ms=120, latency_budget_basis="class default for dual-core Xtensa LX7 at 240MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "raspberry_pi_5": _spec(
        "linux_sbc", vendor="Raspberry Pi Foundation",
        processor_family="Cortex-A76", processor="Broadcom BCM2712",
        cpu_architecture="ARMv8.2-A (quad-core, 64-bit)",
        clock_rate_mhz=2400, ram_kb=4194304, runtime_environment="Linux (Raspberry Pi OS, Debian-based)",
        has_ai_accelerator=False,
        latency_budget_ms=30, latency_budget_basis="class default for a 4-core Cortex-A76 Linux SBC",
        supported_precisions=_TFLITE_PREC,
    ),
    "raspberry_pi_pico": _spec(
        "microcontroller", vendor="Raspberry Pi Foundation",
        processor_family="Cortex-M0+", processor="RP2040", cpu_architecture="ARMv6-M (dual-core)",
        clock_rate_mhz=133, ram_kb=264, rom_kb=2048, runtime_environment="bare metal / Pico SDK",
        has_ai_accelerator=False,
        latency_budget_ms=300, latency_budget_basis="class default for dual Cortex-M0+ at 133MHz, no FPU",
        supported_precisions=["int8"],
    ),
    "stm32h7": _spec(
        "microcontroller", vendor="STMicroelectronics",
        processor_family="Cortex-M7", processor="STM32H7", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=480, ram_kb=1024, rom_kb=2048, runtime_environment="bare metal / Arduino core",
        has_ai_accelerator=False,
        latency_budget_ms=80, latency_budget_basis="class default for Cortex-M7 at 480MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "stm32f7": _spec(
        "microcontroller", vendor="STMicroelectronics",
        processor_family="Cortex-M7", processor="STM32F7", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=216, ram_kb=512, rom_kb=1024, runtime_environment="bare metal / Arduino core",
        has_ai_accelerator=False,
        latency_budget_ms=150, latency_budget_basis="class default for Cortex-M7 at 216MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "nrf52840": _spec(
        "microcontroller", vendor="Nordic Semiconductor",
        processor_family="Cortex-M4F", processor="nRF52840", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=64, ram_kb=256, rom_kb=1024, runtime_environment="Zephyr / bare metal",
        has_ai_accelerator=False,
        latency_budget_ms=100, latency_budget_basis="class default for Cortex-M4F at 64MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "jetson_nano": _spec(
        "linux_sbc", vendor="NVIDIA",
        processor_family="Cortex-A57", processor="NVIDIA Tegra X1",
        cpu_architecture="ARMv8-A (quad-core, 64-bit)",
        clock_rate_mhz=1430, ram_kb=4194304, runtime_environment="Linux (JetPack/L4T, Ubuntu-based)",
        has_ai_accelerator=True, ai_accelerator="128-core NVIDIA Maxwell GPU (CUDA)",
        latency_budget_ms=40,
        latency_budget_basis="class default for a Cortex-A57 Linux SBC with GPU (CPU-only build)",
        supported_precisions=_TFLITE_PREC,
    ),
    "jetson_orin_nano": _spec(
        "linux_sbc", vendor="NVIDIA",
        processor_family="Cortex-A78AE", processor="NVIDIA Orin (Ampere GPU)",
        cpu_architecture="ARMv8.2-A (6-core, 64-bit)",
        clock_rate_mhz=1700, ram_kb=8388608, runtime_environment="Linux (JetPack/L4T, Ubuntu-based)",
        has_ai_accelerator=True, ai_accelerator="NVIDIA Ampere GPU + DLA (Deep Learning Accelerator)",
        latency_budget_ms=25,
        latency_budget_basis="class default for a 6-core Cortex-A78AE Linux SBC with GPU (CPU-only build)",
        supported_precisions=_TFLITE_PREC,
    ),
    "coral_dev_board": _spec(
        "linux_sbc", vendor="Google",
        processor_family="Cortex-A53", processor="NXP i.MX 8M",
        cpu_architecture="ARMv8-A (quad-core, 64-bit)",
        clock_rate_mhz=1500, ram_kb=1048576, runtime_environment="Linux (Mendel, Debian-based)",
        has_ai_accelerator=True, ai_accelerator="Google Edge TPU",
        latency_budget_ms=45,
        latency_budget_basis="class default for a quad-core Cortex-A53 Linux SBC (CPU-only build)",
        supported_precisions=_TFLITE_PREC,
    ),
    "imx_rt1060": _spec(
        "microcontroller", vendor="NXP",
        processor_family="Cortex-M7", processor="i.MX RT1060", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=600, ram_kb=1024, rom_kb=8192, runtime_environment="bare metal / Arduino core",
        has_ai_accelerator=False,
        latency_budget_ms=60, latency_budget_basis="class default for Cortex-M7 at 600MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "himax_we_i_plus": _spec(
        "microcontroller", vendor="Himax",
        processor_family="Cortex-M4F", processor="Himax HX6537-A (WE-I Plus)",
        cpu_architecture="ARMv7E-M + ARC EM9D DSP",
        runtime_environment="bare metal / Himax SDK",
        has_ai_accelerator=True, ai_accelerator="HX6537-A ARC EM9D DSP",
        latency_budget_ms=150,
        latency_budget_basis="class default for Cortex-M4F class MCU (clock not sourced)",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── Arduino (Phase 1b) ──────────────────────────────────────────────
    "arduino_nicla_vision": _spec(
        "microcontroller", vendor="Arduino",
        processor_family="Cortex-M7+M4 (dual-core)", processor="STM32H747", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=480, ram_kb=1024, rom_kb=2048, runtime_environment="Arduino core",
        has_ai_accelerator=False,
        latency_budget_ms=80, latency_budget_basis="class default for Cortex-M7 at 480MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "arduino_nicla_vision_m4": _spec(
        "microcontroller", vendor="Arduino",
        processor_family="Cortex-M4", processor="STM32H747 (M4 core)", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=240, runtime_environment="Arduino core",
        has_ai_accelerator=False,
        latency_budget_ms=150, latency_budget_basis="class default for Cortex-M4 at 240MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "arduino_portenta_h7": _spec(
        "microcontroller", vendor="Arduino",
        processor_family="Cortex-M7+M4 (dual-core)", processor="STM32H747XI", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=480, ram_kb=1024, rom_kb=2048, runtime_environment="Arduino core",
        has_ai_accelerator=False,
        latency_budget_ms=80, latency_budget_basis="class default for Cortex-M7 at 480MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "arduino_ventuno_q": _spec(
        "linux_sbc", vendor="Arduino",
        runtime_environment="Linux + real-time MCU coprocessor",
        latency_budget_ms=100,
        latency_budget_basis="class default for a Linux-class SBC (exact processor unconfirmed)",
    ),

    # ── ESP32 (Phase 1b) ─────────────────────────────────────────────────
    "espressif_esp_eye": _spec(
        "microcontroller", vendor="Espressif",
        processor_family="Xtensa LX6", processor="ESP32", cpu_architecture="Xtensa LX6 (dual-core)",
        clock_rate_mhz=240, ram_kb=520, rom_kb=4096, runtime_environment="ESP-IDF / Arduino core",
        has_ai_accelerator=False,
        latency_budget_ms=150, latency_budget_basis="class default for dual-core Xtensa LX6 at 240MHz",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── Nordic (Phase 1b) ────────────────────────────────────────────────
    "nordic_nrf5340_dk": _spec(
        "microcontroller", vendor="Nordic Semiconductor",
        processor_family="Cortex-M33 (dual-core)", processor="nRF5340", cpu_architecture="ARMv8-M",
        clock_rate_mhz=128, ram_kb=512, rom_kb=1024, runtime_environment="Zephyr / bare metal",
        has_ai_accelerator=False,
        latency_budget_ms=90, latency_budget_basis="class default for Cortex-M33 at 128MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "nordic_nrf54l15_dk": _spec(
        "microcontroller", vendor="Nordic Semiconductor",
        processor_family="Cortex-M33", processor="nRF54L15", cpu_architecture="ARMv8-M",
        clock_rate_mhz=128, runtime_environment="Zephyr / bare metal",
        has_ai_accelerator=False,
        latency_budget_ms=90, latency_budget_basis="class default for Cortex-M33 at 128MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "nordic_nrf9151_dk": _spec(
        "microcontroller", vendor="Nordic Semiconductor",
        processor_family="Cortex-M33", processor="nRF9151", cpu_architecture="ARMv8-M",
        clock_rate_mhz=128, ram_kb=256, rom_kb=1024, runtime_environment="Zephyr / bare metal",
        has_ai_accelerator=False,
        latency_budget_ms=90, latency_budget_basis="class default for Cortex-M33 at 128MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "nordic_nrf9160_dk": _spec(
        "microcontroller", vendor="Nordic Semiconductor",
        processor_family="Cortex-M33", processor="nRF9160", cpu_architecture="ARMv8-M",
        clock_rate_mhz=64, ram_kb=256, rom_kb=1024, runtime_environment="Zephyr / bare metal",
        has_ai_accelerator=False,
        latency_budget_ms=120, latency_budget_basis="class default for Cortex-M33 at 64MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "nordic_nrf9161_dk": _spec(
        "microcontroller", vendor="Nordic Semiconductor",
        processor_family="Cortex-M33", processor="nRF9161", cpu_architecture="ARMv8-M",
        clock_rate_mhz=128, ram_kb=256, rom_kb=1024, runtime_environment="Zephyr / bare metal",
        has_ai_accelerator=False,
        latency_budget_ms=90, latency_budget_basis="class default for Cortex-M33 at 128MHz",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── NVIDIA Jetson (Phase 1b) ─────────────────────────────────────────
    "jetson_orin_nx": _spec(
        "linux_sbc", vendor="NVIDIA",
        processor_family="Cortex-A78AE", processor="NVIDIA Orin (Ampere GPU)",
        cpu_architecture="ARMv8.2-A (8-core, 64-bit)",
        clock_rate_mhz=2000, ram_kb=8388608, runtime_environment="Linux (JetPack/L4T, Ubuntu-based)",
        has_ai_accelerator=True, ai_accelerator="NVIDIA Ampere GPU + DLA (Deep Learning Accelerator)",
        latency_budget_ms=20,
        latency_budget_basis="class default for an 8-core Cortex-A78AE Linux SBC with GPU (CPU-only build)",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── STM32 (Phase 1b) ─────────────────────────────────────────────────
    "st_iot_discovery_kit": _spec(
        "microcontroller", vendor="STMicroelectronics",
        processor_family="Cortex-M33", processor="STM32U585", cpu_architecture="ARMv8-M",
        clock_rate_mhz=160, ram_kb=786, rom_kb=2048, runtime_environment="bare metal / Arduino core",
        has_ai_accelerator=False,
        latency_budget_ms=110, latency_budget_basis="class default for Cortex-M33 at 160MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "st_stm32n6": _spec(
        "microcontroller", vendor="STMicroelectronics",
        processor_family="Cortex-M55", processor="STM32N6", cpu_architecture="ARMv8.1-M",
        clock_rate_mhz=800, ram_kb=4352, runtime_environment="bare metal / Arduino core",
        latency_budget_ms=60, latency_budget_basis="class default for Cortex-M55 at 800MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "st_stm32n6_neural_art": _spec(
        "microcontroller", vendor="STMicroelectronics",
        processor_family="Cortex-M55", processor="STM32N6", cpu_architecture="ARMv8.1-M",
        clock_rate_mhz=800, ram_kb=4352, runtime_environment="bare metal / Arduino core",
        has_ai_accelerator=True, ai_accelerator="Neural-ART NPU (up to 600 GOPS)",
        latency_budget_ms=60, latency_budget_basis="class default for Cortex-M55 at 800MHz",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── Alif ─────────────────────────────────────────────────────────────
    "alif_ensemble_e7_he": _spec(
        "microcontroller", vendor="Alif Semiconductor",
        processor_family="Cortex-M55 (HE core)", processor="Alif Ensemble E7", cpu_architecture="ARMv8.1-M",
        clock_rate_mhz=160, runtime_environment="bare metal / Zephyr",
        has_ai_accelerator=True, ai_accelerator="Arm Ethos-U55 NPU",
        latency_budget_ms=100,
        latency_budget_basis="class default for Cortex-M55 class MCU (CPU-only, NPU not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),
    "alif_ensemble_e7_hp": _spec(
        "microcontroller", vendor="Alif Semiconductor",
        processor_family="Cortex-M55 (HP core)", processor="Alif Ensemble E7", cpu_architecture="ARMv8.1-M",
        clock_rate_mhz=400, runtime_environment="bare metal / Zephyr",
        has_ai_accelerator=True, ai_accelerator="Arm Ethos-U55 NPU",
        latency_budget_ms=70,
        latency_budget_basis="class default for Cortex-M55 at 400MHz (CPU-only, NPU not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── Ambiq ────────────────────────────────────────────────────────────
    "ambiq_apollo4_evb": _spec(
        "microcontroller", vendor="Ambiq",
        processor_family="Cortex-M4F", processor="Apollo4", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=192, runtime_environment="bare metal / FreeRTOS",
        has_ai_accelerator=False,
        latency_budget_ms=100, latency_budget_basis="class default for Cortex-M4F at 192MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "ambiq_apollo5_evb": _spec(
        "microcontroller", vendor="Ambiq",
        processor_family="Cortex-M55", processor="Apollo5", cpu_architecture="ARMv8.1-M",
        clock_rate_mhz=250, runtime_environment="bare metal / FreeRTOS",
        has_ai_accelerator=False,
        latency_budget_ms=90, latency_budget_basis="class default for Cortex-M55 at 250MHz",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── BrainChip (standalone accelerator chips) ────────────────────────
    "brainchip_akd1000": _spec(
        "accelerator", vendor="BrainChip", processor="AKD1000",
        runtime_environment="host-attached (PCIe/USB), MetaTF toolchain",
        has_ai_accelerator=True, ai_accelerator="Akida neuromorphic accelerator",
        latency_budget_ms=20,
        latency_budget_basis="class default for a standalone neuromorphic accelerator (host CPU-only build, NPU not exploited)",
    ),
    "brainchip_akd1500": _spec(
        "accelerator", vendor="BrainChip", processor="AKD1500",
        runtime_environment="host-attached (PCIe/M.2), MetaTF toolchain",
        has_ai_accelerator=True, ai_accelerator="Akida neuromorphic accelerator",
        latency_budget_ms=20,
        latency_budget_basis="class default for a standalone neuromorphic accelerator (host CPU-only build, NPU not exploited)",
    ),

    # ── Cortex — bare processor-class entries ───────────────────────────
    "cortex_m4f_80mhz": _spec(
        "microcontroller", vendor="Arm",
        processor_family="Cortex-M4F", processor="Cortex-M4F", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=80, runtime_environment="bare metal",
        has_ai_accelerator=False,
        latency_budget_ms=200, latency_budget_basis="class default for Cortex-M4F at 80MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "cortex_m7_216mhz": _spec(
        "microcontroller", vendor="Arm",
        processor_family="Cortex-M7", processor="Cortex-M7", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=216, runtime_environment="bare metal",
        has_ai_accelerator=False,
        latency_budget_ms=150, latency_budget_basis="class default for Cortex-M7 at 216MHz",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── Desktop ──────────────────────────────────────────────────────────
    "macbook_pro_16_2020": _spec(
        "desktop", vendor="Apple",
        processor_family="Intel Core (Comet Lake)", processor="Intel Core i7/i9 (9th/10th gen)",
        cpu_architecture="x86_64", clock_rate_mhz=2600, runtime_environment="macOS",
        has_ai_accelerator=False,
        latency_budget_ms=15, latency_budget_basis="class default for an x86_64 desktop-class CPU",
        supported_precisions=_TFLITE_PREC,
    ),
    "macbook_pro_16_2021": _spec(
        "desktop", vendor="Apple",
        processor_family="Apple Silicon", processor="Apple M1 Pro / M1 Max",
        cpu_architecture="ARM64 (Apple Silicon)", clock_rate_mhz=3200, runtime_environment="macOS",
        has_ai_accelerator=True, ai_accelerator="Apple Neural Engine (16-core)",
        latency_budget_ms=10,
        latency_budget_basis="class default for an Apple Silicon desktop-class CPU (CPU-only build, Neural Engine not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── Digi — ConnectCore 93 (CPU) / (NPU) are the same i.MX 93 silicon ──
    "digi_connectcore_93_cpu": _spec(
        "linux_sbc", vendor="Digi International",
        processor_family="Cortex-A55", processor="NXP i.MX 93",
        cpu_architecture="ARMv8.2-A (dual-core, 64-bit)",
        clock_rate_mhz=1700, runtime_environment="Linux (Yocto)",
        has_ai_accelerator=True, ai_accelerator="Arm Ethos-U65 NPU",
        latency_budget_ms=40,
        latency_budget_basis="class default for a dual-core Cortex-A55 Linux SBC (CPU-only build)",
        supported_precisions=_TFLITE_PREC,
    ),
    "digi_connectcore_93_npu": _spec(
        "linux_sbc", vendor="Digi International",
        processor_family="Cortex-A55", processor="NXP i.MX 93",
        cpu_architecture="ARMv8.2-A (dual-core, 64-bit)",
        clock_rate_mhz=1700, runtime_environment="Linux (Yocto)",
        has_ai_accelerator=True, ai_accelerator="Arm Ethos-U65 NPU",
        latency_budget_ms=40,
        latency_budget_basis="class default for a dual-core Cortex-A55 Linux SBC (CPU-only build, NPU not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── IMDT — V2H (CPU) / (RZ/V2H) are the same Renesas RZ/V2H silicon ──
    "imdt_v2h_cpu": _spec(
        "linux_sbc", vendor="IMDT",
        processor_family="Cortex-A55", processor="Renesas RZ/V2H",
        cpu_architecture="ARMv8.2-A", clock_rate_mhz=1600, runtime_environment="Linux (Yocto)",
        has_ai_accelerator=True, ai_accelerator="Renesas DRP-AI3",
        latency_budget_ms=35,
        latency_budget_basis="class default for a Cortex-A55 Linux SBC (CPU-only build)",
        supported_precisions=_TFLITE_PREC,
    ),
    "imdt_v2h_drpai": _spec(
        "linux_sbc", vendor="IMDT",
        processor_family="Cortex-A55", processor="Renesas RZ/V2H",
        cpu_architecture="ARMv8.2-A", clock_rate_mhz=1600, runtime_environment="Linux (Yocto)",
        has_ai_accelerator=True, ai_accelerator="Renesas DRP-AI3",
        latency_budget_ms=35,
        latency_budget_basis="class default for a Cortex-A55 Linux SBC (CPU-only build, DRP-AI3 not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── Infineon ─────────────────────────────────────────────────────────
    "infineon_psoc6_cy8c624": _spec(
        "microcontroller", vendor="Infineon",
        processor_family="Cortex-M4F + Cortex-M0+", processor="PSoC6 CY8C624",
        cpu_architecture="ARMv7E-M / ARMv6-M (dual-core)",
        clock_rate_mhz=150, ram_kb=1024, rom_kb=2048, runtime_environment="bare metal / FreeRTOS",
        has_ai_accelerator=False,
        latency_budget_ms=110, latency_budget_basis="class default for Cortex-M4F at 150MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "infineon_psoc6_cy8c6347": _spec(
        "microcontroller", vendor="Infineon",
        processor_family="Cortex-M4F + Cortex-M0+", processor="PSoC6 CY8C6347",
        cpu_architecture="ARMv7E-M / ARMv6-M (dual-core)",
        clock_rate_mhz=150, runtime_environment="bare metal / FreeRTOS",
        has_ai_accelerator=False,
        latency_budget_ms=110, latency_budget_basis="class default for Cortex-M4F at 150MHz",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── MemryX (standalone accelerator chip) ────────────────────────────
    "memryx_mx3": _spec(
        "accelerator", vendor="MemryX", processor="MX3",
        runtime_environment="host-attached (M.2/USB), MemryX SDK",
        has_ai_accelerator=True, ai_accelerator="MemryX MX3 accelerator",
        latency_budget_ms=20,
        latency_budget_basis="class default for a standalone edge AI accelerator (host CPU-only build)",
    ),

    # ── Microchip ────────────────────────────────────────────────────────
    "microchip_sama7d65": _spec(
        "linux_sbc", vendor="Microchip",
        processor_family="Cortex-A7", processor="SAMA7D65", cpu_architecture="ARMv7-A",
        clock_rate_mhz=1000, runtime_environment="Linux",
        has_ai_accelerator=False,
        latency_budget_ms=60, latency_budget_basis="class default for a Cortex-A7 Linux SBC",
        supported_precisions=_TFLITE_PREC,
    ),
    "microchip_sama7g54": _spec(
        "linux_sbc", vendor="Microchip",
        processor_family="Cortex-A7", processor="SAMA7G54", cpu_architecture="ARMv7-A",
        clock_rate_mhz=1000, runtime_environment="Linux",
        has_ai_accelerator=False,
        latency_budget_ms=60, latency_budget_basis="class default for a Cortex-A7 Linux SBC",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── OpenMV ───────────────────────────────────────────────────────────
    "openmv_cam_h7_plus": _spec(
        "microcontroller", vendor="OpenMV",
        processor_family="Cortex-M7", processor="STM32H743", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=480, ram_kb=1024, rom_kb=2048,
        runtime_environment="MicroPython (OpenMV firmware)",
        has_ai_accelerator=False,
        latency_budget_ms=80, latency_budget_basis="class default for Cortex-M7 at 480MHz",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── Particle ─────────────────────────────────────────────────────────
    "particle_boron": _spec(
        "microcontroller", vendor="Particle",
        processor_family="Cortex-M4F", processor="nRF52840", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=64, ram_kb=256, rom_kb=1024, runtime_environment="Particle Device OS",
        has_ai_accelerator=False,
        latency_budget_ms=100, latency_budget_basis="class default for Cortex-M4F at 64MHz",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── Reloc — no reliable public datasheet found; identity only ───────
    "reloc_brickml": _spec(
        "microcontroller", vendor="Reloc",
        latency_budget_ms=150,
        latency_budget_basis="class default for an unresearched microcontroller-class board",
    ),

    # ── Silex ────────────────────────────────────────────────────────────
    "silex_ep_200q_evk": _spec(
        "linux_sbc", vendor="Silex Technology",
        runtime_environment="Linux",
        latency_budget_ms=60,
        latency_budget_basis="class default for a Linux-class board (specific processor not yet sourced)",
    ),

    # ── Sony ─────────────────────────────────────────────────────────────
    "sony_spresense": _spec(
        "microcontroller", vendor="Sony",
        processor_family="Cortex-M4F (6-core)", processor="Sony CXD5602", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=156, ram_kb=1536, rom_kb=8192,
        runtime_environment="Spresense SDK / Arduino core",
        has_ai_accelerator=False,
        latency_budget_ms=100, latency_budget_basis="class default for Cortex-M4F at 156MHz",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── Synaptics (standalone accelerator chip) ─────────────────────────
    "synaptics_ka10000": _spec(
        "accelerator", vendor="Synaptics", processor="Synaptics KA10000",
        runtime_environment="host-attached, Synaptics SDK",
        has_ai_accelerator=True, ai_accelerator="On-chip NPU",
        latency_budget_ms=20,
        latency_budget_basis="class default for a standalone edge AI accelerator (host CPU-only build)",
    ),

    # ── Qualcomm ─────────────────────────────────────────────────────────
    "rubik_pi_3": _spec(
        "linux_sbc", vendor="Qualcomm",
        processor_family="Kryo (Cortex-A78/A55)", processor="Qualcomm QCS6490",
        cpu_architecture="ARMv8.2-A (octa-core, 64-bit)",
        clock_rate_mhz=2700, runtime_environment="Linux (Debian/Qualcomm Linux)",
        has_ai_accelerator=True, ai_accelerator="Qualcomm Hexagon NPU",
        latency_budget_ms=25,
        latency_budget_basis="class default for an octa-core Linux SBC (CPU-only build, Hexagon NPU not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── Renesas — "(CPU)" and DRP-AI-named rows are the same silicon ────
    "renesas_rz_v2h_cpu": _spec(
        "linux_sbc", vendor="Renesas",
        processor_family="Cortex-A55 (quad-core) + Cortex-R8", processor="Renesas RZ/V2H",
        cpu_architecture="ARMv8.2-A", clock_rate_mhz=1800, runtime_environment="Linux (Yocto)",
        has_ai_accelerator=True, ai_accelerator="Renesas DRP-AI3",
        latency_budget_ms=35,
        latency_budget_basis="class default for a quad-core Cortex-A55 Linux SBC (CPU-only build)",
        supported_precisions=_TFLITE_PREC,
    ),
    "renesas_rz_v2l_cpu": _spec(
        "linux_sbc", vendor="Renesas",
        processor_family="Cortex-A55 (dual-core) + Cortex-M33", processor="Renesas RZ/V2L",
        cpu_architecture="ARMv8.2-A", clock_rate_mhz=1200, runtime_environment="Linux (Yocto)",
        has_ai_accelerator=True, ai_accelerator="Renesas DRP-AI",
        latency_budget_ms=45,
        latency_budget_basis="class default for a dual-core Cortex-A55 Linux SBC (CPU-only build)",
        supported_precisions=_TFLITE_PREC,
    ),
    "renesas_rz_v2h_drpai3": _spec(
        "linux_sbc", vendor="Renesas",
        processor_family="Cortex-A55 (quad-core) + Cortex-R8", processor="Renesas RZ/V2H",
        cpu_architecture="ARMv8.2-A", clock_rate_mhz=1800, runtime_environment="Linux (Yocto)",
        has_ai_accelerator=True, ai_accelerator="Renesas DRP-AI3",
        latency_budget_ms=35,
        latency_budget_basis="class default for a quad-core Cortex-A55 Linux SBC (CPU-only build, DRP-AI3 not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),
    "renesas_rz_v2l_drpai": _spec(
        "linux_sbc", vendor="Renesas",
        processor_family="Cortex-A55 (dual-core) + Cortex-M33", processor="Renesas RZ/V2L",
        cpu_architecture="ARMv8.2-A", clock_rate_mhz=1200, runtime_environment="Linux (Yocto)",
        has_ai_accelerator=True, ai_accelerator="Renesas DRP-AI",
        latency_budget_ms=45,
        latency_budget_basis="class default for a dual-core Cortex-A55 Linux SBC (CPU-only build, DRP-AI not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── Seeed ────────────────────────────────────────────────────────────
    "seeed_wio_terminal": _spec(
        "microcontroller", vendor="Seeed Studio",
        processor_family="Cortex-M4F", processor="Microchip SAMD51", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=120, ram_kb=192, rom_kb=512, runtime_environment="Arduino core",
        has_ai_accelerator=False,
        latency_budget_ms=150, latency_budget_basis="class default for Cortex-M4F at 120MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "seeed_sensecap_a1101": _spec(
        "microcontroller", vendor="Seeed Studio",
        processor_family="Cortex-M55", processor="Himax WiseEye2 HX6538", cpu_architecture="ARMv8.1-M",
        clock_rate_mhz=400, runtime_environment="bare metal / Himax SDK",
        has_ai_accelerator=True, ai_accelerator="HX6537-A ARC DSP",
        latency_budget_ms=100,
        latency_budget_basis="class default for Cortex-M55 class MCU (CPU-only build, ARC DSP not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),
    "seeed_vision_ai_module": _spec(
        "microcontroller", vendor="Seeed Studio",
        processor_family="Cortex-M55", processor="Himax WiseEye HX6538", cpu_architecture="ARMv8.1-M",
        clock_rate_mhz=400, runtime_environment="bare metal / Himax SDK",
        has_ai_accelerator=True, ai_accelerator="HX6537-A ARC DSP",
        latency_budget_ms=100,
        latency_budget_basis="class default for Cortex-M55 class MCU (CPU-only build, ARC DSP not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── SiLabs ───────────────────────────────────────────────────────────
    "silabs_thunderboard_sense_2": _spec(
        "microcontroller", vendor="Silicon Labs",
        processor_family="Cortex-M4", processor="EFR32MG12", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=40, ram_kb=256, rom_kb=1024, runtime_environment="bare metal / Zephyr",
        has_ai_accelerator=False,
        latency_budget_ms=200, latency_budget_basis="class default for Cortex-M4 at 40MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "silabs_efr32mg24": _spec(
        "microcontroller", vendor="Silicon Labs",
        processor_family="Cortex-M33", processor="EFR32MG24", cpu_architecture="ARMv8-M",
        clock_rate_mhz=78, ram_kb=256, rom_kb=1536, runtime_environment="bare metal / Zephyr",
        has_ai_accelerator=True, ai_accelerator="MVP (Matrix Vector Processor) accelerator",
        latency_budget_ms=110,
        latency_budget_basis="class default for Cortex-M33 at 78MHz (CPU-only build, MVP not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),

    # ── Texas Instruments ────────────────────────────────────────────────
    "ti_launchxl_cc1352p": _spec(
        "microcontroller", vendor="Texas Instruments",
        processor_family="Cortex-M4F", processor="CC1352P", cpu_architecture="ARMv7E-M",
        clock_rate_mhz=48, ram_kb=80, rom_kb=352, runtime_environment="TI-RTOS / bare metal",
        has_ai_accelerator=False,
        latency_budget_ms=250, latency_budget_basis="class default for Cortex-M4F at 48MHz",
        supported_precisions=_TFLITE_PREC,
    ),
    "ti_am62a": _spec(
        "linux_sbc", vendor="Texas Instruments",
        processor_family="Cortex-A53 (quad-core)", processor="TI AM62A", cpu_architecture="ARMv8-A",
        clock_rate_mhz=1400, runtime_environment="Linux",
        has_ai_accelerator=True, ai_accelerator="TI C7x DSP + MMA (Deep Learning Accelerator)",
        latency_budget_ms=40,
        latency_budget_basis="class default for a quad-core Cortex-A53 Linux SBC (CPU-only build, DLA not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),
    "ti_am68a": _spec(
        "linux_sbc", vendor="Texas Instruments",
        processor_family="Cortex-A72 (dual-core)", processor="TI AM68A", cpu_architecture="ARMv8-A",
        clock_rate_mhz=2000, runtime_environment="Linux",
        has_ai_accelerator=True, ai_accelerator="TI C7x DSP + MMA (Deep Learning Accelerator)",
        latency_budget_ms=30,
        latency_budget_basis="class default for a dual-core Cortex-A72 Linux SBC (CPU-only build, DLA not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),
    "ti_tda4vm": _spec(
        "linux_sbc", vendor="Texas Instruments",
        processor_family="Cortex-A72 (dual-core)", processor="TI TDA4VM (Jacinto 7)",
        cpu_architecture="ARMv8-A",
        clock_rate_mhz=2000, runtime_environment="Linux",
        has_ai_accelerator=True, ai_accelerator="Matrix Multiply Accelerator (MMA) + C7x DSP",
        latency_budget_ms=30,
        latency_budget_basis="class default for a dual-core Cortex-A72 Linux SBC (CPU-only build, MMA not exploited)",
        supported_precisions=_TFLITE_PREC,
    ),
}


def upgrade() -> None:
    op.create_table(
        "device_specifications",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "catalog_entry_id", sa.String(),
            sa.ForeignKey("device_catalog_entries.id", ondelete="CASCADE"),
            nullable=False, unique=True,
        ),
        sa.Column("vendor", sa.String(), nullable=True),
        sa.Column("device_class", _DEVICE_CLASS_ENUM, nullable=False),
        sa.Column("processor_family", sa.String(), nullable=True),
        sa.Column("processor", sa.String(), nullable=True),
        sa.Column("cpu_architecture", sa.String(), nullable=True),
        sa.Column("clock_rate_mhz", sa.Integer(), nullable=True),
        sa.Column("ram_kb", sa.Integer(), nullable=True),
        sa.Column("rom_kb", sa.Integer(), nullable=True),
        sa.Column("runtime_environment", sa.String(), nullable=True),
        sa.Column("has_ai_accelerator", sa.Boolean(), nullable=True),
        sa.Column("ai_accelerator", sa.String(), nullable=True),
        sa.Column("latency_budget_ms", sa.Integer(), nullable=True),
        sa.Column("latency_budget_basis", sa.String(), nullable=True),
        sa.Column("supported_precisions", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    conn = op.get_bind()
    catalog_rows = conn.execute(
        sa.text("SELECT id, slug FROM device_catalog_entries")
    ).fetchall()

    catalog_slugs = {slug for _id, slug in catalog_rows}
    missing = catalog_slugs - set(_SPECS)
    if missing:
        raise RuntimeError(
            f"No specification seeded for catalog slug(s): {sorted(missing)}"
        )
    extra = set(_SPECS) - catalog_slugs
    if extra:
        raise RuntimeError(
            f"Specification seeded for unknown catalog slug(s): {sorted(extra)}"
        )

    table = sa.table(
        "device_specifications",
        sa.column("id", sa.String()),
        sa.column("catalog_entry_id", sa.String()),
        sa.column("vendor", sa.String()),
        sa.column("device_class", sa.String()),
        sa.column("processor_family", sa.String()),
        sa.column("processor", sa.String()),
        sa.column("cpu_architecture", sa.String()),
        sa.column("clock_rate_mhz", sa.Integer()),
        sa.column("ram_kb", sa.Integer()),
        sa.column("rom_kb", sa.Integer()),
        sa.column("runtime_environment", sa.String()),
        sa.column("has_ai_accelerator", sa.Boolean()),
        sa.column("ai_accelerator", sa.String()),
        sa.column("latency_budget_ms", sa.Integer()),
        sa.column("latency_budget_basis", sa.String()),
        sa.column("supported_precisions", sa.JSON()),
        sa.column("created_at", sa.DateTime()),
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    op.bulk_insert(table, [
        {
            "id": str(uuid.uuid4()),
            "catalog_entry_id": entry_id,
            "created_at": now,
            **_SPECS[slug],
        }
        for entry_id, slug in catalog_rows
    ])


def downgrade() -> None:
    op.drop_table("device_specifications")
    _DEVICE_CLASS_ENUM.drop(op.get_bind(), checkfirst=True)
