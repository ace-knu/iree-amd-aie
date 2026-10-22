#!/usr/bin/env python3
"""Read the NPU power mode (read-only).

Uses DRM_IOCTL_AMDXDNA_GET_INFO with DRM_AMDXDNA_GET_POWER_MODE. Values follow the
amdxdna uapi (amdxdna_accel.h, enum amdxdna_power_mode_type): 0 default, 1 low,
2 medium, 3 high, 4 turbo. Never calls SET_STATE; changing the mode is root-only
and device-wide.

The power mode changes NPU timings (in the default mode the driver changes the DPM
level on every hw context create/destroy), so record it next to every NPU
measurement. First written for docs/2026-10-22/4_NPU_dispatch_고정간격_조사 (T6).

usage: getpm.py [TAG] [--name] [--device PATH]
  default output: "power_mode=<n> (<name>) tag=<TAG> time=<ISO 8601>"
  --name: print only the mode name (for scripts)
  On failure prints "power_mode=? (<error>) ..." (or "unknown" with --name) and exits 1.
"""
import argparse
import ctypes
import datetime
import fcntl
import os
import struct
import sys

NAMES = {0: "default", 1: "low", 2: "medium", 3: "high", 4: "turbo"}
# DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDXDNA_GET_INFO, struct amdxdna_drm_get_info)
DRM_IOCTL_AMDXDNA_GET_INFO = 0xC0106447
DRM_AMDXDNA_GET_POWER_MODE = 9


def read_power_mode(device):
    buf = ctypes.create_string_buffer(8)  # struct amdxdna_drm_get_power_mode
    # struct amdxdna_drm_get_info { u32 param; u32 buffer_size; u64 buffer; }
    arg = bytearray(struct.pack("IIQ", DRM_AMDXDNA_GET_POWER_MODE, 8, ctypes.addressof(buf)))
    fd = os.open(device, os.O_RDWR)
    try:
        fcntl.ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, arg, True)
    finally:
        os.close(fd)
    return buf.raw[0]


def main():
    p = argparse.ArgumentParser(description="Read the NPU power mode (read-only).")
    p.add_argument("tag", nargs="?", default="")
    p.add_argument("--name", action="store_true", help="print only the mode name")
    p.add_argument("--device", default="/dev/accel/accel0")
    a = p.parse_args()
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        m = read_power_mode(a.device)
    except OSError as e:
        print("unknown" if a.name else f"power_mode=? ({e}) tag={a.tag} time={now}")
        return 1
    name = NAMES.get(m, f"unknown({m})")
    print(name if a.name else f"power_mode={m} ({name}) tag={a.tag} time={now}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
