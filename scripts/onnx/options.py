"""iree-compile options for running an ONNX model on amd-aie, grouped by function.

DEVICES holds, per device, the `compile` flags (iree-compile) that declare it
and the `run` flags (iree-run-module) that open it; `{symbol}` is filled with
the device symbol (the name with '-' as '_'). A model is placed on the devices
picked with --devices, in that order. The CPU (local-host) comes first: it
runs the host code and is the default (lead) device; the accelerators after it
take contraction/conv dispatches round-robin in the listed order.
Every OPTIONS group holds either fixed `flags` (always passed) or user options,
each a map of choice -> flags. To add, remove or change a device or an option,
edit this file only. A flag that only works around a current gap is marked with
a `# workaround:` comment; remove it once the gap is fixed.
"""

import os

PEANO_INSTALL_DIR = os.environ.get("PEANO_INSTALL_DIR", "/workspace/llvm-aie")

# Device names are <driver>-<target>, so another NPU or GPU gets its own entry.
DEVICES = {
    "amdxdna-npu4": {
        "compile": [
            "--iree-hal-target-device={symbol}=amdxdna",
            "--iree-amdaie-target-device=npu4",
            f"--iree-amd-aie-peano-install-dir={PEANO_INSTALL_DIR}",
        ],
        "run": [
            "--device=amdxdna",
        ],
    },
    "vulkan-gfx1150": {
        "compile": [
            "--iree-hal-target-device={symbol}=vulkan",
            # Required: without it iree-compile silently targets a generic
            # baseline instead of gfx1150.
            "--iree-vulkan-target=gfx1150",
        ],
        "run": [
            "--device=vulkan",
        ],
    },
    "local-host": {
        "compile": [
            "--iree-hal-target-device={symbol}=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
        ],
        "run": [
            "--device=local-task",
        ],
    },
}

# Devices of a model by default, and of its accuracy reference (which uses the
# same options otherwise, with numeric f32).
DEFAULT_DEVICES = ["local-host", "amdxdna-npu4"]
REFERENCE_DEVICES = ["local-host"]

OPTIONS = {
    # Model-level transforms before dispatch formation.
    "graph": {
        "flags": [
            "--iree-preprocessing-pass-pipeline="
            "builtin.module(iree-preprocessing-convert-conv-to-channels-last)",
            "--iree-global-opt-use-im2col-for-convs",
            # workaround: contraction dispatch needs a plain linalg.fill init.
            "--iree-global-opt-detach-elementwise-through-reshape",
        ],
    },
    # How ops are grouped into dispatches.
    "dispatch": {
        "flags": [
            # workaround: amd-aie cannot tile a contraction with fused elementwise.
            "--iree-dispatch-creation-no-fuse-into-contraction-conv-roots",
            # workaround: address_patch cannot carry a runtime weight offset.
            "--iree-flow-enable-executable-deduplication=false",
        ],
    },
    # User choice: ukernel and vectorization.
    "vector": {
        "ukernel": {
            "none": [
                "--iree-amdaie-enable-ukernels=none",
            ],
            "all": [
                "--iree-amdaie-enable-ukernels=all",
            ],
        },
        "vectorize": {
            "off": [
                "--iree-amdaie-enable-vectorization-passes=false",
            ],
            "on": [
                "--iree-amdaie-enable-vectorization-passes=true",
            ],
        },
    },
    # User choice: what f32 contraction inputs are converted to.
    "numeric": {
        "numeric": {
            "f32": [],
            "bf16": [
                "--iree-amdaie-demote-contraction-inputs-to-bf16",
            ],
        },
    },
}

# Default choice of each user option.
DEFAULTS = {
    "ukernel": "all",
    "vectorize": "off",
    "numeric": "f32",
}


def user_options():
    """Returns {option name: [choices]} over all groups."""
    return {name: list(choices)
            for group in OPTIONS.values()
            for name, choices in group.items() if name != "flags"}


def check(selected, devices):
    """Returns an error message for an unsupported combination, else None."""
    if devices[0] != "local-host":
        return ("local-host must be the first device: the CPU runs the host "
                "code and is the lead device")
    # workaround: bf16 contractions fail to compile for the GPU; use f32.
    if selected.get("numeric") == "bf16" and any(d.startswith("vulkan-") for d in devices):
        return "--numeric bf16 is not supported with a vulkan device; use f32"
    return None


def device_symbol(device):
    return device.replace("-", "_")


def compile_flags(selected, devices):
    """Returns the iree-compile flags for `selected` ({option name: choice})
    on `devices` (names in DEVICES; the first is the default device)."""
    flags = []
    for device in devices:
        flags += [f.format(symbol=device_symbol(device))
                  for f in DEVICES[device]["compile"]]
    flags.append(f"--iree-hal-default-device={device_symbol(devices[0])}")
    for group in OPTIONS.values():
        for name, value in group.items():
            flags += value if name == "flags" else value[selected[name]]
    return flags


def run_flags(devices):
    """Returns the iree-run-module flags that open `devices`, in order."""
    return [f for device in devices for f in DEVICES[device]["run"]]
