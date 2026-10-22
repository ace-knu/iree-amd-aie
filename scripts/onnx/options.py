"""iree-compile options for running an ONNX model on amd-aie, grouped by function.

The two device groups hold `compile` flags (iree-compile) and `run` flags
(iree-run-module); exactly one of them is used per compile. Every other group
holds either fixed `flags` (always passed) or user options, each a map of
choice -> flags. To add, remove or change an option, edit this file only. A
flag that only works around a current gap is marked with a `# workaround:`
comment; remove it once the gap is fixed.
"""

import os

PEANO_INSTALL_DIR = os.environ.get("PEANO_INSTALL_DIR", "/workspace/llvm-aie")

DEVICE_GROUPS = ("device", "reference_device")

OPTIONS = {
    # Devices the model is placed on: the iree-compile flags that declare them
    # and the iree-run-module flags that open them. Keep the two lists in sync.
    "device": {
        "compile": [
            "--iree-hal-target-device=npu=amdxdna",
            "--iree-hal-target-device=cpu=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-hal-default-device=npu",
        ],
        "run": [
            "--device=amdxdna",
            "--device=local-task",
        ],
    },
    # Device the accuracy reference runs on, in place of "device". The
    # reference uses the same options otherwise, with numeric f32.
    "reference_device": {
        "compile": [
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
        ],
        "run": [
            "--device=local-task",
        ],
    },
    # NPU generation and core compiler.
    "backend": {
        "flags": [
            "--iree-amdaie-target-device=npu4",
            f"--iree-amd-aie-peano-install-dir={PEANO_INSTALL_DIR}",
        ],
    },
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
            for group_name, group in OPTIONS.items() if group_name not in DEVICE_GROUPS
            for name, choices in group.items() if name != "flags"}


def compile_flags(selected, device="device"):
    """Returns the iree-compile flags for `selected` ({option name: choice})
    on the device group `device`, in group order."""
    flags = []
    for group_name, group in OPTIONS.items():
        if group_name in DEVICE_GROUPS:
            flags += group["compile"] if group_name == device else []
            continue
        for name, value in group.items():
            flags += value if name == "flags" else value[selected[name]]
    return flags


def run_flags(device="device"):
    """Returns the iree-run-module flags of the device group `device`."""
    return OPTIONS[device]["run"]
