#!/bin/bash
#
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

this_dir="$(cd $(dirname $0) && pwd)"
RELEASE=$(cat $this_dir/peano_commit_linux.txt)

# Peano is mirrored as a GitHub Release asset on the ace-knu fork rather than
# fetched from the Xilinx nightly index: nightly assets are periodically pruned
# (the earlier v19 pin was removed that way). The pin is v22; it needs full
# unrolling disabled in opt (see makePeanoOptArgs in XCLBinGen.cpp). The v19
# mirror (tag peano-v19) is kept for rollback.
# See docs/2026-07-06_env_setup/DEV_CONTAINER.md.
URL="https://github.com/ace-knu/iree-amd-aie/releases/download/peano-v22/peano-v22-linux.tar.gz"

echo "Downloading Peano ${RELEASE} from ${URL}"
rm -rf "$PWD/llvm-aie" "$PWD"/llvm_aie-*.dist-info
curl -fSL "$URL" | tar xz -C "$PWD"
