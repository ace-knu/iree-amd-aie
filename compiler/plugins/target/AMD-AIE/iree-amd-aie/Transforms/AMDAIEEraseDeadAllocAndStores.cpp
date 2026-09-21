// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"

#define DEBUG_TYPE "iree-amdaie-erase-dead-alloc-and-stores"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEEraseDeadAllocAndStoresPass
    : public impl::AMDAIEEraseDeadAllocAndStoresBase<
          AMDAIEEraseDeadAllocAndStoresPass> {
 public:
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    memref::eraseDeadAllocAndStores(rewriter, getOperation());
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEEraseDeadAllocAndStoresPass() {
  return std::make_unique<AMDAIEEraseDeadAllocAndStoresPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
