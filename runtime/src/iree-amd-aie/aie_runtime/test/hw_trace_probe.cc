// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Standalone dev tool (not a lit-checked unit test): drives aie-rt's
// high-level trace API (xaie_trace.h) against REAL npu4 hardware, not the
// CDO/simulation backends the rest of this directory's tests use.
//
// aie-rt's own IO backends (xaie_io.c) are all compile-time-selected via
// macros (__AIECDO__, __AIEDEBUG__, ...), and this repo's `xaiengine` target
// only compiles in the CDO backend (see cmake/iree_aie_rt.cmake) -- there is
// no "real hardware" backend built in. Rather than reconfiguring that shared
// build target (used by the compiler for CDO/PDI generation) or hand-deriving
// register field layouts a second time, this tool installs its own
// XAie_Backend directly on a throwaway XAie_DevInst (XAie_DevInst::Backend
// and ::IOInst are plain public struct fields, so this is a supported
// override, not a hack around opaque state) that forwards Write32/Read32/
// MaskWrite32 to the amdxdna KMQ shim's already-implemented, already-working,
// previously-uncalled read_aie_reg/write_aie_reg_checked ioctls
// (driver/amdxdna/shim/linux/kmq/device.cpp). This reuses aie-rt's own
// (correctness-critical) register tables/field-packing untouched -- only the
// bottom-most Read/Write primitive is replaced.
//
// See docs/2026-08-27_matmul_bias_fusion_hang_root_cause_refined.md sec 23
// for why (scoping the matmul+bias fusion hang's hardware-trace investigation)
// and the on-chip-vs-DDR follow-up in memory: trace output is a routable
// stream-switch port (StrmSwPortType::TRACE), not a fixed on-chip buffer, so
// it can be sent to DDR via an ordinary shim DMA -- this tool is the first
// step (validate real register read/write plumbing) before wiring that route.

#include <cstdio>
#include <memory>

#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "iree-amd-aie/driver/amdxdna/shim/linux/kmq/device.h"

// xaie_trace.h's XAie_Trace* API comes in transitively via
// iree_aie_runtime.h's own "xaiengine.h" include (already extern-"C"-wrapped
// there).

using namespace mlir::iree_compiler::AMDAIE;

namespace {

// Context stashed in XAie_DevInst::IOInst for our backend's callbacks.
struct AmdxdnaIoCtx {
  shim_xdna::device *device;
  uint32_t colShift;
  uint32_t rowShift;
};

// npu4 (AIE2P/Strix B0) tile address encoding, confirmed against
// third_party/aie-rt/driver/src/lite/xaie_lite_hwcfg.h's
// `XAIE_DEV_GEN_AIE2P_STRIX_B0` block (ColShift=25, RowShift=20) and
// mirrored by AMDAIEDeviceModel::getColumn/RowFromAddress. Duplicated here
// (rather than called through AMDAIEDeviceModel) so this backend has no
// dependency on which XAie_DevInst it happens to be installed on.
void DecodeAddr(const AmdxdnaIoCtx *ctx, uint64_t addr, uint16_t *col,
                uint16_t *row, uint32_t *offset) {
  uint32_t colRowFieldMask = (1u << (ctx->colShift - ctx->rowShift)) - 1;
  *col = static_cast<uint16_t>((addr >> ctx->colShift) & colRowFieldMask);
  *row = static_cast<uint16_t>((addr >> ctx->rowShift) & colRowFieldMask);
  *offset = static_cast<uint32_t>(addr & ((1u << ctx->rowShift) - 1));
}

AieRC AmdxdnaInit(XAie_DevInst *DevInst) {
  // IOInst is set by the caller before installing this backend (see main());
  // nothing left to do here.
  (void)DevInst;
  return XAIE_OK;
}

AieRC AmdxdnaFinish(void *IOInst) {
  (void)IOInst;
  return XAIE_OK;
}

AieRC AmdxdnaRead32(void *IOInst, uint64_t RegOff, uint32_t *Data) {
  auto *ctx = static_cast<AmdxdnaIoCtx *>(IOInst);
  uint16_t col, row;
  uint32_t offset;
  DecodeAddr(ctx, RegOff, &col, &row, &offset);
  int err = ctx->device->read_aie_reg(col, row, offset, Data);
  if (err) {
    fprintf(stderr,
            "[hw_trace_probe] read_aie_reg(col=%u,row=%u,off=0x%x) failed: "
            "errno=%d\n",
            col, row, offset, err);
    return XAIE_ERR;
  }
  return XAIE_OK;
}

AieRC AmdxdnaWrite32(void *IOInst, uint64_t RegOff, uint32_t Value) {
  auto *ctx = static_cast<AmdxdnaIoCtx *>(IOInst);
  uint16_t col, row;
  uint32_t offset;
  DecodeAddr(ctx, RegOff, &col, &row, &offset);
  int err = ctx->device->write_aie_reg_checked(col, row, offset, Value);
  if (err) {
    fprintf(stderr,
            "[hw_trace_probe] write_aie_reg(col=%u,row=%u,off=0x%x) failed: "
            "errno=%d\n",
            col, row, offset, err);
    return XAIE_ERR;
  }
  return XAIE_OK;
}

// NOTE: not a single atomic hardware read-modify-write -- amdxdna's UAPI
// (amdxdna_accel.h) has no masked-write ioctl, only plain register
// read/write. Fine for this probing tool (nothing else is touching these
// registers concurrently); would need revisiting for the real trace-config
// tool if it ever writes a register something else on-chip also mutates
// concurrently.
AieRC AmdxdnaMaskWrite32(void *IOInst, uint64_t RegOff, uint32_t Mask,
                         uint32_t Value) {
  uint32_t cur;
  AieRC rc = AmdxdnaRead32(IOInst, RegOff, &cur);
  if (rc != XAIE_OK) return rc;
  uint32_t newVal = (cur & ~Mask) | (Value & Mask);
  return AmdxdnaWrite32(IOInst, RegOff, newVal);
}

const XAie_Backend kAmdxdnaBackend = {
    /*Type=*/XAIE_IO_BACKEND_MAX,  // unused sentinel; we never go through
                                   // IOBackend[] / XAie_SetIOBackend for this.
    /*Ops=*/
    {
        /*Init=*/AmdxdnaInit,
        /*Finish=*/AmdxdnaFinish,
        /*Write32=*/AmdxdnaWrite32,
        /*Read32=*/AmdxdnaRead32,
        /*MaskWrite32=*/AmdxdnaMaskWrite32,
        /*MaskPoll=*/nullptr,
        /*BlockWrite32=*/nullptr,
        /*BlockSet32=*/nullptr,
        /*CmdWrite=*/nullptr,
        /*RunOp=*/nullptr,
        /*AddressPatching=*/nullptr,
        /*MemAllocate=*/nullptr,
        /*MemFree=*/nullptr,
        /*MemSyncForCPU=*/nullptr,
        /*MemSyncForDev=*/nullptr,
        /*MemAttach=*/nullptr,
        /*MemDetach=*/nullptr,
        /*GetTid=*/nullptr,
        /*SubmitTxn=*/nullptr,
        /*GetShimDmaBdConfig=*/nullptr,
        /*GetAttr=*/nullptr,
        /*SetAttr=*/nullptr,
    },
};

}  // namespace

int main() {
  AMDAIEDeviceModel model = getDeviceModel(AMDAIEDevice::npu4);

  std::unique_ptr<shim_xdna::device> device;
  int err = shim_xdna::device::create(model.devInst.NumRows,
                                       model.devInst.NumCols,
                                       "/dev/accel/accel0", &device);
  if (err) {
    fprintf(stderr, "[hw_trace_probe] failed to open /dev/accel/accel0: "
                     "errno=%d\n", err);
    return 1;
  }

  static AmdxdnaIoCtx ioCtx{device.get(), model.getColumnShift(),
                            model.getRowShift()};
  model.devInst.IOInst = &ioCtx;
  model.devInst.Backend = &kAmdxdnaBackend;

  // Safe (read-only) sanity check: read the shim tile's trace status
  // register at (col=0, row=0). No side effects -- this only validates that
  // our decode-and-ioctl path actually talks to real silicon correctly
  // before anything in this tool ever writes to live hardware state.
  XAie_LocType shimLoc = XAie_TileLoc(0, 0);
  XAie_TraceState state;
  AieRC rc = XAie_TraceGetState(&model.devInst, shimLoc, XAIE_PL_MOD, &state);
  if (rc != XAIE_OK) {
    fprintf(stderr, "[hw_trace_probe] XAie_TraceGetState failed: rc=%d\n",
            rc);
    return 1;
  }

  printf("[hw_trace_probe] shim(0,0) trace state = %d (0=IDLE,1=RUNNING,"
         "2=OVERFLOW expected)\n",
         static_cast<int>(state));
  return 0;
}
