
// RUN: iree-opt --amdaie-assign-buffer-addresses="alloc-scheme=sequential" --verify-diagnostics --split-input-file %s | FileCheck %s

// Core buffers start on the 32-byte vector load/store bus alignment.
// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           aie.buffer(%[[TILE_3_3]]) {address = 1024 : i32, sym_name = "a"} : memref<16xi8>
// CHECK:           aie.buffer(%[[TILE_3_3]]) {address = 1056 : i32, sym_name = "b"} : memref<512xi32>
// CHECK:           aie.buffer(%[[TILE_3_3]]) {address = 3104 : i32, sym_name = "c"} : memref<16xi16>
// CHECK:           %[[TILE_4_4:.*]] = aie.tile(4, 4)
// CHECK:           aie.buffer(%[[TILE_4_4]]) {address = 1024 : i32, sym_name = "_anonymous0"} : memref<500xi32>
// CHECK:           aie.core(%[[TILE_3_3]]) {
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.core(%[[TILE_4_4]]) {
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @tile_test {
 aie.device(xcvc1902) {
  %0 = aie.tile(3, 3)
  %b1 = aie.buffer(%0) { sym_name = "a" } : memref<16xi8>
  %1 = aie.buffer(%0) { sym_name = "b" } : memref<512xi32>
  %b2 = aie.buffer(%0) { sym_name = "c" } : memref<16xi16>
  %3 = aie.tile(4, 4)
  %4 = aie.buffer(%3) : memref<500xi32>
  aie.core(%0) {
    aie.end
  }
  aie.core(%3) {
    aie.end
  }
 }
}

// -----

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           aie.buffer(%[[TILE_3_1]]) {address = 0 : i32, sym_name = "a"} : memref<65536xi32>
// CHECK:           aie.memtile_dma(%[[TILE_3_1]]) {
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @memtile_test {
 aie.device(xcve2302) {
  %0 = aie.tile(3, 1)
  %b1 = aie.buffer(%0) { sym_name = "a" } : memref<65536xi32>
  aie.memtile_dma(%0) {
    aie.end
  }
 }
}

// -----

// On AIE2P a core buffer that can hold a full-width (64-byte) vector starts on
// a 64-byte boundary, a smaller one on the 32-byte bus alignment. Without it,
// the 32x f32 buffer after a 172x f32 (688-byte) one would sit at a 16-byte
// boundary, where a vector store writes the wrong address.
// CHECK-LABEL:   aie.device(npu4) {
// CHECK:           aie.buffer(%{{.*}}) {address = 1024 : i32, sym_name = "a"} : memref<172xf32>
// CHECK:           aie.buffer(%{{.*}}) {address = 1728 : i32, sym_name = "c"} : memref<32xf32>
// CHECK:           aie.buffer(%{{.*}}) {address = 1856 : i32, sym_name = "d"} : memref<4xf32>
// CHECK:           aie.buffer(%{{.*}}) {address = 1920 : i32, sym_name = "e"} : memref<16xf32>
module @npu4_align_test {
 aie.device(npu4) {
  %0 = aie.tile(0, 2)
  %a = aie.buffer(%0) { sym_name = "a" } : memref<172xf32>
  %c = aie.buffer(%0) { sym_name = "c" } : memref<32xf32>
  %d = aie.buffer(%0) { sym_name = "d" } : memref<4xf32>
  %e = aie.buffer(%0) { sym_name = "e" } : memref<16xf32>
  aie.core(%0) {
    aie.end
  }
 }
}

// -----

module @error_test {
  aie.device(xcve2302) {
    // expected-error @+1 {{'aie.tile' op allocated buffers exceeded available memory (528000>524288)}}
    %0 = aie.tile(3, 1)
    %b1 = aie.buffer(%0) { sym_name = "a" } : memref<132000xi32>
    aie.memtile_dma(%0) {
      aie.end
    }
  }
}
