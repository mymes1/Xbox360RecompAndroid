/**
 * LZX decompressor for XEX2 "normal" compression.
 *
 * The XEX2 payload after de-blocking is a raw LZX stream (no CHM call-frame
 * header). Only verbatim and aligned blocks are emitted by the Xbox 360 packer;
 * uncompressed blocks are supported too because they appear in small XEXs.
 *
 * Format knowledge: Microsoft's public LZX description (as used by MS-PAC / CHM
 * and documented by libmspack and Xenia's port of it) — reimplemented here, not
 * copied. This file replaces the previous "copy the remaining bytes" stub, which
 * produced a corrupted image and made every later stage look like a JIT bug.
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace x360 {
namespace lzx {

/**
 * Decompress `srcLen` bytes into `dst` (capacity `dstCap`).
 *
 * @param windowBits  LZX window size as a power of two (XEX: 17..21, 17 default).
 * @param outWritten  number of bytes produced (never more than dstCap).
 * @return true if the stream was fully consumed and the output length is exact.
 */
bool decompress(const uint8_t* src, size_t srcLen, uint8_t* dst, size_t dstCap,
                int windowBits, size_t* outWritten);

}  // namespace lzx
}  // namespace x360
