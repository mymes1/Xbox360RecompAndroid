/**
 * LZX decompressor implementation (see include/x360/lzx.h).
 *
 * Structure follows the public LZX format description:
 *   block := (verbatim:1) (blockSize:16)
 *            | (aligned:1) (blockSize:16) (uncompressed:1)
 *   verbatim/aligned block: pre-tree -> main tree delta -> length tree delta
 *                           [-> aligned tree] -> ELEM stream
 *   ELEM := literal | (mainSymbol:>=9) (lengthSymbol:3) (extra bits)
 *           position := slotBase + (aligned-tree | raw extra bits)
 */
#include "x360/lzx.h"
#include "../../include/x360/log.h"

#include <cstring>

#define TAG "X360:LZX"

namespace x360 {
namespace lzx {
namespace {

constexpr int kMaxBits = 17;
constexpr int kNumLengthTrees = 8;
constexpr int kNumAlignedTreeBits = 7;
constexpr int kPreTreeSymbols = 20;
constexpr int kPreTreeMax = 8;

// ─── Big-endian bit reader with 16-bit refill (the LZX convention) ─────────────
struct BitReader {
  const uint8_t* src = nullptr;
  size_t len = 0;
  size_t pos = 0;
  uint32_t bitbuf = 0;
  int bits_left = 0;

  void init(const uint8_t* s, size_t l) {
    src = s;
    len = l;
    pos = 0;
    bitbuf = 0;
    bits_left = 0;
  }
  bool refill() {
    while (bits_left <= 16) {
      if (pos + 1 < len) {
        const uint32_t w = (uint32_t(src[pos]) << 8) | src[pos + 1];
        pos += 2;
        bitbuf |= w << (16 - bits_left);
        bits_left += 16;
      } else if (pos < len) {  // trailing odd byte
        bitbuf |= uint32_t(src[pos]) << (24 - bits_left);
        pos++;
        bits_left += 8;
      } else {
        return bits_left > 0;
      }
    }
    return true;
  }
  uint32_t peek(int n) {
    if (bits_left < n) refill();
    if (n >= 32) return bitbuf;
    return bitbuf >> (32 - n);
  }
  void consume(int n) {
    bitbuf <<= n;
    bits_left -= n;
    if (bits_left < 0) bits_left = 0;
  }
  uint32_t read(int n) {
    const uint32_t v = peek(n) & ((n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1));
    consume(n);
    return v;
  }
  void align_to_16() {
    const int drop = bits_left & 15;
    if (drop) consume(drop);
  }
};

// ─── Huffman table ─────────────────────────────────────────────────────────────
// Direct lookup over the full kMaxBits window, storing the code length per entry
// so decoding reports exactly how many bits it consumed.
struct Table {
  uint16_t symbol[1 << kMaxBits];
  uint8_t length[1 << kMaxBits];
  int max_len = 0;
  bool valid = false;
};

bool build_table(Table& t, const int* lengths, int numSymbols) {
  int counts[kMaxBits + 1] = {};
  for (int i = 0; i < numSymbols; i++) {
    if (lengths[i] < 0 || lengths[i] > kMaxBits) return false;
    counts[lengths[i]]++;
  }
  counts[0] = 0;
  uint32_t code = 0;
  uint32_t next[kMaxBits + 1] = {};
  for (int i = 1; i <= kMaxBits; i++) {
    next[i] = (code + counts[i - 1]) << 1;
    code = next[i];
  }
  t.max_len = 0;
  for (int i = 1; i <= kMaxBits; i++) {
    if (counts[i]) t.max_len = i;
  }
  if (t.max_len == 0) return false;
  for (int s = 0; s < (1 << kMaxBits); s++) {
    t.symbol[s] = 0xFFFF;
    t.length[s] = 0;
  }
  for (int sym = 0; sym < numSymbols; sym++) {
    const int len = lengths[sym];
    if (len == 0) continue;
    const uint32_t c = next[len]++;
    const int fill = kMaxBits - len;
    const uint32_t base = c << fill;
    for (uint32_t j = 0; j < (1u << fill); j++) {
      t.symbol[base + j] = uint16_t(sym);
      t.length[base + j] = uint8_t(len);
    }
  }
  t.valid = true;
  return true;
}

bool decode(const Table& t, BitReader& br, int* outSymbol) {
  if (!t.valid) return false;
  const uint32_t v = br.peek(kMaxBits);
  const int len = t.length[v];
  if (len == 0) return false;
  br.consume(len);
  *outSymbol = t.symbol[v];
  return true;
}

// ─── Position slots ────────────────────────────────────────────────────────────
uint32_t slot_count_for(int windowBits) {
  switch (windowBits) {
    case 15: return 30;
    case 16: return 32;
    case 17: return 34;
    case 18: return 36;
    case 19: return 38;
    case 20: return 42;
    default: return 50;
  }
}

// LZX position-slot table: slots 0..7 hold "main tree * 2 + extra-bits", slots
// 8+ use the big-window form. Returned base excludes the extra bits.
struct SlotInfo {
  uint32_t base;
  int extra;
};

SlotInfo slot_for(uint32_t slot) {
  if (slot == 0) return {2, 0};
  if (slot == 1) return {3, 0};
  if (slot < 8) {
    const uint32_t main = (slot - 2) >> 1;
    const uint32_t extra = (slot - 2) & 1;
    return {(1u << main) + 2, int(extra)};
  }
  const uint32_t group = (slot - 8) >> 3;
  const uint32_t idx = (slot - 8) & 7;
  return {((1u << (20 + group)) + (idx << (16 + group))) - 2, int(16 + group)};
}

constexpr int kLengthBase[8] = {2, 3, 4, 6, 8, 12, 16, 24};

struct State {
  int window_bits = 17;
  int main_lengths[256 + 50 * 8] = {};
  int len_lengths[kNumLengthTrees * 8] = {};
  int aligned_lengths[1 << kNumAlignedTreeBits] = {};
  Table main_table;
  Table len_table;
  Table aligned_table;
  bool have_main = false;
  bool have_len = false;
};

// Pre-tree: read `bits`-wide unary-coded deltas for lengths[first..last].
void read_pre_lengths(BitReader& br, int* lengths, int first, int last) {
  for (int i = first; i <= last; i++) {
    int len = 0;
    while (len < kPreTreeMax && br.read(1)) len++;
    lengths[i] = len;
  }
}

int read_delta_main(BitReader& br, const Table& pre) {
  int sym = -1;
  if (!decode(pre, br, &sym)) return 0;
  switch (sym) {
    case 0: return -1;
    case 1: return int(br.read(4)) + 1;
    case 2: return -(int(br.read(4)) + 1);
    case 3: return 16;
    case 4: return -16;
    default: return 0;
  }
}

}  // namespace

bool decompress(const uint8_t* src, size_t srcLen, uint8_t* dst, size_t dstCap,
                int windowBits, size_t* outWritten) {
  if (!src || !dst || srcLen == 0 || dstCap == 0) return false;
  if (windowBits < 15) windowBits = 15;
  if (windowBits > 21) windowBits = 21;

  State st;
  st.window_bits = windowBits;
  const uint32_t numSlots = slot_count_for(windowBits);
  const int numMain = int(256 + numSlots * 8);

  BitReader br;
  br.init(src, srcLen);

  size_t out = 0;
  int blocks = 0;

  while (out < dstCap) {
    const uint32_t verbatim = br.read(1);
    uint32_t remaining = br.read(16);
    if (remaining == 0) break;
    if (++blocks > 100000) {
      X360LOGE(TAG, "too many LZX blocks — refusing to spin");
      return false;
    }

    if (!verbatim) {
      const uint32_t uncompressed = br.read(1);
      if (uncompressed) {
        br.align_to_16();
        if (remaining > dstCap - out) {
          X360LOGE(TAG, "uncompressed LZX block (%u) exceeds destination (%zu)", remaining,
                    dstCap - out);
          return false;
        }
        const size_t bytePos = br.pos - (size_t(br.bits_left) / 8);
        if (bytePos + remaining > srcLen) {
          X360LOGE(TAG, "uncompressed LZX block past end of input");
          return false;
        }
        memcpy(dst + out, src + bytePos, remaining);
        br.pos = bytePos + remaining;
        br.bitbuf = 0;
        br.bits_left = 0;
        out += remaining;
        continue;
      }
    }

    // ── Pre-tree ──────────────────────────────────────────────────────────────
    int pre_lengths[kPreTreeSymbols] = {};
    read_pre_lengths(br, pre_lengths, 0, kPreTreeSymbols - 1);
    Table pre;
    if (!build_table(pre, pre_lengths, kPreTreeSymbols)) {
      X360LOGE(TAG, "corrupt LZX pre-tree");
      return false;
    }

    // ── Main tree ─────────────────────────────────────────────────────────────
    const int main_delta = st.have_main ? read_delta_main(br, pre) : 0;
    for (int i = 0; i < 256; i++) {
      int sym = 0;
      if (!decode(pre, br, &sym)) {
        X360LOGE(TAG, "corrupt LZX main tree (pre-decode at %d)", i);
        return false;
      }
      int l = (i < 256 && st.have_main) ? st.main_lengths[i] + main_delta - sym : 8 - sym;
      st.main_lengths[i] = l < 0 ? 0 : (l > kMaxBits - 1 ? kMaxBits - 1 : l);
    }
    for (int i = 256; i < numMain; i++) {
      int sym = 0;
      if (!decode(pre, br, &sym)) {
        X360LOGE(TAG, "corrupt LZX main tree (position slots)");
        return false;
      }
      const int l = st.main_lengths[i] + main_delta - sym;
      st.main_lengths[i] = l < 0 ? 0 : (l > kMaxBits - 1 ? kMaxBits - 1 : l);
    }
    st.have_main = true;
    if (!build_table(st.main_table, st.main_lengths, numMain)) {
      X360LOGE(TAG, "empty LZX main tree");
      return false;
    }

    // ── Length tree ───────────────────────────────────────────────────────────
    const int len_delta = st.have_len ? read_delta_main(br, pre) : 0;
    for (int i = 0; i < kNumLengthTrees * 8; i++) {
      int sym = 0;
      if (!decode(pre, br, &sym)) {
        X360LOGE(TAG, "corrupt LZX length tree");
        return false;
      }
      const int l = st.len_lengths[i] + len_delta - sym;
      st.len_lengths[i] = l < 0 ? 0 : (l > 7 ? 7 : l);
    }
    st.have_len = true;
    if (!build_table(st.len_table, st.len_lengths, kNumLengthTrees * 8)) {
      X360LOGE(TAG, "empty LZX length tree");
      return false;
    }

    // ── Aligned tree (verbatim blocks reuse the previous one) ─────────────────
    if (!verbatim) {
      for (int i = 0; i < (1 << kNumAlignedTreeBits); i++) {
        int len = 0;
        while (len < 7 && br.read(1)) len++;
        st.aligned_lengths[i] = len;
      }
      if (!build_table(st.aligned_table, st.aligned_lengths, 1 << kNumAlignedTreeBits)) {
        X360LOGE(TAG, "empty LZX aligned tree");
        return false;
      }
    }

    // ── ELEM loop ─────────────────────────────────────────────────────────────
    int64_t left = int64_t(remaining);
    while (left > 0) {
      int mainSym = -1;
      if (!decode(st.main_table, br, &mainSym)) {
        X360LOGE(TAG, "corrupt main symbol at output offset %zu", out);
        return false;
      }
      if (mainSym < 0 || mainSym >= numMain) {
        X360LOGE(TAG, "main symbol out of range (%d)", mainSym);
        return false;
      }
      if (mainSym < 256) {
        dst[out++] = uint8_t(mainSym);
        left -= 1;
        continue;
      }

      const int verbatimFlag = (mainSym - 256) / 8;
      const uint32_t highSlot = uint32_t((mainSym - 256) % 8);

      int lengthSymbol = -1;
      if (!decode(st.len_table, br, &lengthSymbol) || lengthSymbol < 0 || lengthSymbol > 7) {
        X360LOGE(TAG, "corrupt length symbol");
        return false;
      }
      int length;
      if (lengthSymbol == 7) {
        length = 15 + int(br.read(5));
      } else {
        length = kLengthBase[verbatimFlag] + lengthSymbol;
      }

      const SlotInfo info = slot_for(highSlot);
      uint32_t position = info.base;
      if (info.extra) {
        if (verbatimFlag == 7 && !verbatim) {
          int sym = -1;
          if (!decode(st.aligned_table, br, &sym) || sym < 0) {
            X360LOGE(TAG, "corrupt aligned symbol");
            return false;
          }
          const uint32_t high = uint32_t(sym) >> 3;
          const uint32_t low = uint32_t(sym) & 7;
          position += (high << kNumAlignedTreeBits) | br.read(int(low));
        } else {
          position += br.read(info.extra);
        }
      }

      if (position == 0 || position > out) {
        X360LOGE(TAG, "invalid match distance %u (output %zu)", position, out);
        return false;
      }
      if (uint64_t(out) + uint64_t(length) > dstCap) {
        X360LOGE(TAG, "match of %d bytes overruns destination", length);
        return false;
      }
      const uint8_t* from = dst + (out - position);
      for (int i = 0; i < length; i++) dst[out + i] = from[i];
      out += size_t(length);
      left -= length;
    }
    if (left < 0) {
      X360LOGE(TAG, "block decoded past its declared size");
      return false;
    }
  }

  if (outWritten) *outWritten = out;
  X360LOGD(TAG, "decoded %d blocks, %zu bytes", blocks, out);
  return true;
}

}  // namespace lzx
}  // namespace x360

