/**
 * PowerPC Xenon interpreter — the authoritative implementation of guest ISA
 * semantics. Two jobs:
 *   1. execute everything the ARM64 backend declines to translate (VMX128,
 *      Xenon 64-bit vector ops, obscure integer forms), so a real title makes
 *      forward progress instead of hanging on "nop"s;
 *   2. act as the reference the JIT is checked against (tests/jit_parity_test).
 *
 * Endianness rules applied here:
 *   - guest memory is big-endian, so every load/store byte-swaps;
 *   - PPCContext fields are host-native (this struct lives in host memory);
 *   - PPC GPRs are 64-bit: 32-bit results are truncated then zero-extended,
 *     exactly what Xenon does (no sign extension for add/or/slw/...).
 */
#include "../../include/jit/jit_engine.h"
#include "../../include/jit/ppc_decode.h"
#include "../../include/loader/xex_loader.h"
#include "../../include/x360/diagnostics.h"
#include "../../include/x360/log.h"

#include <climits>
#include <cmath>
#include <cstring>

#define TAG "X360:CPU"

namespace x360 {
namespace jit {
namespace {

// ─── Guest memory access ──────────────────────────────────────────────────────
struct Mem {
  uint8_t* base = nullptr;
  uint64_t guestBase = 0;
  uint64_t size = 0;

  bool ok(uint64_t guest, uint64_t len) const {
    if (guest < guestBase) return false;
    const uint64_t off = guest - guestBase;
    return off + len >= off && off + len <= size;
  }
  bool load(uint64_t guest, uint32_t len, uint64_t* out) const {
    if (!ok(guest, len)) return false;
    uint64_t raw = 0;
    memcpy(&raw, base + (guest - guestBase), len);
    // `raw` holds the little-endian byte image of the guest word; the guest is
    // big-endian, so reassemble in BE order.
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
    uint64_t v = 0;
    for (uint32_t i = 0; i < len; i++) v = (v << 8) | b[len - 1 - i];
    *out = v;
    return true;
  }
  bool store(uint64_t guest, uint32_t len, uint64_t value) const {
    if (!ok(guest, len)) return false;
    uint64_t raw = 0;
    uint8_t* b = reinterpret_cast<uint8_t*>(&raw);
    for (uint32_t i = 0; i < len; i++) b[len - 1 - i] = uint8_t(value >> (8 * i));
    memcpy(base + (guest - guestBase), &raw, len);
    return true;
  }
};

inline Mem memOf(PPCContext& ctx) {
  Mem m;
  m.base = ctx.hostMem;
  m.guestBase = ctx.guestBase;
  m.size = ctx.guestSize;
  return m;
}

// ─── CR / XER helpers ─────────────────────────────────────────────────────────
enum : uint32_t { kXerSo = 1u << 29, kXerOv = 1u << 28, kXerCa = 1u };

inline void setCrField(PPCContext& ctx, int crf, uint32_t bits) {
  const int shift = 31 - (crf * 4) - 3;
  ctx.cr = (ctx.cr & ~(0xFu << shift)) | ((bits & 0xFu) << shift);
}
inline uint32_t getCrBit(PPCContext& ctx, int bi) { return (ctx.cr >> (31 - bi)) & 1u; }
inline void setXerFlag(PPCContext& ctx, uint32_t flag, bool on) {
  ctx.xer = on ? (ctx.xer | flag) : (ctx.xer & ~flag);
}
inline void applyCompare(PPCContext& ctx, int crf, int64_t diff, bool so) {
  uint32_t bits = 0;
  if (diff < 0) bits |= 8;
  else if (diff > 0) bits |= 4;
  else bits |= 2;
  if (so) bits |= 1;
  setCrField(ctx, crf, bits);
}

inline uint64_t gpr(const PPCContext& ctx, int r) { return r < 0 ? 0 : ctx.gpr[r & 31]; }
inline void setGpr(PPCContext& ctx, int r, uint64_t v) {
  if (r >= 0) ctx.gpr[r & 31] = v;
}
inline uint64_t trunc32(uint64_t v) { return v & 0xFFFFFFFFull; }

// 64x64 -> high 64 bits, without compiler extensions (the project builds with
// -Wpedantic, so __int128 is off the table). Hacker's Delight's muluhalf/MUL_HW.
inline uint64_t mulhiU64(uint64_t u, uint64_t v) {
  const uint64_t u0 = u & 0xFFFFFFFFull, u1 = u >> 32;
  const uint64_t v0 = v & 0xFFFFFFFFull, v1 = v >> 32;
  const uint64_t w0 = u0 * v0;
  const uint64_t t = u1 * v0 + (w0 >> 32);
  const uint64_t w1 = (t & 0xFFFFFFFFull) + u0 * v1;
  return u1 * v1 + (t >> 32) + (w1 >> 32);
}
inline int64_t mulhiS64(int64_t u, int64_t v) {
  uint64_t w = mulhiU64(uint64_t(u), uint64_t(v));
  if (u < 0) w -= uint64_t(v);
  if (v < 0) w -= uint64_t(u);
  return int64_t(w);
}
// x + y + carry-in, returning the carry out.
inline uint64_t addCarry(uint64_t x, uint64_t y, uint64_t cin, uint64_t* out) {
  const uint64_t s1 = x + y;
  const bool c1 = s1 < x;
  const uint64_t s2 = s1 + cin;
  const bool c2 = cin != 0 && s2 < s1;
  *out = s2;
  return (c1 || c2) ? 1u : 0u;
}

inline void updateRec(PPCContext& ctx, uint64_t result, bool so) {
  uint32_t bits = 0;
  const int64_t s = int64_t(result);
  if (s < 0) bits |= 8;
  else if (s > 0) bits |= 4;
  else bits |= 2;
  if (so) bits |= 1;
  setCrField(ctx, 0, bits);
}

// Effective address: rA==0 means "use the immediate only" for the *update-less*
// forms (per the ISA, and unlike plain arithmetic where r0 is a real register).
inline uint64_t eaOf(const PPCContext& ctx, const DecodedPpc& di) {
  // D-form (immediate) treats rA == 0 as "the constant 0"; X-form (indexed) uses
  // register 0 as a real register. Getting this backwards breaks prologues.
  const bool indexed = di.has(PpcFlag::kFlagIndex);
  const uint64_t base = (di.ra == 0 && !indexed) ? 0 : gpr(ctx, di.ra);
  const uint64_t off = di.has(PpcFlag::kFlagIndex) ? gpr(ctx, di.rb) : uint64_t(di.imm);
  return (base + off) & 0xFFFFFFFFull;
}

}  // namespace

// ─── One decoded instruction ──────────────────────────────────────────────────
static bool runDecoded(PPCContext& ctx, const DecodedPpc& di) {
  Mem mem = memOf(ctx);

  switch (di.cls) {
    case PpcClass::Invalid:
      return false;
    case PpcClass::Nop:
      ctx.pc = di.pc + 4;
      return true;
    case PpcClass::Dcbz: {
      // Zero the 128-byte line the address falls in (Xenon's cache-line size).
      const uint64_t ea = gpr(ctx, di.ra >= 0 ? di.ra : 0) & ~0x7Full;
      if (!mem.ok(ea, 128)) {
        ctx.exitCode = kExitError;
        ctx.haltReason = 9;
        return true;
      }
      memset(mem.base + (ea - mem.guestBase), 0, 128);
      ctx.pc = di.pc + 4;
      return true;
    }

    // ── Add / Subf / Subfc / Subfe ──
    // ── integer arithmetic ────────────────────────────────────────────────
    // Xenon computes these on the full 64-bit registers (Xenia does the same);
    // `size` carries the *operand* width for the 32-bit forms (mullw, divw, ...).
    case PpcClass::Add:
    case PpcClass::Subf: {
      // Every one of these is "sum of two 64-bit operands plus a carry input"; the
      // subtract family uses the ISA's ~rA + rB + cin identity, which is what makes
      // XER[CA] come out right. Carry out is computed exactly (128-bit add).
      const bool sub = di.cls == PpcClass::Subf;
      const int sel = int(di.imm);
      const uint64_t a64 = gpr(ctx, di.ra);
      const uint64_t b64 = di.has(PpcFlag::kFlagImm) ? uint64_t(di.imm) : gpr(ctx, di.rb);
      const uint64_t caIn = ctx.xer & kXerCa;
      const uint64_t x = sub ? ~a64 : a64;
      uint64_t y = b64, cin = 0;
      bool touchCa = false;
      switch (sel) {
        case 0: cin = sub ? 1u : 0u; break;                     // add / subf
        case 1: cin = caIn; touchCa = true; break;               // adde / subfe
        case 2: y = 0; cin = caIn; touchCa = true; break;        // addze / subfze
        case 3: touchCa = true; break;                           // addc / subfc
        case 4: y = ~0ull; cin = caIn; touchCa = true; break;    // addme / subfme
        default: return false;
      }
      uint64_t res = 0;
      const bool caOut = addCarry(x, y, cin, &res) != 0;
      setGpr(ctx, di.rd, res);
      if (touchCa) setXerFlag(ctx, kXerCa, caOut);
      if (di.has(PpcFlag::kFlagRc) || di.has(PpcFlag::kFlagOvf)) {
        const int64_t sa = int64_t(a64), sb = int64_t(b64), sr = int64_t(res);
        const bool ovf = di.has(PpcFlag::kFlagOvf) &&
                         (sub ? (((sa ^ sb) & (sb ^ sr)) < 0) : (((sa ^ sr) & (sb ^ sr)) < 0));
        if (di.has(PpcFlag::kFlagOvf)) {
          setXerFlag(ctx, kXerOv, ovf);
          setXerFlag(ctx, kXerSo, ovf ^ (touchCa ? caOut : (ctx.xer & kXerCa) != 0));
        }
        updateRec(ctx, res, (ctx.xer & kXerSo) != 0);
      }
      ctx.pc = di.pc + 4;
      return true;
    }

    case PpcClass::Neg: {
      const uint64_t a = gpr(ctx, di.ra);
      const uint64_t res = 0 - a;
      setGpr(ctx, di.rd, res);
      if (di.has(PpcFlag::kFlagRc)) {
        const bool ovf = di.has(PpcFlag::kFlagOvf) && a == (1ull << 63);
        if (di.has(PpcFlag::kFlagOvf)) setXerFlag(ctx, kXerOv, ovf);
        updateRec(ctx, res, ovf);
      }
      ctx.pc = di.pc + 4;
      return true;
    }

    case PpcClass::Mul: {
      const bool wide = di.mb >= 8;   // operand width: 4 = "lw/hw" forms, 8 = 64-bit
      const bool high = di.has(PpcFlag::kFlagHigh);
      const bool sgn = di.has(PpcFlag::kFlagSigned);
      const uint64_t a = gpr(ctx, di.ra);
      const uint64_t b = di.has(PpcFlag::kFlagImm) ? uint64_t(di.imm) : gpr(ctx, di.rb);
      uint64_t res = 0;
      if (wide) {
        if (high) {
          if (sgn) res = uint64_t(mulhiS64(int64_t(a), int64_t(b)));
          else res = mulhiU64(a, b);
        } else {
          res = a * b;
        }
      } else {
        const uint32_t ua = uint32_t(a), ub = uint32_t(b);
        if (high) {
          if (sgn) res = uint64_t(int64_t(int32_t(ua)) * int64_t(int32_t(ub)) >> 32);
          else res = (uint64_t(ua) * uint64_t(ub)) >> 32;
        } else {
          // mullw on Xenon: the 32-bit halves are sign-extended and multiplied in
          // 64 bits (this is what Xenia does, and it is observable).
          res = uint64_t(int64_t(int32_t(ua)) * int64_t(int32_t(ub)));
        }
      }
      setGpr(ctx, di.rd, res);
      if (di.has(PpcFlag::kFlagRc)) {
        const bool so = di.has(PpcFlag::kFlagOvf) && sgn && !high &&
                        (int64_t(res) != int64_t(int32_t(uint32_t(res))));
        if (di.has(PpcFlag::kFlagOvf)) setXerFlag(ctx, kXerOv, so);
        updateRec(ctx, res, so);
      }
      ctx.pc = di.pc + 4;
      return true;
    }

    case PpcClass::Div: {
      const bool wide = di.mb >= 8;
      const bool sgn = di.has(PpcFlag::kFlagSigned);
      const uint64_t a = gpr(ctx, di.ra), b = gpr(ctx, di.rb);
      uint64_t res = 0;
      bool ovf = false;
      if (wide) {
        if (b == 0) ovf = true;
        else if (sgn) {
          if (int64_t(a) == INT64_MIN && int64_t(b) == -1) ovf = true;
          else res = uint64_t(int64_t(a) / int64_t(b));
        } else {
          res = a / b;
        }
      } else {
        if (uint32_t(b) == 0) ovf = true;
        else if (sgn) {
          const int32_t x = int32_t(uint32_t(a)), y = int32_t(uint32_t(b));
          if (x == INT32_MIN && y == -1) ovf = true;
          else res = uint64_t(int64_t(x / y));
        } else {
          res = uint64_t(uint32_t(a) / uint32_t(b));
        }
      }
      setGpr(ctx, di.rd, res);
      if (di.has(PpcFlag::kFlagRc)) {
        if (di.has(PpcFlag::kFlagOvf)) setXerFlag(ctx, kXerOv, ovf);
        updateRec(ctx, res, ovf);
      }
      ctx.pc = di.pc + 4;
      return true;
    }

    case PpcClass::Logic:
    case PpcClass::LogicImm: {
      const uint64_t a = gpr(ctx, di.ra);
      const uint64_t b = (di.cls == PpcClass::LogicImm) ? uint64_t(di.imm) : gpr(ctx, di.rb);
      const int sel = int(di.cls == PpcClass::LogicImm ? di.rc : di.imm);
      uint64_t res;
      switch (sel) {
        case kLogicAnd: res = a & b; break;
        case kLogicAndc: res = a & ~b; break;
        case kLogicNand: res = ~(a & b); break;
        case kLogicNor: res = ~(a | b); break;
        case kLogicOr: res = a | b; break;
        case kLogicOrc: res = a | ~b; break;
        case kLogicXor: res = a ^ b; break;
        case kLogicEqv: res = ~(a ^ b); break;
        default: return false;
      }
      if (di.has(PpcFlag::kFlagZero32)) res = trunc32(res);  // andi./oris style
      setGpr(ctx, di.rd, res);
      if (di.has(PpcFlag::kFlagRc)) updateRec(ctx, res, (ctx.xer & kXerSo) != 0);
      ctx.pc = di.pc + 4;
      return true;
    }

    case PpcClass::Cntlz: {
      const uint64_t a = gpr(ctx, di.ra);
      const uint64_t res = di.mb >= 8 ? (a ? uint64_t(__builtin_clzll(a)) : 64)
                                      : (uint32_t(a) ? uint64_t(__builtin_clz(uint32_t(a))) : 32);
      setGpr(ctx, di.rd, res);
      if (di.has(PpcFlag::kFlagRc)) updateRec(ctx, res, false);
      ctx.pc = di.pc + 4;
      return true;
    }

    case PpcClass::Popcnt: {
      const uint64_t a = gpr(ctx, di.ra);
      const int bytes = di.mb ? di.mb : 8;
      uint64_t res = 0;
      for (int i = 0; i < bytes; i++) {
        const uint8_t byte = uint8_t(a >> (8 * (bytes - 1 - i)));
        res += uint64_t(__builtin_popcount(byte));
      }
      setGpr(ctx, di.rd, res);
      if (di.has(PpcFlag::kFlagRc)) updateRec(ctx, res, false);
      ctx.pc = di.pc + 4;
      return true;
    }

    case PpcClass::Extend: {
      const uint64_t a = gpr(ctx, di.ra);
      uint64_t res;
      switch (di.mb) {
        case 1: res = uint64_t(int64_t(int8_t(uint8_t(a)))); break;
        case 2: res = uint64_t(int64_t(int16_t(uint16_t(a)))); break;
        default: res = uint64_t(int64_t(int32_t(uint32_t(a)))); break;
      }
      setGpr(ctx, di.rd, res);
      if (di.has(PpcFlag::kFlagRc)) updateRec(ctx, res, false);
      ctx.pc = di.pc + 4;
      return true;
    }

    case PpcClass::Rl: {
      // Width comes from the decode (rc); mb/me are the mask (bit 0 = MSB).
      const int width = di.rc == 64 ? 64 : 32;
      const bool wide = width == 64;
      const uint64_t a = gpr(ctx, di.ra);
      int sh;
      bool shiftOver = false;
      if (di.rb >= 0) {
        const uint64_t shv = gpr(ctx, di.rb);
        // 32-bit forms zero the result when the count's bit 5 is set.
        shiftOver = wide ? (shv & ~0x3Full) != 0 : (shv & 0x20ull) != 0;
        sh = int(shv & 0x3Full);
      } else {
        sh = di.sh;
        shiftOver = wide ? (sh > 63) : (sh > 31);
      }
      int mb = di.mb, me = di.me;
      const int sel = int(di.imm);
      if (sel != kRlRotateMasked) {
        // slw/srw/sraw (and their 64-bit brothers): the mask is "everything".
        mb = wide ? 0 : (width - 32);
        me = wide ? 63 : width - 1;
        if (sel == kRlShiftLeft) { mb = 0; me = width - 1 - sh; }
        else { mb = sh; me = width - 1; }
      }
      const bool asr = sel == kRlShiftRightA;
      const uint64_t mask = ppc_mask(mb, me, width);
      uint64_t r;
      bool ca = false;
      const uint64_t sv = a & (wide ? ~0ull : 0xFFFFFFFFull);
      if (shiftOver && !asr) {
        r = 0;
      } else if (asr) {
        const int64_t x = wide ? int64_t(a) : int64_t(int32_t(uint32_t(a)));
        const int s = sh > width - 1 ? width - 1 : sh;
        const int64_t v = x >> s;
        r = uint64_t(v);
        ca = (uint64_t(v) & ~mask) != 0 && x < 0;
      } else {
        const uint64_t rot = wide ? rotl64(a, sh) : uint64_t(rotl32(uint32_t(sv), sh));
        r = rot & mask;
        // XER[CA] for the mask-forms: set when the mask excludes bit 0 of the
        // result and the rotated value's bit 0 (the value shifted out) is set.
        const bool maskHasBit0 = (mask & 1ull) != 0;
        ca = !maskHasBit0 && ((rot >> (width - 1)) & 1u) != 0;
      }
      if (di.has(PpcFlag::kFlagInsert)) r = r | (gpr(ctx, di.rd) & ~mask);
      if (!wide && sel != kRlRotateMasked) r = trunc32(r);
      setGpr(ctx, di.rd, r);
      if (di.has(PpcFlag::kFlagRc)) {
        setXerFlag(ctx, kXerCa, ca);
        updateRec(ctx, r, (ctx.xer & kXerSo) != 0);
      }
      ctx.pc = di.pc + 4;
      return true;
    }

    case PpcClass::Cmp: {
      // X-form cmp/cmpl: sel 0 = signed, 1 = unsigned; kFlagWide = 64-bit.
      // D-form cmpwi/cmplwi are always 32-bit and carry signedness in kFlagSigned.
      const uint64_t a = gpr(ctx, di.ra);
      const uint64_t b = di.has(PpcFlag::kFlagImm) ? uint64_t(di.imm) : gpr(ctx, di.rb);
      const bool isImm = di.has(PpcFlag::kFlagImm);
      const bool is64 = isImm ? di.has(PpcFlag::kFlagWide)
                              : (di.has(PpcFlag::kFlagWide) || int(di.imm) == kCmpD ||
                                 int(di.imm) == kCmplD);
      const bool uns = isImm ? !di.has(PpcFlag::kFlagSigned)
                             : (int(di.imm) == kCmplW || int(di.imm) == kCmplD);
      int64_t diff;
      bool so = false;
      if (is64) {
        if (uns) diff = a > b ? 1 : (a < b ? -1 : 0);
        else {
          const int64_t x = int64_t(a), y = int64_t(b);
          diff = x > y ? 1 : (x < y ? -1 : 0);
        }
      } else {
        const uint32_t x = uint32_t(a), y = uint32_t(b);
        if (uns) diff = x > y ? 1 : (x < y ? -1 : 0);
        else {
          const int32_t sx = int32_t(x), sy = int32_t(y);
          diff = sx > sy ? 1 : (sx < sy ? -1 : 0);
          // L=1 with the 32-bit compare is the Xenon "full" form: the high words
          // take part in the result, which is how `cmpwi` reports overflow.
          so = (sx == INT32_MIN && sy == INT32_MIN && (a >> 32) != (b >> 32)) ||
               (di.has(PpcFlag::kFlagRc) && diff == 0 && (a >> 32) != (b >> 32));
        }
      }
      applyCompare(ctx, di.crf, diff, so);
      ctx.pc = di.pc + 4;
      return true;
    }


    // ── memory ──
    case PpcClass::Ld:
    case PpcClass::St: {
      const uint32_t size = uint32_t(di.mb);
      if (size != 1 && size != 2 && size != 4 && size != 8) {
        ctx.haltReason = 1;
        return false;
      }
      const uint64_t ea = eaOf(ctx, di);
      const bool fp = di.has(PpcFlag::kFlagSingle);
      if (di.cls == PpcClass::Ld) {
        uint64_t v = 0;
        if (!mem.load(ea, size, &v)) {
          diag::report_unsupported(diag::Subsystem::Memory, "load outside guest window");
          ctx.exitCode = kExitError;
          ctx.haltReason = 2;
          ctx.pc = ea;  // keep the faulting address for the crash report
          return true;
        }
        if (di.has(PpcFlag::kFlagSwap)) {  // lhbrx/lwbrx: guest stored it little-endian
          uint64_t sw = 0;
          for (uint32_t i = 0; i < size; i++) sw = (sw << 8) | ((v >> (8 * i)) & 0xFF);
          v = sw;
        }
        if (fp) {
          double d = 0;
          if (size == 8) {
            memcpy(&d, &v, 8);
          } else {
            const uint32_t bits = uint32_t(v);
            float f;
            memcpy(&f, &bits, 4);
            d = double(f);
          }
          ctx.fpr[di.rd] = d;
        } else if (size == 8) {
          setGpr(ctx, di.rd, v);
        } else if (size == 4) {
          setGpr(ctx, di.rd, di.has(PpcFlag::kFlagSigned) ? uint64_t(int64_t(int32_t(uint32_t(v))))
                                                            : trunc32(v));
        } else if (size == 2) {
          setGpr(ctx, di.rd, di.has(PpcFlag::kFlagSigned) ? uint64_t(int64_t(int16_t(uint16_t(v))))
                                                            : trunc32(v));
        } else {
          setGpr(ctx, di.rd, trunc32(v & 0xFF));
        }
      } else {
        uint64_t v = 0;
        if (di.has(PpcFlag::kFlagIntStore)) {
          // stfiwx: the FPR holds a double; store trunc(value) as a big-endian word.
          const double d = ctx.fpr[di.rs];
          const int32_t i = (std::isnan(d) || d > double(INT32_MAX) || d < double(INT32_MIN))
                                ? INT32_MIN
                                : int32_t(std::trunc(d));
          v = uint64_t(uint32_t(i));
        } else if (fp) {
          const double d = ctx.fpr[di.rs];
          if (size == 8) {
            memcpy(&v, &d, 8);
          } else {
            const float f = float(d);
            memcpy(&v, &f, 4);
          }
        } else {
          v = gpr(ctx, di.rs);
          if (di.has(PpcFlag::kFlagSwap)) {
            uint64_t sw = 0;
            for (uint32_t i = 0; i < size; i++) sw = (sw << 8) | ((v >> (8 * i)) & 0xFF);
            v = sw;
          }
        }
        if (!mem.store(ea, size, v)) {
          diag::report_unsupported(diag::Subsystem::Memory, "store outside guest window");
          ctx.exitCode = kExitError;
          ctx.haltReason = 3;
          ctx.pc = ea;
          return true;
        }
        ctx.reserveValid = 0;
      }
      if (di.has(PpcFlag::kFlagUpdate)) setGpr(ctx, di.ra, ea);
      ctx.pc = di.pc + 4;
      return true;
    }

    case PpcClass::LdBlock:
    case PpcClass::StBlock: {  // lmw / stmw
      const int first = di.cls == PpcClass::LdBlock ? di.rd : di.rs;
      const uint64_t ea = eaOf(ctx, di);
      for (int i = first; i <= 31; i++) {
        const uint64_t addr = ea + uint64_t(i - first) * 4;
        if (di.cls == PpcClass::LdBlock) {
          uint64_t v = 0;
          if (!mem.load(addr, 4, &v)) return false;
          setGpr(ctx, i, trunc32(v));
        } else if (!mem.store(addr, 4, gpr(ctx, i))) {
          return false;
        }
      }
      ctx.pc = di.pc + 4;
      return true;
    }

    case PpcClass::Atomic: {
      const uint64_t ea = eaOf(ctx, di);
      const uint32_t size = uint32_t(di.mb);
      const bool isStore = di.rs >= 0;
      if (!isStore) {  // lwarx / ldarx
        uint64_t v = 0;
        if (!mem.load(ea, size, &v)) return false;
        if (size == 4) v = trunc32(v);
        setGpr(ctx, di.rd, v);
        ctx.reserveAddr = uint32_t(ea);
        ctx.reserveValid = 1;
      } else {  // stwcx. / stdcx. (rS set by the decoder for those)
        const uint64_t want = [&] {
          uint64_t x = 0;
          mem.load(ea, size, &x);
          return x;
        }();
        const uint64_t val = size == 4 ? trunc32(gpr(ctx, di.rs)) : gpr(ctx, di.rs);
        const bool ok = ctx.reserveValid && ctx.reserveAddr == uint32_t(ea) && want == val;
        if (ok && !mem.store(ea, size, val)) return false;
        setCrField(ctx, 0, ok ? 2u : 4u);
        ctx.reserveValid = 0;
      }
      ctx.pc = di.pc + 4;
      return true;
    }

    // ── special registers ──
    case PpcClass::Mfspr: {
      const uint32_t spr = uint32_t(di.imm) & 0x3FF;
      uint64_t v = 0;
      switch (spr) {
        case 1: v = ctx.xer; break;
        case 8: v = ctx.lr; break;
        case 9: v = ctx.ctr; break;
        case 26: v = ctx.fpscr; break;
        case 256: v = 0x000000000000001Full; break;  //pir  (6 threads, one per SMT pair)
        case 287: v = ctx.vscr; break;
        default: v = 0; break;
      }
      setGpr(ctx, di.rd, v);
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::Mtspr: {
      const uint32_t spr = uint32_t(di.imm) & 0x3FF;
      const uint64_t v = gpr(ctx, di.rs);
      switch (spr) {
        case 1: ctx.xer = uint32_t(v); break;
        case 8: ctx.lr = v; break;
        case 9: ctx.ctr = v; break;
        case 26: ctx.fpscr = uint32_t(v); break;
        case 287: ctx.vscr = uint32_t(v); break;
        default: break;
      }
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::Mfcr:
      setGpr(ctx, di.rd, ctx.cr);
      ctx.pc = di.pc + 4;
      return true;
    case PpcClass::Mtcrf: {
      const uint32_t fm = uint32_t(di.imm) & 0xFF;
      const uint32_t v = uint32_t(gpr(ctx, di.rs));
      uint32_t cr = 0;
      for (int i = 0; i < 8; i++) cr = (cr << 4) | ((fm & (1u << (7 - i))) ? ((v >> (4 * (7 - i))) & 0xFu)
                                                                          : ((ctx.cr >> (4 * (7 - i))) & 0xFu));
      ctx.cr = cr;
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::Mfmsr:
      setGpr(ctx, di.rd, ctx.msr);
      ctx.pc = di.pc + 4;
      return true;
    case PpcClass::Mtmsr:
      ctx.msr = uint32_t(gpr(ctx, di.rs));
      ctx.pc = di.pc + 4;
      return true;

    // ── branches ──
    case PpcClass::B: {
      if (di.has(PpcFlag::kFlagLink)) ctx.lr = di.pc + 4;
      ctx.pc = uint64_t(di.imm) & 0xFFFFFFFFull;
      if (isHleSentinel(ctx.pc, &ctx.hleIndex)) {
        ctx.pc = di.pc;
        ctx.exitCode = kExitHle;
      }
      return true;
    }
    case PpcClass::Bc:
    case PpcClass::Bclr: {
      const uint32_t bo = uint32_t(di.rd);
      const uint32_t bi = uint32_t(di.ra) & 31;
      const bool isCtr = di.has(PpcFlag::kFlagCount);
      const bool testCtr = (bo & 0x10) == 0;
      const bool invert = (bo & 0x08) != 0;
      const bool decCtr = (bo & 0x04) == 0;
      if (decCtr || isCtr) ctx.ctr = (ctx.ctr - 1) & 0xFFFFFFFFull;
      bool take = invert ? (getCrBit(ctx, int(bi)) == 0) : (getCrBit(ctx, int(bi)) != 0);
      if (testCtr) {
        const bool ctrZero = (ctx.ctr & 0xFFFFFFFFull) == 0;
        take = take && (ctrZero != invert);
      }
      if (di.has(PpcFlag::kFlagLink)) ctx.lr = di.pc + 4;
      uint64_t target;
      if (di.cls == PpcClass::Bc) target = uint64_t(di.imm) & 0xFFFFFFFFull;
      else if (isCtr) target = ctx.ctr & 0xFFFFFFFFull;
      else target = ctx.lr & 0xFFFFFFFFull;
      if (take) {
        if (target == kGuestReturnSentinel) {
          ctx.exitCode = kExitReturn;
          ctx.pc = target;
          return true;
        }
        if (isHleSentinel(target, &ctx.hleIndex)) {
          ctx.exitCode = kExitHle;
          ctx.pc = di.pc;  // re-dispatch continues after the call
          return true;
        }
        ctx.pc = target;
      } else {
        ctx.pc = di.pc + 4;
      }
      return true;
    }
    case PpcClass::Sync:
      ctx.pc = di.pc + 4;
      return true;
    case PpcClass::Trap:
      ctx.exitCode = kExitError;
      ctx.haltReason = 4;
      ctx.pc = di.pc;
      return true;
    case PpcClass::Rfid:
      ctx.pc = ctx.lr & 0xFFFFFFFFull;
      return true;

    // ── floating point ──
    case PpcClass::FpArith: {
      const double a = ctx.fpr[di.ra];
      const double b = ctx.fpr[di.rb];
      const double c = di.rc >= 0 ? ctx.fpr[di.rc] : 0.0;
      const bool single = di.has(PpcFlag::kFlagSingle);
      auto pack = [&](double v) { return single ? double(float(v)) : v; };
      double r = 0;
      switch (int(di.imm)) {
        case kFpAdd: r = pack(a + b); break;
        case kFpSub: r = pack(a - b); break;
        case kFpMul: r = pack(a * b); break;
        case kFpDiv:
          if (b == 0) r = (a == 0) ? NAN : (std::signbit(a) ? -INFINITY : INFINITY);
          else r = pack(a / b);
          break;
        case kFpSqrt: r = pack(b < 0 ? NAN : std::sqrt(b)); break;
        case kFpMadd: {
          // PowerPC: fmadd = rA*rC + rB (note the operand order), and the "n"
          // variants negate only the product.
          double prod = a * c;
          if (di.has(PpcFlag::kFlagNeg)) prod = -prod;
          r = pack(di.has(PpcFlag::kFlagSub) ? (prod - b) : (prod + b));
          break;
        }
        default: return false;
      }
      if (di.has(PpcFlag::kFlagHigh)) r = 1.0 / r;  // fre / frsqrte style estimate
      ctx.fpr[di.rd] = r;
      if (di.has(PpcFlag::kFlagRc)) {
        uint32_t bits;
        if (std::isnan(r)) bits = 1;
        else if (r < 0) bits = 8;
        else if (r > 0) bits = 4;
        else bits = 2;
        setCrField(ctx, 0, bits);
      }
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::FpCmp: {
      const double a = ctx.fpr[di.ra];
      const double b = ctx.fpr[di.rb];
      const bool nan = std::isnan(a) || std::isnan(b);
      uint32_t bits;
      if (nan) bits = 1;
      else if (a < b) bits = 8;
      else if (a > b) bits = 4;
      else bits = 2;
      if (di.has(PpcFlag::kFlagOrdered) && nan) ctx.fpscr |= 1u << 31;
      setCrField(ctx, di.crf, bits);
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::FpUnary: {
      const double b = ctx.fpr[di.rb];
      double r = b;
      switch (int(di.imm)) {
        case kFpAbs: r = std::fabs(b); break;
        case kFpNeg: r = -b; break;
        case kFpMr: r = b; break;
        case kFpFnabs: r = -std::fabs(b); break;
        case kFpCpsgn: r = std::copysign(ctx.fpr[di.ra], b); break;
        case kFpSel: {
          const double a = ctx.fpr[di.ra];
          const double c = di.rc >= 0 ? ctx.fpr[di.rc] : 0.0;
          r = (b < 0 ? c : a);
          break;
        }
        default: return false;
      }
      if (di.has(PpcFlag::kFlagSingle)) r = double(float(r));
      ctx.fpr[di.rd] = r;
      if (di.has(PpcFlag::kFlagRc)) {
        uint32_t bits;
        if (std::isnan(r)) bits = 1;
        else if (r < 0) bits = 8;
        else if (r > 0) bits = 4;
        else bits = 2;
        setCrField(ctx, 0, bits);
      }
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::FpSel: {
      const double a = ctx.fpr[di.ra];
      const double b = ctx.fpr[di.rb];
      const double c = di.rc >= 0 ? ctx.fpr[di.rc] : 0.0;
      ctx.fpr[di.rd] = (b < 0 ? c : a);
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::FpRound: {
      double v = ctx.fpr[di.rb];
      if (di.has(PpcFlag::kFlagSingle)) v = double(float(v));  // frsp / frsp.
      ctx.fpr[di.rd] = v;
      if (di.has(PpcFlag::kFlagRc)) {
        uint32_t bits;
        if (std::isnan(v)) bits = 1;
        else if (v < 0) bits = 8;
        else if (v > 0) bits = 4;
        else bits = 2;
        setCrField(ctx, 0, bits);
      }
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::FpConv: {
      const double v = ctx.fpr[di.rb];
      uint64_t bits = 0;
      switch (int(di.imm)) {
        case kFpCtiwz: {
          const double t = std::trunc(v);
          const int32_t i = (std::isnan(v) || t > double(INT32_MAX) || t < double(INT32_MIN))
                                ? int32_t(INT32_MIN)
                                : int32_t(t);
          memcpy(&bits, &v, 8);
          bits = (bits & ~0xFFFFFFFFull) | uint64_t(uint32_t(i));
          break;
        }
        case kFpCtidz: {
          const double t = std::trunc(v);
          const int64_t i = (std::isnan(v) || t > double(INT64_MAX) || t < double(INT64_MIN))
                                ? int64_t(INT64_MIN)
                                : int64_t(t);
          bits = uint64_t(i);
          break;
        }
        case kFpCtiw: {
          // fctiw: round to nearest (even) first, then clamp; the result is placed in
          // the low word of an FPR, exactly like fctiwz but with FPSCR rounding.
          const double t = std::nearbyint(v);
          const int32_t i = (std::isnan(v) || t > double(INT32_MAX) || t < double(INT32_MIN))
                                ? int32_t(INT32_MIN)
                                : int32_t(t);
          memcpy(&bits, &v, 8);
          bits = (bits & ~0xFFFFFFFFull) | uint64_t(uint32_t(i));
          break;
        }
        case kFpCtsw: {  // fctsws / fctswz: 16-bit result
          const double t = std::trunc(v);
          const int16_t i =
              (std::isnan(v) || t > double(INT16_MAX) || t < double(INT16_MIN))
                  ? int16_t(INT16_MIN)
                  : int16_t(t);
          memcpy(&bits, &v, 8);
          bits = (bits & ~0xFFFFull) | uint64_t(uint16_t(i));
          break;
        }
        case kFpCtid:
        case kFpFcid: {
          int64_t i = 0;
          memcpy(&i, &ctx.fpr[di.rb], 8);
          const double d = double(i);
          memcpy(&bits, &d, 8);
          break;
        }
        default: return false;
      }
      memcpy(&ctx.fpr[di.rd], &bits, 8);
      ctx.pc = di.pc + 4;
      return true;
    }

    // ── VMX128 ──
    case PpcClass::VmxLoad:
    case PpcClass::VmxStore: {
      const uint64_t ea = eaOf(ctx, di);
      if (!mem.ok(ea, 16)) return false;
      if (di.cls == PpcClass::VmxLoad) memcpy(ctx.vmx[di.rd & 127], mem.base + (ea - mem.guestBase), 16);
      else memcpy(mem.base + (ea - mem.guestBase), ctx.vmx[di.rs & 127], 16);
      if (di.has(PpcFlag::kFlagUpdate)) setGpr(ctx, di.ra, ea);
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::VmxLogic: {
      const uint8_t* a = ctx.vmx[di.ra & 127];
      const uint8_t* b = ctx.vmx[di.rb & 127];
      uint8_t* d = ctx.vmx[di.rd & 127];
      uint8_t tmp[16];
      for (int i = 0; i < 16; i++) {
        switch (int(di.imm)) {
          case kVmxAnd: tmp[i] = uint8_t(a[i] & b[i]); break;
          case kVmxAndc: tmp[i] = uint8_t(a[i] & ~b[i]); break;
          case kVmxOr: tmp[i] = uint8_t(a[i] | b[i]); break;
          case kVmxOrc: tmp[i] = uint8_t(a[i] | ~b[i]); break;
          case kVmxXor: tmp[i] = uint8_t(a[i] ^ b[i]); break;
          case kVmxNor: tmp[i] = uint8_t(~(a[i] | b[i])); break;
          default: return false;
        }
      }
      memcpy(d, tmp, 16);
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::VmxArith: {
      const int esz = di.mb;
      const bool sat = di.has(PpcFlag::kFlagSat);
      const bool sgn = di.has(PpcFlag::kFlagSigned);
      const bool sub = int(di.imm) == kVmxSub;
      const int n = 16 / esz;
      uint8_t out[16];
      memcpy(out, ctx.vmx[di.rd & 127], 16);
      bool anySat = false;
      for (int lane = 0; lane < n; lane++) {
        const int off = lane * esz;
        uint64_t x = 0, y = 0;
        for (int k = 0; k < esz; k++) {
          x = (x << 8) | ctx.vmx[di.ra & 127][off + k];
          y = (y << 8) | ctx.vmx[di.rb & 127][off + k];
        }
        uint64_t r;
        if (esz == 8) {  // 64-bit integer lanes (vaddudm)
          r = sub ? (x - y) : (x + y);
        } else if (sgn) {
          const int64_t sx = esz == 1 ? int64_t(int8_t(uint8_t(x)))
                           : esz == 2 ? int64_t(int16_t(uint16_t(x)))
                                      : int64_t(int32_t(uint32_t(x)));
          const int64_t sy = esz == 1 ? int64_t(int8_t(uint8_t(y)))
                           : esz == 2 ? int64_t(int16_t(uint16_t(y)))
                                      : int64_t(int32_t(uint32_t(y)));
          int64_t t = sub ? sx - sy : sx + sy;
          const int64_t lo = esz == 1 ? INT8_MIN : esz == 2 ? INT16_MIN : INT32_MIN;
          const int64_t hi = esz == 1 ? INT8_MAX : esz == 2 ? INT16_MAX : INT32_MAX;
          if (sat && t < lo) { t = lo; anySat = true; }
          if (sat && t > hi) { t = hi; anySat = true; }
          r = uint64_t(t);
        } else {
          const uint64_t hi = (esz == 1) ? 0xFFull : (esz == 2) ? 0xFFFFull : 0xFFFFFFFFull;
          const uint64_t ux = x & hi, uy = y & hi;
          if (sub) {
            if (sat && ux < uy) { r = 0; anySat = true; }
            else r = (ux - uy) & hi;
          } else {
            const uint64_t t = ux + uy;
            if (sat && t > hi) { r = hi; anySat = true; }
            else r = t & hi;
          }
        }
        for (int k = 0; k < esz; k++) out[off + k] = uint8_t(r >> (8 * (esz - 1 - k)));
      }
      memcpy(ctx.vmx[di.rd & 127], out, 16);
      if (anySat) ctx.vscr |= 2;
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::VmxMinMax: {
      const int esz = di.mb;
      const bool sgn = int(di.imm) <= 1;
      const bool isMax = int(di.imm) != kVmxMinS;
      const int n = 16 / esz;
      uint8_t out[16];
      memcpy(out, ctx.vmx[di.rd & 127], 16);
      for (int lane = 0; lane < n; lane++) {
        const int off = lane * esz;
        int64_t x = 0, y = 0;
        for (int k = 0; k < esz; k++) {
          x = (x << 8) | ctx.vmx[di.ra & 127][off + k];
          y = (y << 8) | ctx.vmx[di.rb & 127][off + k];
        }
        if (sgn) {
          if (esz == 1) { x = int64_t(int8_t(uint8_t(x))); y = int64_t(int8_t(uint8_t(y))); }
          else if (esz == 2) { x = int64_t(int16_t(uint16_t(x))); y = int64_t(int16_t(uint16_t(y))); }
          else { x = int64_t(int32_t(uint32_t(x))); y = int64_t(int32_t(uint32_t(y))); }
        } else {
          const uint64_t lim = esz == 1 ? 0xFFull : esz == 2 ? 0xFFFFull : 0xFFFFFFFFull;
          x &= lim;
          y &= lim;
        }
        const int64_t r = isMax ? (x > y ? x : y) : (x < y ? x : y);
        for (int k = 0; k < esz; k++) out[off + k] = uint8_t(uint64_t(r) >> (8 * (esz - 1 - k)));
      }
      memcpy(ctx.vmx[di.rd & 127], out, 16);
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::VmxCmp: {
      const int esz = di.mb;
      const int sel = int(di.imm);
      const bool fp = sel >= kVmxCmpEqFp;
      const bool sgn = di.has(PpcFlag::kFlagSigned);
      const int e = fp ? 4 : esz;
      const int n = 16 / e;
      uint8_t out[16];
      memcpy(out, ctx.vmx[di.rd & 127], 16);
      bool any = false, all = true;
      for (int lane = 0; lane < n; lane++) {
        const int off = lane * e;
        bool result;
        if (fp) {
          uint32_t ax = 0, ay = 0;
          for (int k = 0; k < 4; k++) {
            ax = (ax << 8) | ctx.vmx[di.ra & 127][off + k];
            ay = (ay << 8) | ctx.vmx[di.rb & 127][off + k];
          }
          float x, y;
          memcpy(&x, &ax, 4);
          memcpy(&y, &ay, 4);
          switch (sel) {
            case kVmxCmpEqFp: result = (x == y); break;
            case kVmxCmpGeFp: result = !(x < y); break;
            case kVmxCmpGtFp: result = (x > y); break;
            case kVmxCmpRange: result = !(x < -1.0f || x > 1.0f || std::isnan(x)); break;
            default: return false;
          }
        } else {
          int64_t x = 0, y = 0;
          for (int k = 0; k < e; k++) {
            x = (x << 8) | ctx.vmx[di.ra & 127][off + k];
            y = (y << 8) | ctx.vmx[di.rb & 127][off + k];
          }
          if (sgn && e < 8) {
            const int64_t sb = e * 8 - 1;
            const uint64_t lim = (1ull << (e * 8)) - 1;
            if (uint64_t(x) & (1ll << sb)) x = -((~int64_t(x)) & int64_t(lim)) - 1;
            if (uint64_t(y) & (1ll << sb)) y = -((~int64_t(y)) & int64_t(lim)) - 1;
          }
          switch (sel) {
            case kVmxCmpEq: result = (x == y); break;
            case kVmxCmpGt: result = (x > y); break;
            default: return false;
          }
        }
        for (int k = 0; k < e; k++) out[off + k] = result ? 0xFF : 0x00;
        if (result) any = true; else all = false;
      }
      memcpy(ctx.vmx[di.rd & 127], out, 16);
      // CR6: bit 0 = "all lanes true", bit 1 = "any lane true" (documented Xenia
      // behaviour for the vcmp* family).
      const uint32_t cr6 = (all ? 8u : 0u) | (any ? 4u : 0u);
      setCrField(ctx, 6, cr6);
      ctx.vscr = (ctx.vscr & ~0xF0u) | ((cr6 & 0xFu) << 4);
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::VmxMul: {
      const int sel = int(di.imm);
      if (sel <= kVmxMulFp) {
        float d[4];
        for (int i = 0; i < 4; i++) {
          const auto lane = [&](const uint8_t* v) {
            uint32_t bits = 0;
            for (int k = 0; k < 4; k++) bits = (bits << 8) | v[i * 4 + k];
            float f;
            memcpy(&f, &bits, 4);
            return f;
          };
          const float a = lane(ctx.vmx[di.ra & 127]);
          const float b = lane(ctx.vmx[di.rb & 127]);
          const float c = lane(ctx.vmx[di.rc & 127]);
          float r;
          if (sel == kVmxMulFp) r = a * b;
          else if (sel == kVmxMaddFp) r = a * b + c;
          else r = -(a * b) - c;
          d[i] = r;
        }
        for (int i = 0; i < 4; i++) {
          uint32_t bits;
          memcpy(&bits, &d[i], 4);
          for (int k = 0; k < 4; k++) ctx.vmx[di.rd & 127][i * 4 + k] = uint8_t(bits >> (8 * (3 - k)));
        }
        ctx.pc = di.pc + 4;
        return true;
      }
      return false;  // vmules/vmulos/vmsum*: not implemented yet (reported)
    }
    case PpcClass::VmxSel: {
      uint8_t out[16];
      for (int i = 0; i < 16; i++)
        out[i] = (ctx.vmx[di.rc & 127][i] & 0x80) ? ctx.vmx[di.ra & 127][i] : ctx.vmx[di.rb & 127][i];
      memcpy(ctx.vmx[di.rd & 127], out, 16);
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::VmxPerm: {
      uint8_t out[16];
      for (int i = 0; i < 16; i++) {
        const uint8_t idx = ctx.vmx[di.rc & 127][i] & 0x1F;
        const uint8_t* src = idx < 16 ? ctx.vmx[di.ra & 127] : ctx.vmx[di.rb & 127];
        out[i] = src[idx & 15];
      }
      memcpy(ctx.vmx[di.rd & 127], out, 16);
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::VmxSldoi: {
      const int sh = di.sh & 15;
      uint8_t tmp[32];
      memcpy(tmp, ctx.vmx[di.ra & 127], 16);
      memcpy(tmp + 16, ctx.vmx[di.rb & 127], 16);
      memcpy(ctx.vmx[di.rd & 127], tmp + sh, 16);
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::VmxShift: {
      const int sel = int(di.imm);
      if (sel <= kVmSplatIsW) {
        const int esz = (sel == kVmSplatB || sel == kVmSplatIsB) ? 1
                      : (sel == kVmSplatH || sel == kVmSplatIsH) ? 2
                                                                 : 4;
        const bool imm = sel >= kVmSplatIsB;
        const int idx = di.sh;
        uint8_t out[16];
        memcpy(out, ctx.vmx[di.rd & 127], 16);
        for (int lane = 0; lane < 16 / esz; lane++) {
          uint64_t v = 0;
          if (imm) {
            const int8_t s = int8_t(uint8_t(idx));
            v = (esz == 1) ? uint64_t(uint8_t(s))
              : (esz == 2) ? uint64_t(uint16_t(uint16_t(s)))
                           : uint64_t(uint32_t(uint32_t(s)));
          } else {
            const int srcOff = (sel == kVmSplatB ? idx : sel == kVmSplatH ? idx * 2 : idx * 4);
            for (int k = 0; k < esz; k++) v = (v << 8) | ctx.vmx[di.rb & 127][srcOff + k];
          }
          for (int k = 0; k < esz; k++) out[lane * esz + k] = uint8_t(v >> (8 * (esz - 1 - k)));
        }
        memcpy(ctx.vmx[di.rd & 127], out, 16);
        ctx.pc = di.pc + 4;
        return true;
      }
      // vsl/vsr per-lane shift with the shift-count vector in rB
      const int esz = (sel == kVmShiftLoB || sel == kVmShiftHiB) ? 1
                    : (sel == kVmShiftLoH || sel == kVmShiftHiH) ? 2
                                                                 : 4;
      const bool right = sel >= kVmShiftHiW;
      uint8_t out[16];
      memcpy(out, ctx.vmx[di.rd & 127], 16);
      for (int lane = 0; lane < 16 / esz; lane++) {
        const int off = lane * esz;
        uint64_t v = 0, sh = 0;
        for (int k = 0; k < esz; k++) v = (v << 8) | ctx.vmx[di.ra & 127][off + k];
        for (int k = 0; k < esz; k++) sh = (sh << 8) | ctx.vmx[di.rb & 127][off + k];
        if (sh >= uint64_t(esz * 8)) v = 0;
        else v = right ? (v >> sh) : (v << sh);
        const uint64_t lim = esz == 1 ? 0xFFull : esz == 2 ? 0xFFFFull : 0xFFFFFFFFull;
        v &= lim;
        for (int k = 0; k < esz; k++) out[off + k] = uint8_t(v >> (8 * (esz - 1 - k)));
      }
      memcpy(ctx.vmx[di.rd & 127], out, 16);
      ctx.pc = di.pc + 4;
      return true;
    }
    case PpcClass::VmxFloat:
      return false;  // vcfux/vctsxs/vrefp/... : reported as unimplemented

    default:
      return false;
  }
}

// ─── Public entry points ──────────────────────────────────────────────────────
bool stepOneGuestInstruction(PPCContext& ctx) {
  Mem mem = memOf(ctx);
  const uint64_t pc = ctx.pc & 0xFFFFFFFFull;
  uint64_t raw = 0;
  if (!mem.load(pc, 4, &raw)) {
    ctx.exitCode = kExitError;
    ctx.haltReason = 5;
    X360LOGE(TAG, "t%d fetch fault at guest PC 0x%08llX", int(ctx.threadIndex),
             (unsigned long long)pc);
    return false;
  }
  const DecodedPpc di = ppcDecode(uint32_t(raw), pc);
  if (di.cls == PpcClass::Invalid) {
    char buf[160];
    snprintf(buf, sizeof(buf), "unimplemented PPC instruction 0x%08X at 0x%08llX", uint32_t(raw),
             (unsigned long long)pc);
    diag::report_unsupported(diag::Subsystem::Kernel, buf);
    ctx.exitCode = kExitError;
    ctx.haltReason = 6;
    return false;
  }
  const bool ok = runDecoded(ctx, di);
  if (!ok) {
    char buf[160];
    snprintf(buf, sizeof(buf), "PPC class %d (word 0x%08X) unimplemented at 0x%08llX", int(di.cls),
             uint32_t(raw), (unsigned long long)pc);
    diag::report_unsupported(diag::Subsystem::Kernel, buf);
    ctx.exitCode = kExitError;
    ctx.haltReason = 7;
  }
  return ok;
}

uint32_t interpretRange(PPCContext& ctx, uint32_t maxInstrs, uint64_t stopPc) {
  for (uint32_t i = 0; i < maxInstrs; i++) {
    const uint64_t pc = ctx.pc & 0xFFFFFFFFull;
    if (stopPc && pc == (stopPc & 0xFFFFFFFFull)) return kExitContinue;
    uint32_t importIndex = 0;
    if (isHleSentinel(pc, &importIndex)) {
      ctx.hleIndex = importIndex;
      return kExitHle;
    }
    if (pc == kGuestReturnSentinel) return kExitReturn;
    ctx.exitCode = kExitContinue;
    if (!stepOneGuestInstruction(ctx)) return ctx.exitCode ? ctx.exitCode : kExitError;
    if (ctx.exitCode != kExitContinue) return ctx.exitCode;
  }
  return kExitYield;
}

}  // namespace jit
}  // namespace x360
