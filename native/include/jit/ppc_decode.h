/**
 * Shared PowerPC Xenon instruction classification.
 *
 * This header + ppc_decode.cpp are the single definition of guest ISA semantics
 * used by BOTH the interpreter (jit_interpreter.cpp) and the ARM64 backend
 * (arm64_backend.cpp). The backend only translates instructions whose class it
 * can prove it implements bit-exactly; everything else runs through the
 * interpreter. That is what makes "JIT result == interpreter result" a checkable
 * property instead of a hope, and it is why an unimplemented opcode can never
 * silently become a NOP (the bug in the previous version, which produced
 * "hangs at 100% CPU with no log line").
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace x360 {
namespace jit {

// ─── Field extraction ─────────────────────────────────────────────────────────
static inline uint32_t ppc_primary(uint32_t i) { return i >> 26; }
static inline uint32_t ppc_xo10(uint32_t i) { return (i >> 1) & 0x3FF; }
static inline uint32_t ppc_xo9(uint32_t i) { return (i >> 1) & 0x1FF; }
static inline int ppc_ra(uint32_t i) { return (i >> 16) & 0x1F; }
static inline int ppc_rb(uint32_t i) { return (i >> 11) & 0x1F; }
static inline int ppc_rc(uint32_t i) { return (i >> 6) & 0x1F; }
static inline int ppc_rd(uint32_t i) { return (i >> 21) & 0x1F; }
static inline int ppc_rs(uint32_t i) { return (i >> 21) & 0x1F; }
static inline int ppc_vd(uint32_t i) { return (i >> 21) & 0x7F; }
static inline int ppc_va(uint32_t i) { return (i >> 16) & 0x7F; }
static inline int ppc_vb(uint32_t i) { return (i >> 11) & 0x7F; }
static inline int ppc_vc(uint32_t i) { return (i >> 6) & 0x7F; }
static inline int ppc_sh(uint32_t i) { return (i >> 11) & 0x1F; }
static inline int ppc_mb(uint32_t i) { return (i >> 6) & 0x1F; }
static inline int ppc_me(uint32_t i) { return (i >> 1) & 0x1F; }
static inline int ppc_bf(uint32_t i) { return (i >> 23) & 7; }
static inline int ppc_bo(uint32_t i) { return (i >> 21) & 0x1F; }
static inline int ppc_bi(uint32_t i) { return (i >> 16) & 0x1F; }
static inline int ppc_spr(uint32_t i) { return ((i >> 11) & 0x1F) | (((i >> 16) & 0x1F) << 5); }
static inline int ppc_fra(uint32_t i) { return (i >> 16) & 0x1F; }
static inline int ppc_frb(uint32_t i) { return (i >> 11) & 0x1F; }
static inline int ppc_frc(uint32_t i) { return (i >> 6) & 0x1F; }
static inline int ppc_frd(uint32_t i) { return (i >> 21) & 0x1F; }
// D-form memory size / index register (rd/rs already give the data register).
static inline int ppc_d_size(uint32_t i) { return ppc_rd(i); }
static inline int ppc_d_si(uint32_t i) { return ppc_rc(i); }
static inline int ppc_frd2(uint32_t i) { return ppc_rd(i); }   // store FP data reg
static inline int32_t ppc_simm16(uint32_t i) { return int32_t(int16_t(i & 0xFFFF)); }
static inline uint32_t ppc_uimm16(uint32_t i) { return i & 0xFFFF; }
static inline int32_t ppc_bd(uint32_t i) { return int32_t(int16_t(i & 0xFFFC)); }
static inline int32_t ppc_li(uint32_t i) {
  return int32_t((i & 0x03FFFFFCu) ^ 0x02000000u) - 0x02000000;
}
static inline int32_t ppc_dq(uint32_t i) {  // ds-form displacement (already x4)
  int32_t v = int32_t(i & 0xFFFCu);
  if (v & 0x8000) v |= ~0xFFFF;
  return v;
}
static inline bool ppc_aa(uint32_t i) { return ((i >> 1) & 1) != 0; }
static inline bool ppc_lk(uint32_t i) { return (i & 1) != 0; }
static inline bool ppc_rc_bit(uint32_t i) { return (i & 1) != 0; }

static inline uint32_t rotl32(uint32_t v, int n) {
  n &= 31;
  return n ? ((v << n) | (v >> (32 - n))) : v;
}
static inline uint64_t rotl64(uint64_t v, int n) {
  n &= 63;
  return n ? ((v << n) | (v >> (64 - n))) : v;
}
// PPC mask with bits [mb..me] set, bit 0 = MSB. width is 32 or 64.
static inline uint64_t ppc_mask(int mb, int me, int width) {
  uint64_t m = 0;
  int i = mb;
  while (true) {
    m |= 1ull << uint64_t(width - 1 - i);
    if (i == me) break;
    i = (i + 1) % width;
  }
  return m;
}

// ─── Instruction classes ──────────────────────────────────────────────────────
// Field usage per class (unused fields are -1 / 0):
//   Add      rd, ra, rb|imm     kFlagImm kFlagCarry kFlagRc kFlagZero32
//   Subf     rd, ra(rA), rb      " (result = rb - ra, or ~ra+rb+imm for .O forms)
//   Mul      rd, ra, rb|imm     kFlagImm kFlagHigh kFlagWide kFlagSigned
//   Div      rd, ra, rb         kFlagSigned kFlagWide (0 = 32-bit)
//   Neg      rd, ra             kFlagWide
//   Popcnt   rd, ra             0=byte 1=word 2=dword
//   Cntlz    rd, ra             kFlagWide
//   Extend   rd, ra             0=byte 1=half 2=word  (32 -> 64 sign extend)
//   Logic    rd, ra, rb         sel = imm (see LogicSel)
//   LogicImm rd, ra, imm        sel = rb (0=and 2=xor 4=or) + kFlagRc
//   Cmp      crf, ra, rb|imm    kFlagImm kFlagWide kFlagSigned
//   Rl       rd, ra, rb|imm     sh/mb/me  kFlagImm kFlagWide kFlagInsert
//   Ld/St    rd|rs, ra, rb|imm  size = mb, kFlagIndex kFlagUpdate kFlagSign
//   LsEx     rd, ra             kFlagSingle (lfs/lfdu...)  (same as Ld path)
//   Atomic   rd|rs, ra, rb      kFlagWide
//   Mfspr/Mtspr  rd|rs, imm=SPR
//   Mfcr     rd     Mtcrf  rs, imm=FM
//   B        imm      kFlagLink
//   Bc       bo(=rd), bi(=ra), imm  kFlagLink
//   Bclr     bo, bi, kFlagCount(kFlagZero32)  kFlagLink
//   FpArith  rd, ra, rb, rc     op = imm, kFlagSingle, kFlagSub, kFlagNeg
//   FpCmp    crf, ra, rb        kFlagOrdered
//   FpRound  rd, rb             kFlagSingle
//   FpConv   rd, rb             kFlagToFloat kFlagWide kFlagSingle
//   FpUnary  rd, rb             imm: 0=abs 1=neg 2=mr, kFlagRc => fmr.
//   VmxLogic rd, ra, rb         sel = imm
//   VmxArith rd, ra, rb         sel = imm (op), elem = mb, signedness in flags
//   VmxMul   rd, ra, rb, rc     sel = imm
//   VmxSel   rd, ra, rb, rc
//   VmxPerm  rd, ra, rb
//   VmxShift rd, ra, rb         element size in mb
//   VmxSldoi rd, ra, rb, sh=rc
//   VmxCmp   rd, ra, rb         sel = imm, crf=6 (VSCR) or 0..7 per class
//   VmxPack  rd, ra, rb         sel = imm
//   VmxFloat rd, ra, rb         element size in mb, kFlagSigned
//   VmxLoad  rd, ra, rb         kFlagIndex (lvx/lvrx/lvsl/lvsr)
//   VmxStore rs(=rd field), ra, rb
//   Sync/Eieio/Lwsync/Isync/Nop/Trap/Rfid/Invalid — no operands
enum class PpcClass : uint16_t {
  Invalid,
  Nop,
  Add,
  Subf,
  Mul,
  Div,
  Neg,
  Popcnt,
  Cntlz,
  Extend,
  Logic,
  LogicImm,
  Cmp,
  Rl,
  Ld,
  St,
  LdBlock,
  StBlock,
  Mfmsr,
  Mtmsr,
  Subfc,
  Subfe,
  Atomic,
  Mfspr,
  Mtspr,
  Mfcr,
  Mtcrf,
  B,
  Bc,
  Bclr,
  Sync,
  Trap,
  Rfid,
  FpArith,
  FpSub,
  FpCmp,
  FpRound,
  FpConv,
  FpUnary,
  FpSel,
  VmxLogic,
  VmxArith,
  VmxMul,
  VmxSel,
  VmxPerm,
  VmxShift,
  VmxSldoi,
  VmxCmp,
  VmxPack,
  VmxFloat,
  VmxMinMax,
  VmxLoad,
  VmxStore,
  Dcbz,     // dcbz: zero the 128-byte guest line (it is a *store*, not a hint)
  Count,
};

struct DecodedPpc {
  PpcClass cls = PpcClass::Invalid;
  uint32_t word = 0;
  uint64_t pc = 0;
  int rd = -1, ra = -1, rb = -1, rc = -1, rs = -1;
  int64_t imm = 0;   // immediate / branch target / SPR / selector
  int mb = 0;        // element size (bytes) or mask begin
  int me = 0;        // mask end
  int sh = 0;        // shift amount
  int crf = 0;       // CR field
  uint32_t flags = 0;
  bool term = false;  // ends the basic block (can change PC)

  bool has(uint32_t f) const { return (flags & f) != 0; }
  bool isVmx() const { return cls >= PpcClass::VmxLogic && cls <= PpcClass::VmxStore; }
};

enum PpcFlag : uint32_t {
  kFlagRc = 1u << 0,
  kFlagLink = 1u << 1,
  kFlagUpdate = 1u << 2,
  kFlagSigned = 1u << 3,
  kFlagImm = 1u << 4,
  kFlagSingle = 1u << 5,
  kFlagCarry = 1u << 6,
  kFlagIndex = 1u << 7,
  kFlagHigh = 1u << 8,     // mul*hw / mulh*
  kFlagWide = 1u << 9,     // 64-bit form (divw/d, addc/d ...)
  kFlagSwap = 1u << 10,    // byte-reversed access
  kFlagZero32 = 1u << 11,  // result is zero-extended from 32 bits
  kFlagInsert = 1u << 12,  // rlwimi/rldimi (merge with previous rA)
  kFlagSub = 1u << 13,     // fmsub/fnmsub, or "rb - ra" ordering
  kFlagNeg = 1u << 14,     // fnmadd/fnmsub / vsub
  kFlagToFloat = 1u << 15,
  kFlagOrdered = 1u << 16,  // fcmpo
  kFlagCount = 1u << 17,    // bclr uses CTR (bcctr)
  kFlagSat = 1u << 18,      // saturating VMX op
  kFlagOvf = 1u << 19,      // Xenon division form that sets XER[OV]
  kFlagTermConst = 1u << 20,
  kFlagVmxD = 1u << 21,     // VMX V3 format (rD/rA/rB/rC are 7-bit)
  kFlagVmxVA = 1u << 22,
  kFlagHint = 1u << 23,     // cache-hint style op, no architectural effect
  kFlagVmxA = 1u << 24,     // VMX A-form operand layout (VD/VA/VB)
  kFlagIntStore = 1u << 25, // stfiwx: store the truncated *integer* of an FPR
};

// Selector values stored in `imm` for the multi-op classes. These are part of
// the decode<->interpret contract; tools/gen_ppc_tables.py uses the same
// numbering, and tests/ppc_decode_test.cpp pins it down.
enum LogicSel : int { kLogicAnd = 0, kLogicAndc = 1, kLogicNand = 2, kLogicNor = 3,
                      kLogicOr = 4, kLogicOrc = 5, kLogicXor = 6, kLogicEqv = 7 };
enum AddSel : int { kAdd = 0, kAdde = 1, kAddze = 2, kAddc = 3, kAddme = 4 };
enum SubfSel : int { kSubf = 0, kSubfe = 1, kSubfc = 2, kSubfze = 3, kSubfme = 4 };
enum MulSel : int { kMulWord = 0, kMulHigh = 1 };
enum CmpSel : int { kCmpW = 0, kCmplW = 1, kCmpD = 2, kCmplD = 3 };
enum MemSel : int { kMemNormal = 0, kMemSwap = 1, kMemStfiwx = 2, kMemFloat = 4 };
enum RlSel : int { kRlShiftLeft = 0, kRlShiftRight = 1, kRlShiftRightA = 2,
                   kRlRotateMasked = 3 };  // rlwinm/rldicl/... explicit mask
enum FpArithSel : int { kFpAdd = 0, kFpMul = 1, kFpMadd = 2, kFpDiv = 3, kFpSqrt = 4,
                        kFpSub = 5 };
enum FpConvSel : int { kFpCtiwz = 0, kFpCtidz = 1, kFpFcid = 2, kFpCtiw = 3, kFpCtsw = 4,
                       kFpCtid = 5, kFpCtsws = 6 };
enum FpUnarySel : int { kFpAbs = 0, kFpNeg = 1, kFpMr = 2, kFpSel = 3, kFpFnabs = 4,
                        kFpCpsgn = 5 };
enum VmxLogicSel : int { kVmxAnd = 0, kVmxAndc = 1, kVmxOr = 2, kVmxOrc = 3, kVmxXor = 4,
                          kVmxNor = 5 };
enum VmxArithSel : int { kVmxAdd = 0, kVmxSub = 1 };
enum VmxMinMaxSel : int { kVmxMinS = 0, kVmxMaxS = 1, kVmxMaxU = 2 };
enum VmxCmpSel : int { kVmxCmpEq = 0, kVmxCmpGt = 1, kVmxCmpEqFp = 2, kVmxCmpGeFp = 3,
                       kVmxCmpGtFp = 4, kVmxCmpRange = 5 };
enum VmxDotSel : int { kVmxMaddFp = 0, kVmxNmsubFp = 1, kVmxMulFp = 2 };
enum VmxShiftSel : int { kVmSplatB = 0, kVmSplatH = 1, kVmSplatW = 2, kVmSplatIsB = 3,
                         kVmSplatIsH = 4, kVmSplatIsW = 5, kVmShiftLoW = 6,
                         kVmShiftHiW = 7, kVmShiftLoH = 8, kVmShiftLoB = 9,
                         kVmShiftHiW2 = 10, kVmShiftHiH = 11, kVmShiftHiB = 12 };

// Decode a raw word. Never fails: unknown encodings map to PpcClass::Invalid so
// callers can report "unimplemented instruction 0x%08X at 0x%llX" instead of
// executing nothing.
DecodedPpc ppcDecode(uint32_t word, uint64_t pc);

}  // namespace jit
}  // namespace x360
