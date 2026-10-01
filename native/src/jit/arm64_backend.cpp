/**
 * ARM64 (AArch64) code generation backend.
 *
 * Model: "context memory-operand". All guest architectural state lives in
 * PPCContext; generated code loads from [Xctx + offsetof(...)], computes with
 * real ARM64 instructions, and stores back. There is deliberately no persistent
 * register allocation across guest instructions: it is the simplest translation
 * scheme where "JIT result == interpreter result" is checkable, and it cannot
 * corrupt guest state when an HLE call happens in the middle of a block (the bug
 * the previous revision had, where guest r0..r27 were pinned to x0..x27 with no
 * save/restore around calls).
 *
 * Register use (all caller-saved, so no prologue/epilogue and no ABI break):
 *   x9  XCTX  PPCContext* (first argument, per the AAPCS64)
 *   x10 XA    address scratch
 *   x11 XB    value scratch
 *   x12 XC    value scratch
 *   x13 XD    value scratch / host-memory magic
 *   x14 XI1   guestBase
 *   x15 XHST  hostMem (host pointer of guest address 0)
 *   x16 XCT   constant scratch
 *
 * Guest addressing: the guest window is one flat 4 GB allocation mapped at host
 * address `hostMem`, with guest address 0 == hostMem. So
 *     host = guestAddr + hostMem
 * and the bounds test is a plain 32-bit `guestAddr < guestSize` compare (the
 * guest address space cannot exceed 4 GB, so no wrap is possible). An access
 * outside the window branches to the block's error exit, which returns
 * kExitError and the *interpreter* re-executes that single instruction and
 * produces the precise diagnostic — generated code can therefore never fault.
 *
 * Only instructions whose semantics this file implements bit-exactly are
 * translated; `backendSupports()` is the single gate, and everything else simply
 * ends the block so the interpreter runs it. Declining is always the right answer
 * when unsure: performance loss is recoverable, silent mis-emulation is not.
 */
#include "../../include/jit/jit_engine.h"
#include "../../include/jit/ppc_decode.h"
#include "../../include/x360/log.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#define TAG "X360:ARM64"

namespace x360 {
namespace jit {
namespace detail {

constexpr bool kHostIsArm64 =
#if defined(__aarch64__)
    true;
#else
    false;
#endif

// ─── Context offsets (validated against PPCContext by static_assert) ──────────
constexpr uint32_t kGpr0 = uint32_t(offsetof(PPCContext, gpr));
constexpr uint32_t kFpr0 = uint32_t(offsetof(PPCContext, fpr));
constexpr uint32_t kCr = uint32_t(offsetof(PPCContext, cr));
constexpr uint32_t kXer = uint32_t(offsetof(PPCContext, xer));
constexpr uint32_t kCtr = uint32_t(offsetof(PPCContext, ctr));
constexpr uint32_t kLr = uint32_t(offsetof(PPCContext, lr));
constexpr uint32_t kPc = uint32_t(offsetof(PPCContext, pc));
constexpr uint32_t kHostMem = uint32_t(offsetof(PPCContext, hostMem));
constexpr uint32_t kGuestBase = uint32_t(offsetof(PPCContext, guestBase));
constexpr uint32_t kGuestSize = uint32_t(offsetof(PPCContext, guestSize));
constexpr uint32_t kHleIndex = uint32_t(offsetof(PPCContext, hleIndex));

static_assert(sizeof(PPCContext) <= 4096,
              "generated code reaches every context field with a 12-bit LDR/STR "
              "immediate; grow the offsets scheme (or shrink the context) if this "
              "fires");
static_assert(kGpr0 + 31 * 8 + 8 <= 4096 && kFpr0 + 31 * 8 + 8 <= 4096,
              "register file must stay inside the immediate window");

constexpr int XCTX = 9, XA = 10, XB = 11, XC = 12, XD = 13, XI1 = 14, XHST = 15, XCT = 16;

enum Cond : uint32_t { kEq = 0, kNe = 1, kHs = 2, kLo = 3, kMi = 4, kPl = 5, kVs = 6, kVc = 7,
                       kHi = 8, kLs = 9, kGe = 10, kLt = 11, kGt = 12, kLe = 13 };

/**
 * Minimal AArch64 encoder.
 *
 * Every encoding below is written out field by field from the ARM ARM (DDI
 * 0487) C6-2 group, and `tools/verify_arm64_asm.py` re-disassembles this file's
 * output with Capstone to make sure the fields are right (an emulator that emits
 * a wrong encoding is worse than one that emits none).
 */
class Asm {
 public:
  size_t label() { return labels_.size(); }
  void bind(size_t l) {
    // Labels must be bound in creation order (the builder never uses two labels at
    // once). Asserting here turns a mis-threaded label into a compile failure
    // rather than a silently wrong branch.
    if (l != labels_.size()) { m_bad = true; return; }
    labels_.push_back(code_.size());
  }
  bool bad() const { return m_bad; }

  void word(uint32_t enc) { code_.push_back(enc); }
  bool empty() const { return code_.empty(); }
  size_t count() const { return code_.size(); }

  void movz(int rd, uint32_t imm16, int shift = 0) {
    // MOVZ xd, #imm16, LSL #shift  sf=1 opc=10 hw=shift
    word(0xD2800000u | (uint32_t(shift) << 21) | ((imm16 & 0xFFFFu) << 5) | uint32_t(rd));
  }
  void movk(int rd, uint32_t imm16, int shift) {
    word(0xF2800000u | (uint32_t(shift) << 21) | ((imm16 & 0xFFFFu) << 5) | uint32_t(rd));
  }
  // Materialise an arbitrary 64-bit constant (up to 4 MOVs, no temp needed).
  void movConst(int rd, uint64_t v) {
    movz(rd, uint32_t(v & 0xFFFFu), 0);
    if ((v >> 16) | (v >> 32) | (v >> 48)) movk(rd, uint32_t((v >> 16) & 0xFFFFu), 1);
    if ((v >> 32) | (v >> 48)) movk(rd, uint32_t((v >> 32) & 0xFFFFu), 2);
    if (v >> 48) movk(rd, uint32_t((v >> 48) & 0xFFFFu), 3);
  }
  void addImm(int rd, int rn, uint32_t imm12, bool sf = true) {
    // ADD xd, xn, #imm12{, LSL #12}
    word((sf ? 0x91000000u : 0x11000000u) | (imm12 << 10) | (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void subImm(int rd, int rn, uint32_t imm12, bool sf = true) {
    word((sf ? 0xD1000000u : 0x51000000u) | (imm12 << 10) | (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void addReg(int rd, int rn, int rm, bool sf = true) {
    word((sf ? 0x8B000000u : 0x0B000000u) | (uint32_t(rm) << 16) | (uint32_t(rn) << 5) |
         uint32_t(rd));
  }
  void subReg(int rd, int rn, int rm, bool sf = true) {
    word((sf ? 0xCB000000u : 0x4B000000u) | (uint32_t(rm) << 16) | (uint32_t(rn) << 5) |
         uint32_t(rd));
  }
  void andReg(int rd, int rn, int rm, bool sf = true) {
    word((sf ? 0x8A000000u : 0x0A000000u) | (uint32_t(rm) << 16) | (uint32_t(rn) << 5) |
         uint32_t(rd));
  }
  void orReg(int rd, int rn, int rm, bool sf = true) {
    word((sf ? 0xAA000000u : 0x2A000000u) | (uint32_t(rm) << 16) | (uint32_t(rn) << 5) |
         uint32_t(rd));
  }
  void eorReg(int rd, int rn, int rm, bool sf = true) {
    word((sf ? 0xCA000000u : 0x4A000000u) | (uint32_t(rm) << 16) | (uint32_t(rn) << 5) |
         uint32_t(rd));
  }
  void madd(int rd, int rn, int rm, int ra) {
    // MADD xd, xn, xm, xa   (64-bit)
    word(0x9B000000u | (uint32_t(rm) << 16) | (uint32_t(ra) << 10) | (uint32_t(rn) << 5) |
         uint32_t(rd));
  }
  void sdiv(int rd, int rn, int rm, bool sf = true) {
    // SDIV (verified against Capstone): opc 0x0C in the 2-source group.
    word((sf ? 0x9AC00C00u : 0x1AC00C00u) | (uint32_t(rm) << 16) | (uint32_t(rn) << 5) |
         uint32_t(rd));
  }
  void udiv(int rd, int rn, int rm, bool sf = true) {
    word((sf ? 0x9AC00800u : 0x1AC00800u) | (uint32_t(rm) << 16) | (uint32_t(rn) << 5) |
         uint32_t(rd));
  }
  void msub(int rd, int rn, int rm, int ra) {
    word(0x9B008000u | (uint32_t(rm) << 16) | (uint32_t(ra) << 10) | (uint32_t(rn) << 5) |
         uint32_t(rd));
  }
  void rev16w(int rd, int rn) { word(0x5AC00400u | (uint32_t(rn) << 5) | uint32_t(rd)); }
  void rev32w(int rd, int rn) { word(0x5AC00800u | (uint32_t(rn) << 5) | uint32_t(rd)); }
  void revx(int rd, int rn) { word(0xDAC00C00u | (uint32_t(rn) << 5) | uint32_t(rd)); }
  void lslImm(int rd, int rn, int imm, bool sf = true) {
    const int width = sf ? 64 : 32;
    // UBFM xd, xn, #(-imm) mod width, #(width-1-imm)
    const int immr = (width - imm) % width;  // LSL #imm == UBFM #(-imm), #(width-1-imm)
    const int imms = width - 1 - imm;
    // LSL is UBFM with immr == imms == (width - shift). Shift-by-zero is the
    // caller's job (movReg); UBFM with #0 would encode as a plain ORR here.
    word((sf ? 0xD3400000u : 0x53400000u) | (uint32_t(immr) << 16) | (uint32_t(imms) << 10) |
         (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void fMovDD(int rd, int rn) { word(0x1E604000u | (uint32_t(rn) << 5) | uint32_t(rd)); }
  void lsrImm(int rd, int rn, int imm, bool sf = true) {
    const int width = sf ? 64 : 32;
    // UBFM xd, xn, #imm, #(width-1)
    word((sf ? 0xD3400000u : 0x53400000u) | (uint32_t(imm) << 16) |
         (uint32_t(width - 1) << 10) | (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void asrImm(int rd, int rn, int imm, bool sf = true) {
    const int width = sf ? 64 : 32;
    // SBFM xd, xn, #imm, #(width-1)
    word((sf ? 0x93400000u : 0x13400000u) | (uint32_t(imm) << 16) |
         (uint32_t(width - 1) << 10) | (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void ldur(int rt, int rn, uint32_t off, int size) {
    // LDR (unsigned offset). size is the access width in bytes.
    switch (size) {
      case 1: word(0x39400000u | (off << 10) | (uint32_t(rn) << 5) | uint32_t(rt)); break;
      case 2: word(0x79400000u | ((off >> 1) << 10) | (uint32_t(rn) << 5) | uint32_t(rt)); break;
      case 4: word(0xB9400000u | ((off >> 2) << 10) | (uint32_t(rn) << 5) | uint32_t(rt)); break;
      default: word(0xF9400000u | ((off >> 3) << 10) | (uint32_t(rn) << 5) | uint32_t(rt)); break;
    }
  }
  void stur(int rt, int rn, uint32_t off, int size) {
    switch (size) {
      case 1: word(0x39000000u | (off << 10) | (uint32_t(rn) << 5) | uint32_t(rt)); break;
      case 2: word(0x79000000u | ((off >> 1) << 10) | (uint32_t(rn) << 5) | uint32_t(rt)); break;
      case 4: word(0xB9000000u | ((off >> 2) << 10) | (uint32_t(rn) << 5) | uint32_t(rt)); break;
      default: word(0xF9000000u | ((off >> 3) << 10) | (uint32_t(rn) << 5) | uint32_t(rt)); break;
    }
  }
  // The context holds the FPR file as doubles, so an FPR lives at kFpr0+8*n. Rather
  // than rely on the FP load/store immediate form (imm7 field, not the imm12 of the
  // integer form - easy to get wrong), move through an integer register, which uses
  // the LDR/STR encoding already verified above.
  void fLoadD(int rt, int rn, uint32_t off, int tmp) {
    ldur(tmp, rn, off, 8);
    fMovDX(rt, tmp);  // FMOV Dv, Xn
  }
  void fStoreD(int rt, int rn, uint32_t off, int tmp) {
    fMovXD(tmp, rt);  // FMOV Xn, Dv
    stur(tmp, rn, off, 8);
  }
  void fDadd(int rd, int rn, int rm) {
    word(0x1E602800u | (uint32_t(rm) << 16) | (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void fDsub(int rd, int rn, int rm) {
    word(0x1E603800u | (uint32_t(rm) << 16) | (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void fDmul(int rd, int rn, int rm) {
    word(0x1E600800u | (uint32_t(rm) << 16) | (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void fDdiv(int rd, int rn, int rm) {
    word(0x1E601800u | (uint32_t(rm) << 16) | (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void fDneg(int rd, int rn) { word(0x1E614000u | (uint32_t(rn) << 5) | uint32_t(rd)); }
  void fDabs(int rd, int rn) { word(0x1E60C000u | (uint32_t(rn) << 5) | uint32_t(rd)); }
  void fMaddD(int rd, int rn, int rm, int ra) {
    // FMADD Dd, Dn, Dm, Da - ftype bits 23:22 = 11 selects double (01 would be
    // single, which is what the earlier draft silently emitted).
    word(0x1F400000u | (uint32_t(rm) << 16) | (uint32_t(ra) << 10) | (uint32_t(rn) << 5) |
         uint32_t(rd));
  }
  void fMovXD(int rd, int rn) { word(0x9E660000u | (uint32_t(rn) << 5) | uint32_t(rd)); }
  void fMovDX(int rd, int rn) { word(0x9E670000u | (uint32_t(rn) << 5) | uint32_t(rd)); }
  void fCmpD(int rn, int rm) { word(0x1E602000u | (uint32_t(rm) << 16) | (uint32_t(rn) << 5)); }
  void fCvtZD(int rd, int rn) {
    // FCVTZS Xd, Dn  (round toward zero, 64-bit result)
    word(0x9E780000u | (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void sxtw(int rd, int rn) {
    // SBFM Xd, Xn, #32, #31  == SXTW
    word(0x93407C00u | (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void sxth(int rd, int rn) {
    // SXTH Xd, Xn == SBFM Xd, Xn, #48, #15
    word(0x93400000u | (48u << 16) | (15u << 10) | (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void sxtb(int rd, int rn) {
    // SXTB Xd, Xn == SBFM Xd, Xn, #56, #7
    word(0x93400000u | (56u << 16) | (7u << 10) | (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void cmpReg(int rn, int rm, bool sf = true) {
    // SUBS XZR, Xn, Xm
    word((sf ? 0xEB00001Fu : 0x6B00001Fu) | (uint32_t(rm) << 16) | (uint32_t(rn) << 5));
  }
  void cmpImm(int rn, int imm9, bool sf = true) {
    const uint32_t sh = imm9 < 0 ? (1u << 22) : 0u;
    const uint32_t v = uint32_t(imm9 < 0 ? -imm9 : imm9);
    word((sf ? 0xF100001Fu : 0x7100001Fu) | sh | ((v & 0x1FFu) << 10) | (uint32_t(rn) << 5));
  }
  void csel(int rd, int rn, int rm, uint32_t cond, bool sf = true) {
    word((sf ? 0x9A800000u : 0x1A800000u) | (uint32_t(rm) << 16) | (uint32_t(cond) << 12) |
         (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void movReg(int rd, int rn) {
    // ORR Xd, XZR, Xn
    word(0xAA1F0000u | (uint32_t(rn) << 5) | uint32_t(rd));
  }
  void b(size_t l) {
    fixups_.push_back({code_.size(), l, 0, 0});
    word(0x14000000u);
  }
  void bcond(uint32_t cond, size_t l) {
    fixups_.push_back({code_.size(), l, 1, cond});
    word(0x54000000u | cond);
  }
  void cbnz(int rt, bool sf, size_t l) {
    fixups_.push_back({code_.size(), l, 3, sf ? 1u : 0u});
    word((sf ? 0xB5000000u : 0x35000000u) | (uint32_t(rt) & 31u));
  }
  void cbz(int rt, bool sf, size_t l) {
    fixups_.push_back({code_.size(), l, 2, 0});
    word((sf ? 0xB4000000u : 0x34000000u) | (uint32_t(rt) & 31u));
  }
  void dmbIsh() { word(0xD5033BBFu); }  // DMB ISH
  void movWz(int rd, uint32_t v) {       // MOVZ Wd, #imm
    word(0x52800000u | ((v & 0xFFFFu) << 5) | uint32_t(rd));
  }
  void ret() { word(0xD65F03C0u); }       // RET (X30 = link register of the trampoline)

  bool link(uint32_t* out, size_t maxInstrs, size_t* outBytes) {
    // Resolve forward branches. Labels are emitted before use, so a label that was
    // never bound (index past the end) is a bug -> fail closed.
    for (const Fixup& f : fixups_) {
      if (f.target >= labels_.size()) return false;
      const int64_t delta = int64_t(labels_[f.target]) - int64_t(f.at);
      if (f.kind == 0) {
        if (delta < -0x02000000 || delta > 0x01FFFFFF) return false;
        code_[f.at] = 0x14000000u | (uint32_t(delta) & 0x03FFFFFFu);
      } else if (f.kind == 1) {
        if (delta < -0x00040000 || delta > 0x0003FFFF) return false;
        code_[f.at] = 0x54000000u | ((uint32_t(delta) & 0x7FFFFu) << 5) | f.aux;
      } else {
        if (delta < -0x00040000 || delta > 0x0003FFFF) return false;
        code_[f.at] |= (uint32_t(delta) & 0x7FFFFu) << 5;
      }
    }
    if (m_bad || code_.size() > maxInstrs) return false;
    for (size_t i = 0; i < code_.size(); i++) out[i] = code_[i];
    *outBytes = code_.size() * 4;
    return true;
  }

 private:
  struct Fixup {
    size_t at;
    size_t target;
    int kind;  // 0 = B, 1 = B.cond, 2 = CBZ
    uint32_t aux;
  };
  bool m_bad = false;
  std::vector<uint32_t> code_;
  std::vector<Fixup> fixups_;
  std::vector<size_t> labels_;
};

// ─── Block builder ────────────────────────────────────────────────────────────
class BlockBuilder {
 public:
  BlockBuilder(uint64_t guestPc, uint32_t* out, size_t maxInstrs, MemCtx& mem)
      : m_pc(guestPc), m_out(out), m_max(maxInstrs), m_mem(mem) {}

  // Returns the number of instructions emitted, or 0 if nothing could be
  // translated (the block then runs interpreted).
  bool build(uint32_t* outInstrs, uint32_t* outGuestBytes) {
    prologue();
    uint64_t pc = m_pc;
    const uint64_t limit = m_mem.guestBase + m_mem.guestSize;
    for (int n = 0; n < 48; n++) {
      if (pc + 4 > limit) break;
      uint32_t word;
      memcpy(&word, m_mem.guestMemory + (pc - m_mem.guestBase), 4);
      word = __builtin_bswap32(word);  // guest code is big-endian
      const DecodedPpc di = ppcDecode(word, pc);
      if (!supported(di)) break;
      if (!emit(di)) break;
      m_instrs++;
      pc += 4;
      if (di.term || m_stop) break;  // a branch ends the block
    }
    tail(pc);
    if (m_instrs == 0) return false;
    size_t bytes = 0;
    if (!m_a.link(m_out, m_max, &bytes)) return false;
    *outInstrs = m_instrs;
    *outGuestBytes = m_instrs * 4;
    m_bytes = bytes;
    return true;
  }
  size_t bytes() const { return m_bytes; }

 private:
  bool supported(const DecodedPpc& di) const {
    if (!backendSupports(di.cls)) return false;
    // Record-form and carry/overflow forms are interpreter-only: XER/CR0 rules
    // are where subtle divergence hides, and the gain from JIT'ing them is small.
    if (di.has(PpcFlag::kFlagRc) || di.has(PpcFlag::kFlagOvf) || di.has(PpcFlag::kFlagCarry)) {
      if (di.cls != PpcClass::FpCmp) return false;
    }
    if (di.cls == PpcClass::Ld || di.cls == PpcClass::St) {
      if (di.has(PpcFlag::kFlagUpdate) || di.has(PpcFlag::kFlagSwap) ||
          di.has(PpcFlag::kFlagIntStore) || di.has(PpcFlag::kFlagIndex)) {
        return false;
      }
      const uint32_t size = uint32_t(di.mb);
      if (size != 1 && size != 2 && size != 4 && size != 8) return false;
      // The immediate displacement must fit the ARM64 addressing mode, or be
      // buildable with one ADD #imm12 (+ LSL #12 is not needed).
      const int64_t off = di.imm;
      if (off < 0 || off > 4095) return false;
    }
    if (di.cls == PpcClass::Add || di.cls == PpcClass::Subf) {
      // plain add/sub only (no carry-in/out, no OE, no record)
      if (di.imm != 0) return false;
    }
    if (di.cls == PpcClass::Rl) {
      if (di.imm != kRlRotateMasked) return false;
      if (di.has(PpcFlag::kFlagInsert)) return false;
    }
    if (di.cls == PpcClass::Mfspr || di.cls == PpcClass::Mtspr) {
      const uint32_t spr = uint32_t(di.imm) & 0x3FF;
      if (spr != 8 && spr != 9 && spr != 1) return false;  // LR / CTR / XER
    }
    if (di.cls == PpcClass::FpArith) {
      switch (int(di.imm)) {
        case kFpAdd: case kFpSub: case kFpMul: case kFpDiv: case kFpMadd:
          break;
        default: return false;  // fsqrt: no verified encoding on this host
      }
      if (di.has(PpcFlag::kFlagSingle)) return false;  // single-precision rounding
    }
    return true;
  }

  void prologue() {
    m_a.movReg(XCTX, 0);  // x0 = PPCContext* (AAPCS64 first argument)
    m_a.ldur(XHST, XCTX, kHostMem, 8);
    m_lblError = m_a.label();
    m_lblTail = m_a.label();
    m_lblTailReg = m_a.label();
    m_lblExitContinue = m_a.label();
  }

  // Block tail. Two entries share the epilogue:
  //   m_lblTailReg  - taken branches arrive with the new guest PC in XB
  //   m_lblTail     - fall-through: the linear next PC is stored here
  // Returning kExitContinue hands control to the dispatcher, which recognises HLE
  // thunks and out-of-window PCs (see JitEngine::threadLoop) - so the generated
  // code never has to know what an import sentinel looks like.
  void tail(uint64_t nextPc) {
    m_a.bind(m_lblTailReg);
    m_a.stur(XB, XCTX, kPc, 8);
    m_a.b(m_lblExitContinue);
    m_a.bind(m_lblTail);
    m_a.movConst(XB, nextPc);
    m_a.stur(XB, XCTX, kPc, 8);
    m_a.bind(m_lblExitContinue);
    m_a.movWz(0, kExitContinue);
    m_a.ret();
    m_a.bind(m_lblError);
    m_a.movWz(0, kExitError);  // out-of-window data access: the interpreter re-runs
    m_a.ret();                    // this block and reports the precise instruction
  }

  uint32_t m_addrOff = 0;
  // XA = host address of the guest effective address (bounds checked). The
  // displacement is folded into the LDR/STR immediate when it fits (m_addrOff),
  // otherwise it is added here and the access uses offset 0.
  void emitAddress(const DecodedPpc& di) {
    const int64_t off = di.imm;
    if (di.ra == 0) {
      m_a.movConst(XA, uint64_t(uint32_t(off)));
      m_addrOff = 0;
    } else if (off >= 0 && off <= 4095) {
      m_a.ldur(XA, XCTX, kGpr0 + uint32_t(di.ra) * 8, 8);
      m_addrOff = uint32_t(off);
    } else {
      m_a.ldur(XA, XCTX, kGpr0 + uint32_t(di.ra) * 8, 8);
      m_a.movConst(XCT, uint64_t(off));
      m_a.addReg(XA, XA, XCT);
      m_addrOff = 0;
    }
    // Guest addresses are 32-bit: isolate the low word (LSL#32,LSR#32 - the order
    // matters; reversing them would keep the *high* half).
    m_a.lslImm(XA, XA, 32, true);
    m_a.lsrImm(XA, XA, 32, true);
    // Bounds: XA < guestSize. Both operands are 32-bit now, so the compare cannot
    // wrap and an out-of-window access branches to the error exit.
    m_a.ldur(XC, XCTX, kGuestSize, 8);
    m_a.cmpReg(XA, XC, false);
    m_a.bcond(kHs, m_lblError);
    m_a.addReg(XA, XA, XHST);  // host = guest + hostMem (guestBase == 0 enforced)
  }

  // Per-instruction dispatch. Returning false means "this instruction cannot be
  // translated": the block simply ends there and the interpreter runs it, which is
  // always safe (the tail has already been emitted for the preceding instructions).
  bool emit(const DecodedPpc& di) {
    switch (di.cls) {
      case PpcClass::Nop:
        return true;  // the pc is written once, in the block tail
      case PpcClass::Add:
      case PpcClass::Subf:
      case PpcClass::Logic:
      case PpcClass::LogicImm:
      case PpcClass::Mul:
      case PpcClass::Div:
      case PpcClass::Cntlz:
      case PpcClass::Extend:
      case PpcClass::Rl:
        return emitInt(di);
      case PpcClass::Ld:
      case PpcClass::St:
        return emitMem(di);
      case PpcClass::Cmp:
        return emitCmp(di);
      case PpcClass::Mfspr:
      case PpcClass::Mtspr: {
        const uint32_t spr = uint32_t(di.imm) & 0x3FF;
        const uint32_t off = spr == 8 ? kLr : (spr == 9 ? kCtr : kXer);
        const int size = spr == 1 ? 4 : 8;  // ctx.xer is 32-bit, ctr/lr are 64-bit
        if (di.cls == PpcClass::Mfspr) {
          m_a.ldur(XB, XCTX, off, size);  // LDR W zero-extends: what mfxer wants
          m_a.stur(XB, XCTX, kGpr0 + uint32_t(di.rd) * 8, 8);
        } else {
          m_a.ldur(XB, XCTX, kGpr0 + uint32_t(di.rs) * 8, 8);
          m_a.stur(XB, XCTX, off, size);
          m_stop = true;  // mtlr/mtctr change where a branch lands: stop the block
        }
        return true;
      }
      case PpcClass::Sync:
        m_a.dmbIsh();
        return true;
      case PpcClass::B:
      case PpcClass::Bc:
      case PpcClass::Bclr:
        return emitBranch(di);
      case PpcClass::FpArith:
      case PpcClass::FpCmp:
      case PpcClass::FpUnary:
        return emitFp(di);
      default:
        return false;
    }
  }

  // Load the second source operand into XC (register or materialised immediate)
  // and return the register holding it.
  int secondOperand(const DecodedPpc& di) {
    if (!di.has(PpcFlag::kFlagImm)) {
      m_a.ldur(XC, XCTX, kGpr0 + uint32_t(di.rb) * 8, 8);
      return XC;
    }
    m_a.movConst(XC, uint64_t(di.imm));
    return XC;
  }

  bool emitInt(const DecodedPpc& di) {
    const int rd = di.cls == PpcClass::Rl || di.cls == PpcClass::Logic ||
                           di.cls == PpcClass::Cntlz || di.cls == PpcClass::Extend
                       ? di.rd
                       : di.rd;
    m_a.ldur(XB, XCTX, kGpr0 + uint32_t(di.ra) * 8, 8);
    switch (di.cls) {
      case PpcClass::Add: {
        const int r = secondOperand(di);
        m_a.addReg(XCT, XB, r);
        m_a.stur(XCT, XCTX, kGpr0 + uint32_t(rd) * 8, 8);
        return true;
      }
      case PpcClass::Subf: {
        const int r = secondOperand(di);
        m_a.subReg(XCT, r, XB);  // subf: rB - rA
        m_a.stur(XCT, XCTX, kGpr0 + uint32_t(rd) * 8, 8);
        return true;
      }
      case PpcClass::Logic:
      case PpcClass::LogicImm: {
        const int sel = int(di.cls == PpcClass::LogicImm ? di.rc : di.imm);
        const int r = secondOperand(di);
        switch (sel) {
          case kLogicAnd: m_a.andReg(XCT, XB, r); break;
          case kLogicOr: m_a.orReg(XCT, XB, r); break;
          case kLogicXor: m_a.eorReg(XCT, XB, r); break;
          case kLogicNand: m_a.andReg(XCT, XB, r); m_a.movConst(XD, ~0ull); m_a.eorReg(XCT, XCT, XD); break;
          case kLogicNor: m_a.orReg(XCT, XB, r); m_a.movConst(XD, ~0ull); m_a.eorReg(XCT, XCT, XD); break;
          case kLogicEqv: m_a.eorReg(XCT, XB, r); m_a.movConst(XD, ~0ull); m_a.eorReg(XCT, XCT, XD); break;
          case kLogicAndc: m_a.movConst(XD, ~0ull); m_a.eorReg(XCT, r, XD); m_a.andReg(XCT, XB, XCT); break;
          case kLogicOrc: m_a.movConst(XD, ~0ull); m_a.eorReg(XCT, r, XD); m_a.orReg(XCT, XB, XCT); break;
          default: return false;
        }
        if (di.has(PpcFlag::kFlagZero32)) m_a.lsrImm(XCT, XCT, 32, true), m_a.lslImm(XCT, XCT, 32, true);
        m_a.stur(XCT, XCTX, kGpr0 + uint32_t(rd) * 8, 8);
        return true;
      }
      case PpcClass::Mul: {
        if (di.has(PpcFlag::kFlagHigh)) return false;  // mulh*: interpreter (64x64
                                                       // high half needs MADD chains)
        const int r = secondOperand(di);
        m_a.madd(XCT, XB, r, 31);
        m_a.stur(XCT, XCTX, kGpr0 + uint32_t(rd) * 8, 8);
        return true;
      }
      case PpcClass::Div: {
        if (di.mb < 8) return false;  // 32-bit forms need sign handling: interpreter
        const int r = secondOperand(di);
        m_a.ldur(XB, XCTX, kGpr0 + uint32_t(di.ra) * 8, 8);
        if (di.has(PpcFlag::kFlagSigned)) m_a.sdiv(XCT, XB, r);
        else m_a.udiv(XCT, XB, r);
        m_a.stur(XCT, XCTX, kGpr0 + uint32_t(rd) * 8, 8);
        return true;
      }
      case PpcClass::Cntlz:
        return false;  // CLZ is not in the verified encoding set: interpreter
      case PpcClass::Extend: {
        if (di.mb == 1) m_a.sxtb(XCT, XB);
        else if (di.mb == 2) m_a.sxth(XCT, XB);
        else m_a.sxtw(XCT, XB);
        m_a.stur(XCT, XCTX, kGpr0 + uint32_t(rd) * 8, 8);
        return true;
      }
      case PpcClass::Rl: {
        const int width = di.rc == 64 ? 64 : 32;
        const uint64_t mask = ppc_mask(di.mb, di.me, width);
        const int sh = di.sh % width;
        if (sh == 0) {
          m_a.movReg(XCT, XB);
        } else {
          m_a.lslImm(XCT, XB, sh, width == 64);
          m_a.lsrImm(XD, XB, width - sh, width == 64);
          m_a.orReg(XCT, XCT, XD);
        }
        m_a.movConst(XD, mask);
        m_a.andReg(XCT, XCT, XD);
        m_a.stur(XCT, XCTX, kGpr0 + uint32_t(rd) * 8, 8);
        return true;
      }
      default:
        return false;
    }
  }

  bool emitMem(const DecodedPpc& di) {
    const uint32_t size = uint32_t(di.mb);
    emitAddress(di);
    if (di.cls == PpcClass::Ld) {
      if (di.has(PpcFlag::kFlagSingle)) {
        // lfs / lfsu: the single->double conversion and the byte order are done by
        // the interpreter (exact); only the double form is translated here.
        if (size != 8) return false;
        m_a.ldur(XB, XA, m_addrOff, 8);
        m_a.revx(XB, XB);
        m_a.fMovXD(XB, XB);
        m_a.fStoreD(XB, XCTX, kFpr0 + uint32_t(di.rd) * 8, XD);
        return true;
      }
      m_a.ldur(XB, XA, m_addrOff, int(size));
      if (size == 2 && di.has(PpcFlag::kFlagSigned)) m_a.sxth(XB, XB);
      if (size == 1 && di.has(PpcFlag::kFlagSigned)) m_a.sxtb(XB, XB);
      if (size == 2) m_a.rev16w(XB, XB);
      if (size == 4) m_a.rev32w(XB, XB);
      if (size == 8) m_a.revx(XB, XB);
      m_a.stur(XB, XCTX, kGpr0 + uint32_t(di.rd) * 8, 8);
      return true;
    }
    if (di.has(PpcFlag::kFlagSingle)) {
      if (size != 8) return false;  // stfs -> interpreter
      m_a.fLoadD(XB, XCTX, kFpr0 + uint32_t(di.rs) * 8, XD);
      m_a.fMovDX(XB, XB);
      m_a.revx(XB, XB);
      m_a.stur(XB, XA, m_addrOff, 8);
      return true;
    }
    m_a.ldur(XB, XCTX, kGpr0 + uint32_t(di.rs) * 8, 8);
    if (size == 2) m_a.rev16w(XB, XB);
    if (size == 4) m_a.rev32w(XB, XB);
    if (size == 8) m_a.revx(XB, XB);
    m_a.stur(XB, XA, m_addrOff, int(size));
    return true;
  }

  // Write CR0..CR7 field `crf` from the flags left by a compare. The guest field
  // layout is LT=8 GT=4 EQ=2 SO=1 and CR is stored big-endian-bitwise in a uint32_t
  // (field 0 in the most significant nibble), so field `crf` sits at bit 28-4*crf.
  void storeCrBits(int crf, uint32_t ltCond, uint32_t gtCond, bool soFromXer) {
    m_a.movWz(XD, 0);
    m_a.movConst(XCT, 2);
    m_a.csel(XD, XD, XCT, kEq, false);
    m_a.movConst(XCT, 4);
    m_a.csel(XD, XD, XCT, gtCond, false);
    m_a.movConst(XCT, 8);
    m_a.csel(XD, XD, XCT, ltCond, false);
    const int sh = 28 - crf * 4;
    m_a.ldur(XB, XCTX, kCr, 4);
    if (sh) m_a.lslImm(XD, XD, sh, true);
    m_a.movConst(XC, ~uint64_t(0xF) << sh);
    m_a.andReg(XB, XB, XC, false);
    m_a.orReg(XB, XB, XD, false);
    if (soFromXer) {
      // CR0[SO] <- XER[SO] for record forms. The plain compares emitted here never
      // set XER, so this is the "carry the old value" case and needs no work.
      (void)soFromXer;
    }
    m_a.stur(XB, XCTX, kCr, 4);
  }

  bool emitCmp(const DecodedPpc& di) {
    // di.imm carries the CmpSel from the decoder row (kCmpW/kCmplW/kCmpD/kCmplD);
    // for the D-form cmpwi/cmplwi the width comes from the table's `size` column.
    const int sel = int(di.imm);
    const bool uns = sel == kCmplW || sel == kCmplD || di.has(PpcFlag::kFlagHigh);
    const bool is64 = sel == kCmpD || sel == kCmplD || di.mb == 8;
    m_a.ldur(XB, XCTX, kGpr0 + uint32_t(di.ra) * 8, 8);
    const int r = secondOperand(di);
    m_a.movReg(XC, r);
    if (!is64) {
      // cmpw/cmplw compare the *low* 32 bits: sign- or zero-extend them so a 64-bit
      // ARM compare yields the PowerPC flags.
      if (uns) {
        m_a.lslImm(XB, XB, 32, true);
        m_a.lsrImm(XB, XB, 32, true);
        m_a.lslImm(XC, XC, 32, true);
        m_a.lsrImm(XC, XC, 32, true);
      } else {
        m_a.sxtw(XB, XB);
        m_a.sxtw(XC, XC);
      }
    }
    m_a.cmpReg(XB, XC, true);
    storeCrBits(di.crf, uns ? kLo : kLt, uns ? kHi : kGt, false);
    return true;
  }

  bool emitBranch(const DecodedPpc& di) {
    if (di.cls == PpcClass::B) {
      m_a.movConst(XB, uint64_t(di.imm));
      m_a.b(m_lblTailReg);
      m_stop = true;
      return true;
    }
    if (di.cls == PpcClass::Bclr) {
      if (!di.has(PpcFlag::kFlagCount)) return false;  // bctr: interpreter
      m_a.ldur(XB, XCTX, kCtr, 8);
      m_a.b(m_lblTailReg);
      m_stop = true;
      return true;
    }
    // Conditional branch. Only the four shapes real compilers emit are translated:
    //   bo=0x04 bdnz  (decrement CTR, branch if != 0)
    //   bo=0x0C bdz   (decrement CTR, branch if == 0)
    //   bo=0x0A       (branch if CR[bi] set)
    //   bo=0x02       (branch if CR[bi] clear)
    // Anything else ends the block so the interpreter takes it.
    const uint32_t bo = uint32_t(di.rd) & 0x1Fu;
    const uint32_t bi = uint32_t(di.ra) & 31u;
    const bool link = di.has(PpcFlag::kFlagLink);
    const uint64_t target = uint64_t(di.imm);
    bool ctrForm = false;
    if (bo == 0x04 || bo == 0x0C) ctrForm = true;
    else if (bo != 0x02 && bo != 0x0A) return false;
    if (link) {
      m_a.movConst(XB, di.pc + 4);
      m_a.stur(XB, XCTX, kLr, 8);
    }
    if (ctrForm) {
      m_a.ldur(XC, XCTX, kCtr, 8);
      m_a.subImm(XC, XC, 1);
      m_a.stur(XC, XCTX, kCtr, 8);
      const size_t skip = m_a.label();
      if (bo == 0x04) m_a.cbnz(XC, true, skip);
      else m_a.cbz(XC, true, skip);
      m_a.b(m_lblTail);          // not taken -> linear continuation
      m_a.bind(skip);
    } else {
      // WB = the selected CR bit (0/1); test it with CBZ/CBNZ.
      m_a.ldur(XC, XCTX, kCr, 4);
      const int sh = 31 - int(bi);
      if (sh) m_a.lsrImm(XC, XC, sh, false);
      m_a.movConst(XD, 1);
      m_a.andReg(XC, XC, XD, false);
      const size_t skip = m_a.label();
      if (bo == 0x0A) m_a.cbnz(XC, false, skip);
      else m_a.cbz(XC, false, skip);
      m_a.b(m_lblTail);
      m_a.bind(skip);
    }
    m_a.movConst(XB, target);
    m_a.b(m_lblTailReg);
    m_stop = true;
    return true;
  }

  bool emitFp(const DecodedPpc& di) {
    const uint32_t rdOff = kFpr0 + uint32_t(di.rd) * 8;
    m_a.fLoadD(XB, XCTX, kFpr0 + uint32_t(di.ra) * 8, XD);
    m_a.fLoadD(XC, XCTX, kFpr0 + uint32_t(di.rb) * 8, XI1);
    if (di.cls == PpcClass::FpCmp) {
      // fcmpu/fcmpo: FP compare flags are Lt=1,Gt=2,Eq=4 in CR1..7 (XER[SO] on fcmpo
      // is not modelled here, so only fcmpu is translated; fcmpo stays interpreted).
      if (di.has(PpcFlag::kFlagOrdered)) return false;
      m_a.fCmpD(XB, XC);
      storeCrBits(di.crf, kMi, kGt, false);
      return true;
    }
    if (di.cls == PpcClass::FpUnary) {
      switch (int(di.imm)) {
        case kFpAbs: m_a.fDabs(XCT, XC); break;
        case kFpNeg: m_a.fDneg(XCT, XC); break;
        case kFpMr: m_a.fMovDD(XCT, XC); break;
        case kFpFnabs: m_a.fDabs(XCT, XC); m_a.fDneg(XCT, XCT); break;
        default: return false;
      }
      m_a.fStoreD(XCT, XCTX, rdOff, XD);
      return true;
    }
    switch (int(di.imm)) {
      case kFpAdd: m_a.fDadd(XCT, XB, XC); break;
      case kFpSub: m_a.fDsub(XCT, XB, XC); break;
      case kFpMul: m_a.fDmul(XCT, XB, XC); break;
      case kFpDiv: m_a.fDdiv(XCT, XB, XC); break;
      case kFpMadd: {
        // fmadd/fmsub/fnmadd/fnmsub = rA*rC (+/-) rB, optionally negated product.
        const uint32_t rcOff = kFpr0 + uint32_t(di.rc) * 8;
        m_a.fLoadD(XD, XCTX, rcOff, XA);
        m_a.fMaddD(XCT, XB, XD, 31);
        if (di.has(PpcFlag::kFlagNeg)) m_a.fDneg(XCT, XCT);
        if (di.has(PpcFlag::kFlagSub)) m_a.fDsub(XCT, XCT, XC);
        else m_a.fDadd(XCT, XCT, XC);
        break;
      }
      default:
        return false;
    }
    m_a.fStoreD(XCT, XCTX, rdOff, XD);
    return true;
  }

  uint64_t m_pc;
  uint32_t* m_out;
  size_t m_max;
  MemCtx& m_mem;
  Asm m_a;
  uint32_t m_instrs = 0;
  size_t m_bytes = 0;
  bool m_stop = false;
  size_t m_lblError = 0;
  size_t m_lblTail = 0;
  size_t m_lblTailReg = 0;
  size_t m_lblExitContinue = 0;
};

}  // namespace detail

// The encoder is exercised word by word by tests/arm64_asm_probe.cpp, which is why
// `detail` is a named namespace rather than an anonymous one.
using namespace detail;

// ─── Backend capability gate ──────────────────────────────────────────────────
bool backendSupports(PpcClass cls) {
  if (!kHostIsArm64) return false;
  switch (cls) {
    case PpcClass::Nop:
    case PpcClass::Add:
    case PpcClass::Subf:
    case PpcClass::Mul:
    case PpcClass::Div:
    case PpcClass::Logic:
    case PpcClass::LogicImm:
    case PpcClass::Cntlz:
    case PpcClass::Extend:
    case PpcClass::Rl:
    case PpcClass::Cmp:
    case PpcClass::Ld:
    case PpcClass::St:
    case PpcClass::Mfspr:
    case PpcClass::Mtspr:
    case PpcClass::Sync:
    case PpcClass::B:
    case PpcClass::Bc:
    case PpcClass::Bclr:
    case PpcClass::FpArith:
    case PpcClass::FpCmp:
    case PpcClass::FpUnary:
      return true;
    default:
      return false;
  }
}

size_t jitCompileBlock(uint64_t guestPc, uint8_t* out, size_t maxBytes, MemCtx& mem,
                       uint32_t* outInstrCount, uint32_t* outGuestBytes) {
  *outInstrCount = 0;
  *outGuestBytes = 0;
  if (!kHostIsArm64) return 0;
  // The addressing scheme folds guestBase into `hostMem` (see emitAddress), so it
  // requires the runtime's flat mapping. The engine maps guest 0 at hostMem, which
  // is what jit::JitEngine::init does; anything else must stay interpreted.
  if (mem.guestBase != 0) {
    X360LOGW(TAG, "guest base 0x%llX is not 0: keeping this block interpreted",
             (unsigned long long)mem.guestBase);
    return 0;
  }
  if (!mem.guestMemory || mem.guestSize == 0 || maxBytes < 64) return 0;
  const size_t maxInstrs = (maxBytes - 16) / 4;
  BlockBuilder b(guestPc, reinterpret_cast<uint32_t*>(out) + 4, maxInstrs, mem);
  uint32_t instrs = 0, guestBytes = 0;
  if (!b.build(&instrs, &guestBytes)) return 0;
  if (instrs == 0) return 0;
  const size_t codeBytes = b.bytes();
  // Prefix (at out[0..15]): u32 instruction count, u32 code size, 8 bytes pad, so
  // the entry point at out+16 stays 16-byte aligned.
  uint32_t prefix[4] = {instrs, uint32_t(codeBytes), 0, 0};
  memcpy(out, prefix, 16);
  *outInstrCount = instrs;
  *outGuestBytes = guestBytes;
  return 16 + codeBytes;
}

}  // namespace jit
}  // namespace x360
