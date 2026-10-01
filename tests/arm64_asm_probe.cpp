// Host probe for the ARM64 encoder in arm64_backend.cpp.
//
// Prints "<hex> <mnemonic>" for every encoding the backend emits, so
// tools/check_arm64_asm.py can disassemble the words with Capstone and show what a
// real decoder thinks they mean. An emulator that emits a wrong encoding is worse
// than one that emits none, so this file is the gate for the encodings.
//
// Build (host):
//   g++ -std=c++20 -O0 -I native/include -o /tmp/arm64_probe \
//       tests/arm64_asm_probe.cpp -x c++ native/src/jit/arm64_backend.cpp
#include "../native/src/jit/arm64_backend.cpp"

#include <cstdio>

using namespace x360::jit;
using namespace x360::jit::detail;

namespace {
struct Probe {
  Asm a;
  void show(const char* text) {
    uint32_t buf[64];
    size_t bytes = 0;
    if (!a.link(buf, 64, &bytes)) {
      printf("LINKFAIL %s\n", text);
      return;
    }
    for (size_t i = 0; i < bytes / 4; i++) printf("%08X ; %s\n", buf[i], text);
  }
};
}  // namespace

int main() {
  { Probe p; p.a.movz(11, 0x1234, 0); p.show("mov x11, #0x1234"); }
  { Probe p; p.a.movz(11, 0x0, 0); p.a.movk(11, 0xabcd, 1); p.show("mov #0xabcd0000"); }
  { Probe p; p.a.movConst(16, 0x8FFE0000ull); p.show("mov x16, #0x8ffe0000"); }
  { Probe p; p.a.addImm(10, 10, 0x3FF); p.show("add x10, x10, #0x3ff"); }
  { Probe p; p.a.subImm(10, 10, 1); p.show("sub x10, x10, #1"); }
  { Probe p; p.a.addReg(16, 11, 12); p.show("add x16, x11, x12"); }
  { Probe p; p.a.subReg(16, 12, 11); p.show("sub x16, x12, x11"); }
  { Probe p; p.a.andReg(16, 11, 12); p.show("and x16, x11, x12"); }
  { Probe p; p.a.orReg(16, 11, 12); p.show("orr x16, x11, x12"); }
  { Probe p; p.a.eorReg(16, 11, 12); p.show("eor x16, x11, x12"); }
  { Probe p; p.a.madd(16, 11, 12, 31); p.show("madd x16, x11, x12, xzr"); }
  { Probe p; p.a.msub(16, 11, 12, 13); p.show("msub x16, x11, x12, x13"); }
  { Probe p; p.a.sdiv(16, 11, 12); p.show("sdiv x16, x11, x12"); }
  { Probe p; p.a.udiv(16, 11, 12); p.show("udiv x16, x11, x12"); }
  { Probe p; p.a.sdiv(16, 11, 12, false); p.show("sdiv w16, w11, w12"); }
  { Probe p; p.a.rev16w(11, 11); p.show("rev16 w11, w11"); }
  { Probe p; p.a.rev32w(11, 11); p.show("rev32 w11, w11"); }
  { Probe p; p.a.revx(11, 11); p.show("rev x11, x11"); }
  { Probe p; p.a.lslImm(16, 11, 3); p.show("lsl x16, x11, #3"); }
  { Probe p; p.a.lsrImm(16, 11, 32); p.show("lsr x16, x11, #32"); }
  { Probe p; p.a.asrImm(16, 11, 3, false); p.show("asr w16, w11, #3"); }
  { Probe p; p.a.sxtw(16, 11); p.show("sxtw x16, w11"); }
  { Probe p; p.a.sxth(16, 11); p.show("sxth x16, w11"); }
  { Probe p; p.a.sxtb(16, 11); p.show("sxtb x16, w11"); }
  { Probe p; p.a.ldur(11, 9, 512, 8); p.show("ldr x11, [x9, #512]"); }
  { Probe p; p.a.stur(11, 9, 512, 8); p.show("str x11, [x9, #512]"); }
  { Probe p; p.a.ldur(11, 9, 4, 4); p.show("ldr w11, [x9, #4]"); }
  { Probe p; p.a.ldur(11, 9, 2, 2); p.show("ldrh w11, [x9, #2]"); }
  { Probe p; p.a.ldur(11, 9, 1, 1); p.show("ldrb w11, [x9, #1]"); }
  { Probe p; p.a.stur(11, 10, 8, 4); p.show("str w11, [x10, #8]"); }
  { Probe p; p.a.stur(11, 10, 8, 1); p.show("strb w11, [x10, #8]"); }
  { Probe p; p.a.fLoadD(11, 9, 256, 13); p.show("ldr x13,[x9,#256]; fmov d11, x13"); }
  { Probe p; p.a.fStoreD(11, 9, 256, 13); p.show("fmov x13, d11; str x13,[x9,#256]"); }
  { Probe p; p.a.fDadd(17, 18, 19); p.show("fadd d17, d18, d19"); }
  { Probe p; p.a.fDsub(17, 18, 19); p.show("fsub d17, d18, d19"); }
  { Probe p; p.a.fDmul(17, 18, 19); p.show("fmul d17, d18, d19"); }
  { Probe p; p.a.fDdiv(17, 18, 19); p.show("fdiv d17, d18, d19"); }
  { Probe p; p.a.fDneg(17, 18); p.show("fneg d17, d18"); }
  { Probe p; p.a.fDabs(17, 18); p.show("fabs d17, d18"); }
  { Probe p; p.a.fMovDD(17, 18); p.show("fmov d17, d18"); }
  { Probe p; p.a.fMaddD(17, 18, 19, 31); p.show("fmadd d17, d18, d19, d31"); }
  { Probe p; p.a.fMovXD(17, 18); p.show("fmov x17, d18"); }
  { Probe p; p.a.fMovDX(17, 18); p.show("fmov d17, x18"); }
  { Probe p; p.a.fCmpD(18, 19); p.show("fcmp d18, d19"); }
  { Probe p; p.a.fCvtZD(17, 18); p.show("fcvtzs x17, d18"); }
  { Probe p; p.a.cmpReg(10, 12); p.show("cmp x10, x12"); }
  { Probe p; p.a.cmpReg(10, 12, false); p.show("cmp w10, w12"); }
  { Probe p; p.a.cmpImm(12, 0); p.show("cmp x12, #0"); }
  { Probe p; p.a.csel(13, 11, 12, kLt); p.show("csel x13, x11, x12, lt"); }
  { Probe p; p.a.csel(13, 11, 12, kEq, false); p.show("csel w13, w11, w12, eq"); }
  { Probe p; p.a.movReg(9, 0); p.show("mov x9, x0"); }
  { Probe p; p.a.movWz(0, 3); p.show("mov w0, #3"); }
  { Probe p; p.a.dmbIsh(); p.show("dmb ish"); }
  { Probe p; p.a.ret(); p.show("ret"); }
  { Probe p; size_t l = p.a.label(); p.a.b(l); p.a.bind(l); p.show("b .+4"); }
  { Probe p; size_t l = p.a.label(); p.a.bcond(kHs, l); p.a.bind(l); p.show("b.hs .+4"); }
  { Probe p; size_t l = p.a.label(); p.a.cbz(12, true, l); p.a.bind(l); p.show("cbz x12, .+4"); }
  { Probe p; size_t l = p.a.label(); p.a.cbnz(12, false, l); p.a.bind(l); p.show("cbnz w12, .+4"); }
  return 0;
}
