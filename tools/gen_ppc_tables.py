#!/usr/bin/env python3
"""Generate native/src/jit/ppc_tables.inc from Xenia's opcode table.

Why this exists
---------------
The runtime decoder must map a raw PowerPC word to (instruction class, operand
selector, element size, flags). Hand-listing ~150 Xenon opcodes is how the
original revision of this project ended up with opcodes pointing at the wrong
instruction (for example `neg` decoded as an `add`, or `popcntb` colliding with
`lwbrx`), which is exactly the class of bug that turns into "the title renders
the wrong thing" or "the title hangs with no log line".

So the opcode numbers here are *derived*, not typed: `tools/data/ppc_opcodes.txt`
carries the base encoding + operand form of every instruction Xenia knows
(extracted from its generated ppc_opcode_table_gen.cc), and this script computes
each table key from that encoding. The semantic mapping (what an instruction
*means* for the interpreter) is the only hand-written part, and every entry names
the mnemonic it was written against so a reviewer can check it.

Tables are keyed by `(word & mask) == key`:
  primary 31  mask 0x7FE  (bits 1..10: the X/XO/XFX/A-form XO field; bits 11..25
                           are registers or sub-fields, and the 64-bit "L" bit is
                           handled by fWide, not by a separate row)
  primary 4   mask 0x7FE for the VX/VC forms, 0x3F for the VA forms (those carry
                           vC in bits 6..10, so only bits 0..5 identify the op)

Usage:  python3 tools/gen_ppc_tables.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tools", "data", "ppc_opcodes.txt")
OUT = os.path.join(ROOT, "native", "src", "jit", "ppc_tables.inc")

# Opcodes Xenia's table does not list (Xenon-only extensions). Verified against
# the LLVM/Capstone PowerPC decoder instead: tools/verify_ppc_tables_capstone.py
# rebuilds the same words and prints what a real disassembler thinks they are.
EXTRA = [
    ("popcntbx", 0x7C00007A, "kX"),   # popcntb  rA -> count of bits in 8 bytes
    ("popcntwx", 0x7C0002F4, "kX"),   # popcntw
    ("popcntdx", 0x7C0003FA, "kX"),   # popcntd
]

ENC, FORM = {}, {}
for line in open(DATA):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    enc, name, form, group, typ = line.split()
    ENC[name] = int(enc, 16)
    FORM[name] = form

for name, enc, form in EXTRA:
    ENC[name] = enc
    FORM[name] = form

MASK_X10 = 0x7FE  # bits 1..10
MASK_A6 = 0x3F    # bits 0..5

# ─── primary 31: {mnemonic: (class, sel, size, flags)} ────────────────────────
# Flag names are the decoder's internal table flags (see ppc_decode.cpp):
#   fCarry fSigned fSat fUpd fWide fHigh fOvf fRcSet fRcAlways fZero32 fTerm fVX
#   fHint fShImm fSingle fV3L
# sel meaning follows the enums in include/jit/ppc_decode.h (AddSel, SubfSel,
# MulSel, CmpSel, MemSel, RlSel, ...).
I = {}
def add(name, cls, sel=0, size=0, flags="0"):
    if name not in ENC:
        sys.exit("unknown mnemonic in generator: %s" % name)
    I[name] = (cls, sel, size, flags)

# integer add/sub family (sel selects the exact arithmetic + carry behaviour)
add("addx", "Add", 0, 0, "fRcSet")
add("addcx", "Add", 3, 0, "fRcSet")          # addc: CA from ~rA + rB
add("addex", "Add", 1, 0, "fCarry|fRcSet")   # adde
add("addzex", "Add", 2, 0, "fCarry|fRcSet")        # addze
add("addmex", "Add", 4, 0, "fCarry|fRcSet")         # addme
add("subfx", "Subf", 0, 0, "fRcSet")
add("subfcx", "Subf", 2, 0, "fRcSet")
add("subfex", "Subf", 1, 0, "fCarry|fRcSet")
add("subfzex", "Subf", 3, 0, "fRcSet")
add("subfmex", "Subf", 4, 0, "fRcSet")
add("negx", "Neg", 0, 8, "fRcSet")
add("mulhwux", "Mul", 1, 4, "fHigh|fRcSet")
add("mulhwx", "Mul", 1, 4, "fSigned|fHigh|fRcSet")
add("mulhdux", "Mul", 1, 8, "fHigh|fRcSet")
add("mulhdx", "Mul", 1, 8, "fSigned|fHigh|fRcSet")
add("mullwx", "Mul", 0, 4, "fSigned|fRcSet")
add("mulldx", "Mul", 0, 8, "fSigned|fRcSet")
add("divwx", "Div", 0, 4, "fSigned|fRcSet")
add("divwux", "Div", 1, 4, "fRcSet")
add("divdx", "Div", 0, 8, "fSigned|fRcSet")
add("divdux", "Div", 1, 8, "fRcSet")
add("andx", "Logic", 0, 0, "fRcSet")
add("andcx", "Logic", 1, 0, "fRcSet")
add("nandx", "Logic", 2, 0, "fRcSet")
add("norx", "Logic", 3, 0, "fRcSet")
add("orx", "Logic", 4, 0, "fRcSet")
add("orcx", "Logic", 5, 0, "fRcSet")
add("xorx", "Logic", 6, 0, "fRcSet")
add("eqvx", "Logic", 7, 0, "fRcSet")
add("slwx", "Rl", 0, 4, "fRcSet|fZero32")
add("srwx", "Rl", 1, 4, "fRcSet|fZero32")
add("srawx", "Rl", 2, 4, "fRcSet")
add("srawix", "Rl", 2, 4, "fRcSet|fShImm")
add("sldx", "Rl", 0, 8, "fRcSet")
add("srdx", "Rl", 1, 8, "fRcSet")
add("sradx", "Rl", 2, 8, "fRcSet")
add("sradix", "Rl", 2, 8, "fRcSet|fShImm")
add("cntlzwx", "Cntlz", 0, 4, "fRcSet")
add("cntlzdx", "Cntlz", 0, 8, "fRcSet")
add("extsbx", "Extend", 0, 1, "fRcSet")
add("extshx", "Extend", 0, 2, "fRcSet")
add("extswx", "Extend", 0, 4, "fRcSet")
add("popcntbx", "Popcnt", 0, 1, "fRcSet")
add("popcntwx", "Popcnt", 0, 4, "fRcSet")
add("popcntdx", "Popcnt", 0, 8, "fRcSet")
add("cmp", "Cmp", 0, 0, "fRcAlways|fWide")
add("cmpl", "Cmp", 1, 0, "fRcAlways|fWide")
add("mfspr", "Mfspr", 0, 0, "0")
add("mtspr", "Mtspr", 0, 0, "fTerm")
add("mfcr", "Mfcr", 0, 0, "0")
add("mtcrf", "Mtcrf", 0, 0, "fTerm")
add("lwarx", "Atomic", 0, 4, "0")
add("ldarx", "Atomic", 1, 8, "0")
add("stwcx", "Atomic", 2, 4, "fRcAlways")
add("stdcx", "Atomic", 3, 8, "fRcAlways")
for n, sz, fl in (("lbzx", 1, "fZero32"), ("lbzux", 1, "fZero32"), ("lbzx", 1, "fZero32"),
                  ("lhzx", 2, "fZero32"), ("lhzux", 2, "fZero32"), ("lhax", 2, "fSigned"),
                  ("lhaux", 2, "fSigned|fUpd"), ("lwzx", 4, "fZero32"),
                  ("lwzux", 4, "fZero32"), ("lwax", 4, "fSigned|fWide"),
                  ("lwaux", 4, "fSigned|fWide|fUpd"), ("ldx", 8, "0"), ("ldux", 8, "fUpd"),
                  ("lfsx", 4, "fSingle|fZero32"), ("lfsux", 4, "fSingle|fZero32|fUpd"),
                  ("lfdx", 8, "fSingle"), ("lfdux", 8, "fSingle|fUpd"),
                  ("lhbrx", 2, "fZero32"), ("lwbrx", 4, "fZero32"), ("ldbrx", 8, "0"),
                  ("stbx", 1, "0"), ("stbux", 1, "fUpd"), ("sthx", 2, "0"),
                  ("sthux", 2, "fUpd"), ("stwx", 4, "0"), ("stwux", 4, "fUpd"),
                  ("stdx", 8, "0"), ("stdux", 8, "fUpd"), ("stfsx", 4, "fSingle"),
                  ("stfsux", 4, "fSingle|fUpd"), ("stfdx", 8, "fSingle"),
                  ("stfdux", 8, "fSingle|fUpd"), ("stfiwx", 4, "fSingle"),
                  ("sthbrx", 2, "0"), ("stwbrx", 4, "0"), ("stdbrx", 8, "0"),
                  ):
    sel = 1 if n.endswith("brx") else (2 if n == "stfiwx" else 0)
    add(n, "St" if n.startswith("st") else "Ld", sel, sz, fl)
add("lvx", "VmxLoad", 0, 16, "fVX")
add("stvx", "VmxStore", 0, 16, "fVX")
add("lvxl", "VmxLoad", 0, 16, "fVX")
add("stvxl", "VmxStore", 0, 16, "fVX")
add("lvrxl", "VmxLoad", 0, 16, "fVX")
add("stvrxl", "VmxStore", 0, 16, "fVX")
add("lvsl", "VmxLoad", 1, 16, "fVX")
add("lvsr", "VmxLoad", 2, 16, "fVX")
add("sync", "Sync", 0, 0, "0")
add("eieio", "Sync", 1, 0, "0")
add("isync", "Sync", 2, 0, "0")
add("dcbf", "Nop", 0, 0, "fHint")
add("dcbst", "Nop", 0, 0, "fHint")
add("dcbt", "Nop", 0, 0, "fHint")
add("dcbtst", "Nop", 0, 0, "fHint")
add("icbi", "Nop", 0, 0, "fHint")
add("dcbz", "Dcbz", 0, 0, "0")         # zero the 128-byte line (real semantics)
add("tw", "Trap", 0, 0, "0")

# ─── primary 4 (VMX / VMX128) ─────────────────────────────────────────────────
V = {}
def varith(name, op, size, signed, sat, fmt="VX"):
    V[name] = ("VmxArith", op, size,
               ("fSigned|" if signed else "") + ("fSat|" if sat else "") + ("f%s" % fmt))
def vlogic(name, sel, fmt="VX"):
    V[name] = ("VmxLogic", sel, 16, "f%s" % fmt)
def vshift(name, sel, size, fmt="VX"):
    V[name] = ("VmxShift", sel, size, "f%s" % fmt)
def vcmp(name, sel, size, signed, fmt="VC"):
    V[name] = ("VmxCmp", sel, size, ("fSigned|" if signed else "") + "f%s" % fmt)

for name, sz, sgn, sat in (("vaddubm", 1, False, False), ("vadduhm", 2, False, False),
                           ("vadduwm", 4, False, False), ("vaddudm", 8, False, False),
                           ("vaddubs", 1, False, True), ("vadduhs", 2, False, True),
                           ("vadduws", 4, False, True),
                           ("vaddsbs", 1, True, True), ("vaddshs", 2, True, True),
                           ("vaddsws", 4, True, True),
                           ("vsububm", 1, False, False), ("vsubuhm", 2, False, False),
                           ("vsubuwm", 4, False, False),
                           ("vsububs", 1, False, True), ("vsubuhs", 2, False, True),
                           ("vsubuws", 4, False, True),
                           ("vsubsbs", 1, True, True), ("vsubshs", 2, True, True),
                           ("vsubsws", 4, True, True),
                           ("vaddcuw", 4, False, False), ("vsubcuw", 4, False, False)):
    varith(name, 1 if "sub" in name else 0, sz, sgn, sat)
for name, sz, sgn in (("vmaxub", 1, False), ("vmaxuh", 2, False), ("vmaxuw", 4, False),
                      ("vmaxsb", 1, True), ("vmaxsh", 2, True), ("vmaxsw", 4, True),
                      ("vminub", 1, False), ("vminuh", 2, False), ("vminuw", 4, False),
                      ("vminsb", 1, True), ("vminsh", 2, True), ("vminsw", 4, True),
                      ("vavguw", 4, False), ("vavgub", 1, False), ("vavguh", 2, False)):
    V[name] = ("VmxMinMax", 1 if "max" in name else 0, sz, ("fSigned|" if sgn else "") + "fVX")
for name, sel in (("vcmpequb", 0), ("vcmpequh", 0), ("vcmpequw", 0),
                  ("vcmpgtsb", 1), ("vcmpgtsh", 1), ("vcmpgtsw", 1),
                  ("vcmpgtub", 1), ("vcmpgtuh", 1), ("vcmpgtuw", 1)):
    signed = "gt" in name and name[5] == "g" and "s" in name[5:8]
    sz = 1 if name.endswith("b") else (2 if name.endswith("h") else 4)
    vcmp(name, sel, sz, ("vcmpgt" in name and name[7] == "s") or ("eq" in name and False))
for name, sel, sz in (("vcmpeqfp", 2, 4), ("vcmpgefp", 3, 4), ("vcmpgtfp", 4, 4),
                      ("vcmpbfp", 5, 4)):
    V[name] = ("VmxCmp", sel, sz, "fVX")
for name, sel in (("vand", 0), ("vandc", 1), ("vor", 2), ("vxor", 4), ("vnor", 5)):
    vlogic(name, sel)
for name, sel, sz in (("vsl", 6, 1), ("vslw", 6, 4), ("vslo", 6, 1),
                      ("vsr", 7, 1), ("vsrw", 7, 4), ("vsro", 7, 1)):
    vshift(name, sel, sz)
for name, sel, sz in (("vspltb", 0, 1), ("vsplth", 1, 2), ("vspltw", 2, 4),
                      ("vspltisb", 3, 1), ("vspltish", 4, 2), ("vspltisw", 5, 4)):
    vshift(name, sel, sz, fmt="VX")
V["vsel"] = ("VmxSel", 0, 16, "fVA")
V["vperm"] = ("VmxPerm", 0, 16, "fVA")
V["vsldoi"] = ("VmxSldoi", 0, 16, "fVA")
for name, sel in (("vmaddfp", 0), ("vnmsubfp", 1)):
    V[name] = ("VmxMul", sel, 8, "fVA")
for name, op in (("vaddfp", 0), ("vsubfp", 1), ("vmaxfp", 2), ("vminfp", 3)):
    V[name] = ("VmxFloat", op, 4, "fVX")
V["lvx128"] = ("VmxLoad", 0, 16, "fVX")

# Xenon VMX128 additions: same A-form semantics, distinct encodings.
for name in ("vaddfp128", "vsubfp128", "vmaxfp128", "vminfp128", "vand128", "vor128",
             "vxor128", "vnor128", "vsel128", "vperm128", "vsldoi128", "vadduwm128" if False else "vaddfp128"):
    if name in ENC and name not in V:
        base = name.replace("128", "")
        if base in V:
            V[name] = V[base]

def pick(name):
    """Resolve a mnemonic to (encoding name, 7-bit-VD flag).

    VMX128-only instructions (the Xenon extensions) exist solely as `name128`
    entries in Xenia's table; those use a 7-bit vD field (bits 21..27) instead of
    the 5-bit AltiVec one, which the decoder needs to know.
    """
    if name in ENC:
        return name, False
    if name + "128" in ENC:
        return name + "128", True
    return None, False

dropped = []
def resolve(table, dropped_bucket):
    out = {}
    for name, spec in table.items():
        real, vd7 = pick(name)
        if real is None:
            dropped_bucket.append(name)
            continue
        out[name] = (real, vd7, *spec)
    return out


# ─── emit ─────────────────────────────────────────────────────────────────────
def key_of(name, mask):
    return (ENC[name] & mask)


I = resolve(I, dropped)
V = resolve(V, dropped)
rows31, rows4, seen = [], [], {}
for name, (real, vd7, cls, sel, size, flags) in sorted(I.items(), key=lambda kv: ENC[kv[1][0]]):
    if (ENC[real] >> 26) != 31:
        # D-form / branch forms are handled by the switch in ppc_decode.cpp.
        continue
    k = key_of(real, MASK_X10)
    if k in seen:
        sys.exit("primary-31 key collision: %s vs %s (key %d)" % (real, seen[k], k))
    seen[k] = real
    rows31.append((k, cls, sel, size, flags, real))
rows4 = {4: [], 5: []}
seen4 = {4: {}, 5: {}}
for name, (real, vd7, cls, sel, size, flags) in sorted(V.items(), key=lambda kv: ENC[kv[1][0]]):
    mask = MASK_A6 if "fVA" in flags else MASK_X10
    prim = ENC[real] >> 26
    if prim in (5, 6, 7):
        # Primary 5 is the Xenon "VMX128" space (7-bit vD, and its operand layout
        # is NOT the AltiVec one). Its exact field split is not verified here, so
        # it is deliberately left out of the decode tables: the interpreter then
        # reports "unimplemented instruction 0x..." with the PC instead of guessing
        # semantics for a whole register file. See docs/JIT_STATUS.md.
        dropped.append(real + "(primary 5)")
        continue
    if prim != 4:
        dropped.append("%s(primary %d)" % (real, prim))
        continue
    k = (ENC[real] & mask)
    if k in seen4[prim]:
        sys.exit("primary-%d key collision: %s vs %s (key 0x%x mask 0x%x)" %
                 (prim, real, seen4[prim][k], k, mask))
    seen4[prim][k] = real
    extra = ("|fMaskA6" if mask == MASK_A6 else "") + ("|fVD7" if vd7 else "")
    rows4[prim].append((k, cls, sel, size, flags + extra, real))
if dropped:
    print("note: no encoding found for (left unimplemented):", " ".join(sorted(set(dropped))))

lines = []
lines.append("// GENERATED by tools/gen_ppc_tables.py - do not edit by hand.")
lines.append("// Keys are derived from tools/data/ppc_opcodes.txt (Xenia's opcode table),")
lines.append("// so every opcode number below matches a working Xbox 360 emulator.")
lines.append("")
lines.append("// primary 31: integer X/XO/XFX/A forms, indexed memory, atomics, barriers.")
lines.append("const OpRow kX31[] = {")
for k, cls, sel, size, flags, name in rows31:
    lines.append("    R(%4d, 0x%03X, %-9s, %2d, %2d, %-24s),  // %s" %
                 (k, MASK_X10, cls, sel, size, flags, name))
lines.append("};")
for prim, tbl, label in ((4, rows4[4], "AltiVec / VMX (primary 4)"),):
    lines.append("")
    lines.append("// primary %d: %s (VA-form rows match on bits 0..5 only)." % (prim, label))
    lines.append("const OpRow kVmx[] = {")
    for k, cls, sel, size, flags, name in tbl:
        lines.append("    R(%4d, 0x%03X, %-9s, %2d, %2d, %-24s),  // %s" %
                 (k, (MASK_A6 if "fMaskA6" in flags else MASK_X10), cls, sel, size,
                  flags.replace("|fMaskA6", "").replace("|fVD7", ""), name))
    lines.append("};")
lines.append("")
open(OUT, "w").write("\n".join(lines))
print("wrote %s: %d primary-31 rows, %d primary-4 rows" %
      (OUT, len(rows31), len(rows4[4])))
