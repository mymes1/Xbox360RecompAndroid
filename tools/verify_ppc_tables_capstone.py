#!/usr/bin/env python3
"""Cross-check the generated PPC decode tables against a real disassembler.

`native/src/jit/ppc_tables.inc` is the table the runtime decoder uses. Hand
listing ~150 PowerPC/Xenon opcodes from memory is how the previous revision of
this project ended up with XOs that pointed at the wrong instruction (and with
duplicate cases). This script rebuilds one instruction word per table row and
asks Capstone (which carries the LLVM PowerPC disassembler tables) what that
word actually is, so the row's class/selector can be checked against ground
truth instead of trust.

    /tmp/buildenv/bin/python tools/verify_ppc_tables_capstone.py            # report
    /tmp/buildenv/bin/python tools/verify_ppc_tables_capstone.py --fix-hint # grouped view

Rows whose expected mnemonic (EXPECT below) disagrees with Capstone are printed
with MISMATCH; rows with no expectation entry are printed with "?" so they can
be filled in.  The decoder and the interpreter must both follow whatever the
disassembler says; a row that Capstone cannot decode at all is a row that must
not exist (it becomes an honest "unimplemented instruction" report instead of a
guessed execution).
"""
import os
import re
import sys

import capstone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INC = os.path.join(ROOT, "native", "src", "jit", "ppc_tables.inc")

# ─── expected mnemonics, keyed by (section, xo) ───────────────────────────────
EXPECT = {
    ("x31", 0): "cmp",
    ("x31", 2): "tw",
    ("x31", 8): "subfc",
    ("x31", 10): "addc",
    ("x31", 11): "mulhwu",
    ("x31", 16): "blr",
    ("x31", 19): "mcrxr",
    ("x31", 20): "lwarx",
    ("x31", 21): "ldux",
    ("x31", 23): "lwarx",
    ("x31", 24): "cntlw",
    ("x31", 26): "cntlzw",
    ("x31", 28): "and",
    ("x31", 32): "cmpl",
    ("x31", 40): "subf",
    ("x31", 58): "cntlzd",
    ("x31", 60): "andc",
    ("x31", 72): "subfe",
    ("x31", 75): "mulhdu",
    ("x31", 84): "ldarx",
    ("x31", 88): "mcrfcr",
    ("x31", 104): "aeee",
    ("x31", 120): "lhaux",
    ("x31", 122): "popcntw",
    ("x31", 124): "nand",
    ("x31", 136): "ae",
    ("x31", 144): "rcbd?",
    ("x31", 150): "sync",
    ("x31", 200): "se",
    ("x31", 232): "aee",
    ("x31", 233): "mulhd",
    ("x31", 235): "mulw",
    ("x31", 264): "aze",
    ("x31", 266): "add",
    ("x31", 275): "dec?",
    ("x31", 284): "nor",
    ("x31", 316): "xnor",
    ("x31", 339): "mfspr",
    ("x31", 340): "stwcx",
    ("x31", 375): "min?",
    ("x31", 376): "popcntb",
    ("x31", 412): "orc",
    ("x31", 444): "or",
    ("x31", 459): "divwu",
    ("x31", 470): "divduw?",
    ("x31", 476): "xor",
    ("x31", 491): "divw",
    ("x31", 502): "divdu?",
    ("x31", 528): "bcctr",
    ("x31", 534): "popcntb?",
    ("x31", 535): "lfsux",
    ("x31", 536): "srw",
    ("x31", 566): "ldux?",
    ("x31", 567): "lfdux",
    ("x31", 597): "lhau",
    ("x31", 598): "eieio",
    ("x31", 599): "ldu",
    ("x31", 631): "ldux",
    ("x31", 662): "stwbrx",
    ("x31", 663): "stfsix",
    ("x31", 695): "stfsux",
    ("x31", 727): "stfiwx",
    ("x31", 728): "extsh",
    ("x31", 759): "stfdx",
    ("x31", 790): "lhbrx",
    ("x31", 792): "sraw",
    ("x31", 801): "mtmsr",
    ("x31", 822): "andi",
    ("x31", 824): "srawi",
    ("x31", 854): "lwsync",
    ("x31", 918): "sthbrx",
    ("x31", 922): "extsw",
    ("x31", 925): "extsb",
    ("x31", 954): "extshb?",
    ("x31", 957): "extsb?",
    ("x31", 986): "extsw?",
    ("x31", 1010): "dcbz",
}

FIELDS = dict(rt=3, ra=5, rb=7, rc=9, sh=4, mb=0, me=31, bf=0, spr=8)

WORD_RE = re.compile(
    r"R\(\s*(\d+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9A-Za-z|]*)\s*\)")


def build_word(section, xo, cls, sel, size, flags):
    """Assemble one representative instruction word for a table row."""
    f = FIELDS
    if section == "vmx":
        if cls in ("VmxLoad", "VmxStore"):
            return (4 << 26) | (f["rt"] << 21) | (f["ra"] << 16) | (f["rb"] << 11) | (xo << 1)
        if "fV3" in flags or "fVX" not in flags:
            return ((4 << 26) | (f["rt"] << 21) | (f["ra"] << 16) | (f["rb"] << 11)
                    | (f["rc"] << 6) | (xo << 1))
        return (4 << 26) | (f["rt"] << 21) | (f["ra"] << 16) | (f["rb"] << 11) | (xo << 1)
    if cls in ("Rl",) and "fShImm" in flags:
        return (31 << 26) | (f["rt"] << 21) | (f["ra"] << 16) | (f["sh"] << 11) | (f["mb"] << 6) | (xo << 1)
    if cls == "Rl":
        return (31 << 26) | (f["rt"] << 21) | (f["ra"] << 16) | (f["rb"] << 11) | (xo << 1)
    if cls in ("Cntlz", "Popcnt", "Extend", "Neg"):
        return (31 << 26) | (f["rt"] << 21) | (f["ra"] << 16) | (xo << 1)
    if cls == "Cmp":
        return (31 << 26) | (f["bf"] << 23) | (f["ra"] << 16) | (f["rb"] << 11) | (xo << 1)
    if cls == "Mfspr":
        return (31 << 26) | (f["rt"] << 21) | ((f["spr"] & 0x1F) << 11) | ((f["spr"] >> 5) << 16) | (xo << 1)
    if cls == "Mtspr":
        return (31 << 26) | (f["rb"] << 21) | ((f["spr"] & 0x1F) << 11) | ((f["spr"] >> 5) << 16) | (xo << 1)
    if cls in ("Mfcr", "Mtcrf"):
        return (31 << 26) | (f["rt"] << 21) | (0 << 16) | (f["rb"] << 11) | (xo << 1)
    if cls in ("Bclr",):
        return (31 << 26) | (0 << 21) | (4 << 16) | (xo << 1)
    if cls == "Sync":
        return (31 << 26) | (min(sel, 3) << 11) | (xo << 1)
    if cls == "Trap":
        return (31 << 26) | (f["rb"] << 16) | (f["rt"] << 11) | (xo << 1) | 0b1111100000
    if cls == "Nop":
        return (31 << 26) | (xo << 1)
    if cls == "Atomic":
        # store-conditional forms carry the data register in rS and set bit 0
        rs = 2 if sel >= 2 else 0
        return (31 << 26) | ((f["rt"] if not rs else f["rc"]) << 21) | (f["ra"] << 16) | (f["rb"] << 11) | (xo << 1) | rs
    return (31 << 26) | (f["rt"] << 21) | (f["ra"] << 16) | (f["rb"] << 11) | (xo << 1)


def rows():
    section = None
    for line in open(INC):
        if "kX31" in line:
            section = "x31"
            continue
        if "kVmx" in line:
            section = "vmx"
            continue
        m = WORD_RE.search(line)
        if m and section:
            xo, cls, sel, size, flags = m.groups()
            yield section, int(xo), cls, int(sel), int(size), flags


def main():
    mode = capstone.CS_MODE_64 | capstone.CS_MODE_BIG_ENDIAN
    md = capstone.Cs(capstone.CS_ARCH_PPC, mode)
    md.detail = False
    counts = {"ok": 0, "mismatch": 0, "unknown": 0}
    grouped = []
    for section, xo, cls, sel, size, flags in rows():
        word = build_word(section, xo, cls, sel, size, flags)
        got = None
        for ins in md.disasm(word.to_bytes(4, "big"), 0x1000):
            got = (ins.mnemonic + " " + ins.op_str).strip()
        exp = EXPECT.get((section, xo))
        key = "?" if exp is None else ("ok" if got and got.split()[0] == exp.split("?")[0] else "bad")
        if key == "ok":
            counts["ok"] += 1
        elif got is None:
            counts["unknown"] += 1
        else:
            counts["mismatch"] += 1
        tag = {"ok": "OK ", "bad": "MISMATCH ", "?": "?     "}[key]
        grouped.append(
            f"[{section}] xo={xo:4d} {cls}({sel},{size},{flags or '0'}) -> {tag}{got or 'NO DECODE'}"
            + (f"  (expected {exp})" if exp and key == "bad" else ""))
    width = max(len(g.split("->")[0]) for g in grouped)
    for g in grouped:
        left, right = g.split("->", 1)
        print(f"{left:<{width}}-> {right.strip()}")
    print("\nsummary:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
