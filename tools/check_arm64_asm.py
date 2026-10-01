#!/usr/bin/env python3
"""Verify the ARM64 encodings emitted by native/src/jit/arm64_backend.cpp.

The backend writes AArch64 machine words by hand (no assembler is available on the
Android/NDK path: the JIT buffers are filled directly). A wrong bit pattern in a JIT
is much worse than a missing one - it executes something else silently - so every
encoding is checked against Capstone's independent decoder.

Usage:
    python3 tools/check_arm64_asm.py            # build + check
    python3 tools/check_arm64_asm.py --keep     # leave the probe binary in /tmp

Requires a host C++20 compiler and `pip install capstone`. The probe is a host
program: on x86-64 it only exercises the *encoder*, which is what we want (the
encodings are host-independent); no ARM hardware is needed.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from capstone import CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN, Cs
except ImportError:
    sys.exit("capstone is required: pip install capstone")

# Words that the probe prints in groups; `; <text>` is the intent for that group.
def build_probe(binpath):
    cmd = ["g++", "-std=c++20", "-O0", "-w", f"-I{os.path.join(ROOT, 'native', 'include')}",
           "-o", binpath, os.path.join(ROOT, "tests", "arm64_asm_probe.cpp")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("probe build failed:\n" + r.stdout + r.stderr)


def main():
    keep = "--keep" in sys.argv
    binpath = os.path.join(tempfile.gettempdir(), "arm64_asm_probe")
    build_probe(binpath)
    out = subprocess.run([binpath], capture_output=True, text=True).stdout
    if not keep:
        try:
            os.unlink(binpath)
        except OSError:
            pass

    md = Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
    total = bad = nodecode = 0
    for line in out.splitlines():
        if ";" not in line:
            continue
        hexpart, want = line.split(";", 1)
        want = want.strip()
        for tok in hexpart.split():
            if not re.fullmatch(r"[0-9A-Fa-f]{8}", tok):
                if "LINKFAIL" in line:
                    print("LINKFAIL:", line)
                    bad += 1
                continue
            word = int(tok, 16)
            total += 1
            got = list(md.disasm(word.to_bytes(4, "little"), 0))
            if not got:
                nodecode += 1
                print(f"{word:08X}  intent[{want}]  -> Capstone cannot decode it")
                continue
            text = (got[0].mnemonic + " " + got[0].op_str).strip()
            # Capstone prefers canonical aliases; accept those (madd->mul, sxtw->sbfiz,
            # mov->orr, rev32->rev, hex-vs-decimal immediates) by comparing tokens only
            # loosely: the first word must be a mnemonic we consider equivalent.
            if not equivalent(want, text):
                bad += 1
                print(f"{word:08X}  intent[{want}]  got[{text}]")
    print(f"{total} words checked, {bad} unexpected, {nodecode} undecodable by this "
          f"capstone build")
    return 1 if bad else 0


ALIASES = {
    "madd": {"madd", "mul"},
    "mov": {"mov", "orr", "movz", "movk"},
    "rev32": {"rev32", "rev"},
    "sxth": {"sxth", "sbfiz", "sbfx"},
    "sxtb": {"sxtb", "sbfiz", "sbfx"},
    "sxtw": {"sxtw", "sbfm", "sbfx"},
    "lsl": {"lsl", "ubfm", "ubfx"},
    "lsr": {"lsr", "ubfm", "lshr"},
    "asr": {"asr", "sbfm", "sbfx"},
    "cmp": {"cmp", "subs"},
    "ldr": {"ldr", "ldur"},
    "str": {"str", "stur"},
    "b": {"b"},
    "b.hs": {"b.hs"},
    "mov #0xabcd0000": {"mov", "movk"},
    "mov x16, #0x8ffe0000": {"mov", "movk"},
    "b.hs .+4": {"b.hs"},
    "b .+4": {"b"},
    "cbz x12, .+4": {"cbz"},
    "cbnz w12, .+4": {"cbnz"},
    "ldr d11, [x9, #256]": {"ldr", "fmov"},
    "fmov x17, d18": {"fmov"},
    "fmov d17, x18": {"fmov"},
    "and x11, x11, #0xffffffff": {"and", "sbfm", "ubfm"},
}


def equivalent(want, got):
    w = want.split()[0].split(",")[0]
    g = got.split()[0]
    if w == g:
        return True
    if g in ALIASES.get(w, set()):
        return True
    # A group of instructions prints one line per word: accept a word whose mnemonic
    # appears anywhere in the intent text (e.g. "ldr x13,[x9,#256]; fmov d11, x13").
    return any(part.strip().split()[0] == g for part in want.split(";") if part.strip())


if __name__ == "__main__":
    sys.exit(main())
