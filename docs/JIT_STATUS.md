# JIT status — what is implemented, what is verified, what is blocking

Branch work for running a legally owned Xbox 360 `default.xex` (DBX: Dragon Ball
Xenoverse, title `0x4E4D0862`) on arm64 Android by **dynamic** PPC→ARM64 translation.

Phase 2 decision (still current): do **not** pre-compile the title with XenonRecomp
into a statically recompiled binary. Xenoverse loads content that references kernel
APIs by ordinal at runtime, its game data lives on the disc image the user selects,
and a static recompile would freeze the title into the APK (no user-selected data
directory, no way to keep the compat status honest). The runtime therefore decodes
guest code on the fly and executes it either JIT-compiled or interpreted.

## Layout

| File | Role |
| --- | --- |
| `native/src/jit/ppc_decode.cpp` + `native/include/jit/ppc_decode.h` | Decoder. A single masked scan of generated tables produces a `DecodedPpc` (class + selector + operand fields + semantic flags). No hand-written per-opcode branches. |
| `native/src/jit/ppc_tables.inc` | **Generated, do not edit.** Rows for every opcode Xenia's own table defines. Produced by `tools/gen_ppc_tables.py`. |
| `tools/gen_ppc_tables.py`, `tools/data/ppc_opcodes.txt` | Generator + the authoritative input: the 456 `INSTRUCTION(...)` rows extracted from Xenia's `ppc_opcode_table_gen.cc`. The encoding and mask of every row come from that table, never from prose. |
| `native/src/jit/jit_interpreter.cpp` | Reference semantics for each decoded class. Everything the backend declines runs here, so "JIT result == interpreter result" is the invariant. |
| `native/src/jit/arm64_backend.cpp` | AArch64 emitter for a deliberately small, verified subset, plus `backendSupports()` as the single capability gate. |
| `tests/arm64_asm_probe.cpp`, `tools/check_arm64_asm.py` | Encoder verification: every machine word the emitter can produce is disassembled with Capstone and compared with the intended instruction. |
| `tools/verify_ppc_tables_capstone.py` | Decoder verification: every generated table row must decode to the instruction Xenia says it is. |

## Verification actually performed

* `tools/gen_ppc_tables.py` emits 107 primary-31 rows and 76 primary-4 (AltiVec) rows
  from the Xenia table; keys are per-mask (`key = encoding & mask`), so no two
  instructions can alias. It fails loudly on a collision.
* `tools/check_arm64_asm.py` (Capstone 5) was run on the emitter: 62 words checked.
  It found and fixed real mistakes — `sdiv`/`udiv` swapped, `mov`-through-`ubfm` shift
  fields, `fmadd` emitting the single-precision form, `cbz`/`cbnz` putting `Rt` in the
  wrong field, and the FP load/store immediate form. Instructions whose encoding could
  not be confirmed on this host (`clz`, `fsqrt`, 32-bit `asr`, the FP `imm7` load/store
  form) were **removed from the backend**, not guessed at; those guest opcodes now run
  interpreted.
* Semantics were read out of Xenia's emitters (`ppc_emit_alu.cc`, `ppc_emit_fpu.cc`,
  `ppc_emit_altivec.cc`) rather than the ISA prose, which caused several earlier wrong
  rows: Xenon computes `add`/`subf`/logic on full 64-bit registers, `addc`'s `CA` is the
  plain carry, `mullw` is a sign-extended 64-bit product, `slw`/`srw` zero-extend, and
  bit 21 of an X-form is *only* `L` for compares (it is `OE` for arithmetic).

## Known gaps (not hidden, not faked)

1. **VMX128 (primaries 5 and 6) is unimplemented on purpose.** Xenia's table gives
   `vaddfp128`, `vsubfp128`, `vmaxfp128`, `vand128`, `vor128`, `vxor128`, `vnor128`,
   `vsel128`, `vperm128`, `vsldoi128` base encodings whose bits 6..10 are **non-zero**
   (`vsel128 = 0x14000410`, `vand128 = 0x14000530`, `vperm128 = 0x14000200`, …), so the
   AltiVec `vD/vA/vB/vC` field split provably does not apply and there is no offline
   source to confirm the real layout. The generator drops those rows with a printed
   note; such words decode `Invalid` and the interpreter reports
   `unimplemented <word> at 0x<pc>` instead of guessing. Xenoverse's shader/vertex
   path is where these appear, so this is expected to be a first-blocker candidate.
2. **No `fres`, `frsqrte`, `fre`.** These are architecturally approximate; producing an
   exact `1/x` instead would be a silent behaviour change, so they are unimplemented.
3. **The backend translates a subset** (moves/arithmetic/logic/compares,
   non-updating loads and stores with byte-order fixup, `rlwinm`-family rotates,
   unconditional + the four common conditional branch shapes, double-precision
   `fadd/fsub/fmul/fdiv/fmadd`-family/`fabs/fneg/fmr/fnabs/fcmpu`). Anything else ends
   the block and is interpreted. Record-form (`.`), carry forms (`adde`, `subfc`, …),
   `stfiwx`, `dcbz`, VMX/VMX128, atomics and FP-single are interpreter-only today.
4. **The JIT is disabled by default** (`X360_XBOX_JIT=0`): generated code has been
   verified *as encoded words*, never *as executed code* — that needs a real device.
   The interpreter is the reference path and is what the shipped configuration uses.
5. **No APK can be produced in this workspace.** No Android SDK/NDK, no Flutter, no
   Gradle, no JDK, and no `binutils-aarch64` package is reachable; the published
   repository additionally had four baseline compile errors (fixed). Phase 8's build
   therefore has to run on the user's machine:
   `flutter build apk --release --target-platform android-arm64`.

## Next steps, in order

1. Run `tools/gen_ppc_tables.py` and `tools/check_arm64_asm.py` in CI, and extend the
   generator to also emit `tests/ppc_decode_test_data.inc` (encoding → expected class +
   selector) so the decoder has a golden file test, not just a table test.
2. On device: enable `X360_XBOX_JIT=1`, then add an *in-process* differential mode that
   runs each block both interpreted and JIT-compiled and reports the first divergence —
   that is the only way to gain confidence in the backend.
3. Resolve the VMX128 field layout from a second source (PowerPC ISA book 4 v1.1
   VMX128 extension, or Xenia's `ppc_emit_altivec.cc` handling of `primary 5/6`) and
   teach the generator that form; until then the interpreter's "unimplemented" line is
   the diagnostic to look for.
4. Interpreter additions that unlock the most coverage for Xenoverse: record forms
   (`add.` etc.), `stfiwx`, `dcbz` in generated code, and atomics.
