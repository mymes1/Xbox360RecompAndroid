/**
 * Runtime diagnostics: unsupported-API accounting + crash reporting.
 *
 * Goals (from the Xenoverse bring-up brief):
 *   - "first unsupported kernel / GPU / filesystem / audio call" must be
 *     answerable from the log, not from a debugger;
 *   - a native crash must leave a readable report (guest PC, PPC registers,
 *     host backtrace, recent log lines) in logcat *and* in a file the user can
 *     attach to a bug report;
 *   - it must be cheap and it must never change emulator semantics: every hook
 *     here is observation-only.
 */
#pragma once
#include <cstdint>
#include <string>

namespace x360 {
namespace diag {

enum class Subsystem : uint8_t {
  Kernel = 0,   // xboxkrnl.exe / xam.xex exports
  Gpu = 1,      // PM4 packets, registers, shader translation
  FileSystem = 2,
  Audio = 3,
  Memory = 4,
  Input = 5,
  Count = 6,
};

const char* subsystem_name(Subsystem s);

// A single "not implemented" record. `detail` is e.g. "xboxkrnl 0x48 NtQueryDirectoryFile".
// Returns true the first time this key is seen (i.e. "this is the first one").
bool report_unsupported(Subsystem s, const char* detail);

// Total hits for a key (useful to tell "never implemented" from "hot stub").
uint32_t unsupported_hits(Subsystem s, const char* detail);

// Number of distinct unsupported entries recorded per subsystem.
uint32_t unsupported_count(Subsystem s);

// Render the whole table into `out` (returns bytes written).
uint32_t format_report(char* out, uint32_t max);

// ── Crash diagnostics ─────────────────────────────────────────────────────────
// Install SIGSEGV/SIGBUS/SIGILL/SIGABRT/SIGFPE handlers. Call once at engine
// init; safe to call repeatedly. `report_dir` may be empty (logcat only).
void install_crash_handler(const std::string& report_dir);
void uninstall_crash_handler();

// Attach guest state so that a crash inside JIT'd code can print the PPC PC.
// Called by the JIT thread loop on every block entry (cheap: one store).
void set_guest_pc(void* thread_handle, uint64_t guest_pc);
void set_guest_pc_valid(void* thread_handle, bool valid);

// Register the guest memory mapping so the crash handler can tell "the guest
// dereferenced a bad pointer" from "the JIT itself faulted".
void set_guest_mapping(uintptr_t host_base, uint64_t size, uint64_t guest_base);

// Explicit crash report (used when the loader/JIT detects an unrecoverable
// state rather than faulting). Writes the same format as a signal crash.
void write_crash_report(const char* reason, uintptr_t fault_addr, uint64_t guest_pc);

// Path of the most recent report file ("" when none).
const std::string& last_report_path();

}  // namespace diag
}  // namespace x360
