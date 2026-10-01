"""Regenerate native/src/hle/kernel_ordinals.cpp from Xenia's public export tables.

Run from anywhere; expects a Xenia checkout at $XENIA (default /tmp/xenia).
The tables are public ordinal/name listings only — no code is copied.
"""
import os, re

XENIA = os.environ.get("XENIA", "/tmp/xenia")

def parse_table(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = re.match(r'\s*XE_EXPORT\((\w+),\s*(0x[0-9A-Fa-f]+),\s*(\w+),\s*(\w+)\)', line)
            if m:
                rows.append((int(m.group(2), 16), m.group(3), m.group(4) == 'kVariable'))
    # dedupe on (ordinal) keeping the first, then sort for binary search
    seen = set(); out = []
    for o, n, v in sorted(rows):
        if o in seen:
            continue
        seen.add(o); out.append((o, n, v))
    return out

krnl = parse_table(os.path.join(XENIA, 'src/xenia/kernel/xboxkrnl/xboxkrnl_table.inc'))
xam = parse_table(os.path.join(XENIA, 'src/xenia/kernel/xam/xam_table.inc'))

K = {n: o for o, n, v in krnl}
X = {n: o for o, n, v in xam}

# Ordinals our HLE currently implements (kept short on purpose: the runtime
# registers its own handlers at startup via mark_implemented(), so this list is
# only a compile-time hint of "known-good").
impl_krnl = ["DbgPrint", "ExCreateThread", "ExTerminateThread",
             "ExAcquireReadWriteLockExclusive", "ExAcquireReadWriteLockShared",
             "ExReleaseReadWriteLock", "ExAllocatePoolWithTag", "ExFreePool",
             "NtAllocateVirtualMemory", "NtFreeVirtualMemory", "NtProtectVirtualMemory",
             "NtQueryVirtualMemory", "NtCreateFile", "NtOpenFile", "NtClose", "NtReadFile",
             "NtWriteFile", "NtFlushBuffersFile", "NtQueryInformationFile",
             "NtSetInformationFile", "NtQueryDirectoryFile", "NtQueryFullAttributesFile",
             "NtDeleteFile", "NtCancelIoFile", "NtDeviceIoControlFile", "NtCreateEvent",
             "NtSetEvent", "NtClearEvent", "NtPulseEvent", "NtCreateMutant",
             "NtReleaseMutant", "NtCreateSemaphore", "NtReleaseSemaphore",
             "NtWaitForSingleObjectEx", "NtWaitForMultipleObjectsEx", "NtYieldExecution",
             "NtCreateTimer", "NtCancelTimer", "KeQuerySystemTime",
             "KeQueryPerformanceFrequency", "KeStallExecutionProcessor", "KeSetAffinityThread",
             "RtlInitAnsiString", "RtlInitUnicodeString", "RtlMultiByteToUnicodeN",
             "RtlUnicodeToMultiByteN", "RtlUpcaseUnicodeChar", "RtlCompareUnicodeString",
             "XexGetModuleHandle", "XexGetProcedureAddress", "XexLoadImage", "XexUnloadImage"]
impl_xam = ["XamInputGetState", "XamInputSetState", "XamInputGetCapabilities",
            "XGetLanguage", "XGetVideoFlags", "XGetAudioFlags", "XamGetLocale",
            "XamGetCurrentTitleId", "XamAlloc", "XamFree", "XamGetSystemVersion",
            "XMemAlloc", "XMemFree", "XMemSet",
            "XamNotifyCreateListener", "XamIsUIActive", "XamGetExecutionId",
            "XamLoaderGetLaunchData", "XamLoaderSetLaunchData"]

rows = []
missing = []
for n in impl_krnl:
    if n in K:
        rows.append('    {"xboxkrnl.exe", 0x%05x},  // %s' % (K[n], n))
    else:
        missing.append('xboxkrnl:' + n)
for n in impl_xam:
    tgt = X if n in X else K
    mod = "xam.xex" if n in X else "xboxkrnl.exe"
    if n in tgt:
        rows.append('    {"%s", 0x%05x},  // %s' % (mod, tgt[n], n))
    else:
        missing.append('xam:' + n)

def emit(name, tag, data):
    out = ['// %s: %d documented exports (sorted by ordinal for binary search)' % (tag, len(data)),
           'static const OrdinalEntry %s[] = {' % name]
    for o, n, v in data:
        out.append('    {0x%05x, "%s", %d},' % (o, n, 1 if v else 0))
    out.append('};')
    return '\n'.join(out)

header_note = """/**
 * Generated: Xbox 360 kernel export ordinal tables + implementation registry.
 * Regenerate with tools/gen_kernel_ordinals.py (needs a Xenia checkout to read
 * the public tables from; nothing from Xenia's implementation is copied).
 */
"""

body = """#include "../../include/x360/kernel_ordinals.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_set>

namespace x360 {
namespace kernel {
namespace {

%(krnl)s

%(xam)s

// Ordinals the HLE layer registers a handler for at startup (see
// mark_implemented(), which hle_kernel.cpp calls as it wires dispatch). This
// compile-time list documents the intent and is merged with the runtime set.
struct NamedOrdinal { const char* module; uint32_t ordinal; };
static const NamedOrdinal kBuiltInImplemented[] = {
%(rows)s
};

uint32_t module_id(const char* module) {
  if (!module) return 2u;
  if (!strcmp(module, "xboxkrnl.exe") || !strcmp(module, "xboxkrnl")) return 0u;
  if (!strcmp(module, "xam.xex") || !strcmp(module, "xam")) return 1u;
  return 2u;
}

uint64_t key(const char* module, uint32_t ordinal) {
  return (uint64_t(module_id(module)) << 32) | ordinal;
}

std::unordered_set<uint64_t>& implemented_set() {
  static std::unordered_set<uint64_t> set = [] {
    std::unordered_set<uint64_t> s;
    for (const auto& e : kBuiltInImplemented) s.insert(key(e.module, e.ordinal));
    return s;
  }();
  return set;
}

std::mutex& implemented_mutex() { static std::mutex m; return m; }

const OrdinalEntry* table_for(const char* module, uint32_t* count) {
  if (!module) { *count = 0; return nullptr; }
  if (!strcmp(module, "xboxkrnl.exe") || !strcmp(module, "xboxkrnl")) {
    *count = static_cast<uint32_t>(sizeof(kXboxkrnl) / sizeof(kXboxkrnl[0]));
    return kXboxkrnl;
  }
  if (!strcmp(module, "xam.xex") || !strcmp(module, "xam")) {
    *count = static_cast<uint32_t>(sizeof(kXam) / sizeof(kXam[0]));
    return kXam;
  }
  *count = 0;
  return nullptr;
}

}  // namespace

bool is_kernel_module(const char* name) {
  return name && (!strcmp(name, "xboxkrnl.exe") || !strcmp(name, "xboxkrnl") ||
                  !strcmp(name, "xam.xex") || !strcmp(name, "xam"));
}

const char* name_for_ordinal(const std::string& module, uint32_t ordinal) {
  uint32_t count = 0;
  const OrdinalEntry* t = table_for(module.c_str(), &count);
  if (!t) return nullptr;
  const OrdinalEntry* it = std::lower_bound(
      t, t + count, ordinal, [](const OrdinalEntry& e, uint32_t v) { return e.ordinal < v; });
  if (it != t + count && it->ordinal == ordinal) return it->name;
  return nullptr;
}

bool ordinal_for_name(const char* module, const char* name, uint32_t* out) {
  uint32_t count = 0;
  const OrdinalEntry* t = table_for(module, &count);
  if (!t || !name) return false;
  for (uint32_t i = 0; i < count; i++) {
    if (!strcmp(t[i].name, name)) {
      if (out) *out = t[i].ordinal;
      return true;
    }
  }
  return false;
}

bool is_implemented(const std::string& module, uint32_t ordinal) {
  uint32_t count = 0;
  if (!table_for(module.c_str(), &count)) return false;
  std::lock_guard<std::mutex> lk(implemented_mutex());
  return implemented_set().count(key(module.c_str(), ordinal)) != 0;
}

void mark_implemented(const char* module, uint32_t ordinal) {
  std::lock_guard<std::mutex> lk(implemented_mutex());
  implemented_set().insert(key(module, ordinal));
}

uint32_t xboxkrnl_export_count() { return static_cast<uint32_t>(sizeof(kXboxkrnl)/sizeof(kXboxkrnl[0])); }
uint32_t xam_export_count() { return static_cast<uint32_t>(sizeof(kXam)/sizeof(kXam[0])); }

}  // namespace kernel
}  // namespace x360
""" % dict(krnl=emit('kXboxkrnl', 'xboxkrnl.exe', krnl), xam=emit('kXam', 'xam.xex', xam),
           rows='\n'.join(rows))

out_path = 'native/src/hle/kernel_ordinals.cpp'
open(out_path, 'w').write(header_note + body)
print("wrote", out_path, "krnl", len(krnl), "xam", len(xam), "impl rows", len(rows))
print("missing:", missing)
