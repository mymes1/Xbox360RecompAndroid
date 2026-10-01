/**
 * Diagnostics implementation: unsupported-API accounting + crash reporting.
 *
 * Signal handlers use only async-signal-safe calls (write/backtrace is the
 * exception and is best-effort), because a SIGSEGV inside JIT'd code must not
 * deadlock trying to allocate.
 */
#include "../../include/x360/diagnostics.h"
#include "../../include/x360/log.h"

#include <atomic>
#include <cstdio>
#include <fcntl.h>
#include <ctime>
#include <cstring>
#include <mutex>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <execinfo.h>

#define TAG "X360:DIAG"

namespace x360 {
namespace diag {
namespace {

constexpr uint32_t kMaxRecords = 4096;

struct Record {
  Subsystem subsys = Subsystem::Kernel;
  char key[128] = {};
  uint32_t hits = 0;
  uint64_t first_tick = 0;
  uint64_t last_tick = 0;
};

struct State {
  std::mutex mutex;
  std::unordered_map<uint64_t, uint32_t> index;  // hash(subsys,key) -> record slot
  std::vector<Record> records;
  std::atomic<uint64_t> tick{0};
  bool enabled = true;
};

State& state() {
  static State s;
  return s;
}

uint64_t hash_key(Subsystem s, const char* key) {
  uint64_t h = 1469598103934665603ULL;
  h ^= static_cast<uint64_t>(s);
  h *= 1099511628211ULL;
  for (const char* p = key; p && *p; ++p) {
    h ^= static_cast<uint64_t>(static_cast<unsigned char>(*p));
    h *= 1099511628211ULL;
  }
  return h;
}

// ─── Crash handler shared state ────────────────────────────────────────────────
constexpr int kMaxThreads = 16;

struct GuestPcSlot {
  std::atomic<uint64_t> pc{0};
  std::atomic<bool> valid{false};
};

GuestPcSlot g_pc_slots[kMaxThreads];
std::atomic<uint32_t> g_pc_slot_next{0};

struct Mapping {
  uintptr_t host_base = 0;
  uint64_t size = 0;
  uint64_t guest_base = 0;
};
Mapping g_mapping;

char g_report_dir[512] = {};
std::atomic<bool> g_installed{false};
std::mutex g_report_mutex;
std::string g_last_report_path;

struct sigaction g_old_actions[6];
const int kSignals[6] = {SIGSEGV, SIGBUS, SIGILL, SIGABRT, SIGFPE, SIGTRAP};

// async-signal-safe integer/hex writers (strtol-free)
void safe_write(int fd, const char* buf, size_t len) {
  size_t done = 0;
  while (done < len) {
    ssize_t n = write(fd, buf + done, len - done);
    if (n <= 0) return;
    done += static_cast<size_t>(n);
  }
}

void safe_puts(int fd, const char* s) {
  safe_write(fd, s, strlen(s));
}

void safe_put_hex(int fd, uint64_t v) {
  char buf[19];
  buf[0] = '0';
  buf[1] = 'x';
  static const char* digits = "0123456789ABCDEF";
  for (int i = 0; i < 16; i++) {
    buf[2 + i] = digits[(v >> (60 - 4 * i)) & 0xF];
  }
  buf[18] = '\0';
  safe_puts(fd, buf);
}

void safe_put_dec(int fd, uint64_t v) {
  char buf[24];
  int i = static_cast<int>(sizeof(buf)) - 1;
  buf[i] = '\0';
  if (v == 0) {
    buf[--i] = '0';
  } else {
    while (v) {
      buf[--i] = static_cast<char>('0' + (v % 10));
      v /= 10;
    }
  }
  safe_puts(fd, buf + i);
}

uint64_t translate_guest(uintptr_t host_addr, bool* ok) {
  *ok = false;
  if (g_mapping.size == 0) return 0;
  if (host_addr >= g_mapping.host_base &&
      host_addr < g_mapping.host_base + g_mapping.size) {
    *ok = true;
    return g_mapping.guest_base + (host_addr - g_mapping.host_base);
  }
  return 0;
}

std::string build_report_name() {
  char name[768];
  if (g_report_dir[0]) {
    snprintf(name, sizeof(name), "%s/crash-%ld.txt", g_report_dir,
             static_cast<long>(time(nullptr)));
  } else {
    snprintf(name, sizeof(name), "/data/local/tmp/x360recomp-crash-%ld.txt",
             static_cast<long>(time(nullptr)));
  }
  return std::string(name);
}

void crash_handler(int sig, siginfo_t* info, void* uctx) {
  const int fd_log = 2;  // stderr shows up in logcat on Android

  safe_puts(fd_log, "\n*** XBOX360RECOMP NATIVE CRASH ***\n  signal: ");
  safe_put_dec(fd_log, static_cast<uint64_t>(sig));
  safe_puts(fd_log, "\n  fault address: ");
  safe_put_hex(fd_log, reinterpret_cast<uintptr_t>(info ? info->si_addr : nullptr));

  bool in_guest = false;
  if (info) {
    uint64_t guest = translate_guest(reinterpret_cast<uintptr_t>(info->si_addr), &in_guest);
    if (in_guest) {
      safe_puts(fd_log, "  (guest address: ");
      safe_put_hex(fd_log, guest);
      safe_puts(fd_log, ")\n");
    } else {
      safe_puts(fd_log, "  (outside guest address space)\n");
    }
  }

  // Highest guest PC seen across threads: the closest thing we have to "where in
  // the game did we die" without a full PPC unwinder.
  uint64_t best_pc = 0;
  for (int i = 0; i < kMaxThreads; i++) {
    if (g_pc_slots[i].valid.load(std::memory_order_relaxed)) {
      const uint64_t pc = g_pc_slots[i].pc.load(std::memory_order_relaxed);
      if (pc) best_pc = pc;
    }
  }
  safe_puts(fd_log, "  last guest PC seen: ");
  safe_put_hex(fd_log, best_pc);
  safe_puts(fd_log, "\n");

#ifdef __aarch64__
  // Dump the ARM64 registers the JIT uses; x28 holds the PPCContext* by ABI
  // convention, so a reader can map guest registers from a core dump.
  if (uctx) {
    auto* uc = static_cast<ucontext_t*>(uctx);
    const uint64_t* regs = reinterpret_cast<const uint64_t*>(&uc->uc_mcontext);
    safe_puts(fd_log, "  host regs (x0..x30,sp,pc):\n   ");
    for (int i = 0; i < 32; i++) {
      safe_put_hex(fd_log, regs[i]);
      safe_puts(fd_log, (i % 4 == 3) ? "\n   " : " ");
    }
  }
#endif

  void* frames[64];
  const int n = backtrace(frames, 64);
  safe_puts(fd_log, "  host backtrace:\n");
  char** syms = backtrace_symbols(frames, n);  // may return null under pressure
  for (int i = 0; i < n; i++) {
    safe_puts(fd_log, "   #");
    safe_put_dec(fd_log, static_cast<uint64_t>(i));
    safe_puts(fd_log, " ");
    if (syms && syms[i]) {
      safe_puts(fd_log, syms[i]);
    } else {
      safe_put_hex(fd_log, reinterpret_cast<uintptr_t>(frames[i]));
    }
    safe_puts(fd_log, "\n");
  }
  if (syms) free(syms);

  // Persist the same information to a file the user can attach to a bug report.
  {
    std::lock_guard<std::mutex> lk(g_report_mutex);
    const std::string path = build_report_name();
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd >= 0) {
      safe_puts(fd, "Xbox360RecompAndroid crash report\n");
      safe_puts(fd, "signal: ");
      safe_put_dec(fd, static_cast<uint64_t>(sig));
      safe_puts(fd, "\nfault: ");
      safe_put_hex(fd, reinterpret_cast<uintptr_t>(info ? info->si_addr : nullptr));
      safe_puts(fd, "\nin_guest_mapping: ");
      safe_puts(fd, in_guest ? "1" : "0");
      if (in_guest && info) {
        safe_puts(fd, "\nguest_addr: ");
        safe_put_hex(fd, translate_guest(reinterpret_cast<uintptr_t>(info->si_addr), &in_guest));
      }
      safe_puts(fd, "\nlast_guest_pc: ");
      safe_put_hex(fd, best_pc);
      safe_puts(fd, "\nhost_frames: ");
      safe_put_dec(fd, static_cast<uint64_t>(n));
      for (int i = 0; i < n; i++) {
        safe_puts(fd, "\n  #");
        safe_put_dec(fd, static_cast<uint64_t>(i));
        safe_puts(fd, " ");
        safe_put_hex(fd, reinterpret_cast<uintptr_t>(frames[i]));
      }
      // The recent log lines are copied with write() from the logger ring.
      static char ring[64 * 1024];
      const size_t len = log::snapshot(ring, sizeof(ring) - 1);
      if (len) {
        safe_puts(fd, "\n--- recent log ---\n");
        safe_write(fd, ring, len);
      }
      close(fd);
      g_last_report_path = path;
      safe_puts(fd_log, "  report written: ");
      safe_puts(fd_log, path.c_str());
      safe_puts(fd_log, "\n");
    }
  }

  safe_puts(fd_log, "*** end of crash report ***\n");

  // Chain to whatever handler was installed before us so the platform still
  // gets its tombstone.
  const int idx = [&]() -> int {
    for (int i = 0; i < 6; i++) {
      if (kSignals[i] == sig) return i;
    }
    return -1;
  }();
  if (idx >= 0 && (g_old_actions[idx].sa_flags & SA_SIGINFO) &&
      g_old_actions[idx].sa_sigaction) {
    g_old_actions[idx].sa_sigaction(sig, info, uctx);
  } else if (idx >= 0 && g_old_actions[idx].sa_handler &&
             g_old_actions[idx].sa_handler != SIG_DFL &&
             g_old_actions[idx].sa_handler != SIG_IGN) {
    g_old_actions[idx].sa_handler(sig);
  } else {
    signal(sig, SIG_DFL);
    raise(sig);
  }
}

}  // namespace

const char* subsystem_name(Subsystem s) {
  switch (s) {
    case Subsystem::Kernel:     return "kernel";
    case Subsystem::Gpu:        return "gpu";
    case Subsystem::FileSystem: return "fs";
    case Subsystem::Audio:      return "audio";
    case Subsystem::Memory:     return "memory";
    case Subsystem::Input:      return "input";
    default:                    return "unknown";
  }
}

bool report_unsupported(Subsystem s, const char* detail) {
  State& st = state();
  if (!st.enabled) return false;
  const uint64_t key = hash_key(s, detail ? detail : "");
  std::lock_guard<std::mutex> lk(st.mutex);
  auto it = st.index.find(key);
  const uint64_t tick = st.tick.fetch_add(1, std::memory_order_relaxed) + 1;
  if (it != st.index.end()) {
    Record& r = st.records[it->second];
    r.hits++;
    r.last_tick = tick;
    return false;
  }
  if (st.records.size() >= kMaxRecords) return false;  // cap memory use
  Record r{};
  r.subsys = s;
  r.hits = 1;
  r.first_tick = tick;
  r.last_tick = tick;
  snprintf(r.key, sizeof(r.key), "%s", detail ? detail : "");
  st.index[key] = static_cast<uint32_t>(st.records.size());
  st.records.push_back(r);
  X360LOGW(TAG, "first %s unsupported: %s", subsystem_name(s), r.key);
  return true;
}

uint32_t unsupported_hits(Subsystem s, const char* detail) {
  State& st = state();
  const uint64_t key = hash_key(s, detail ? detail : "");
  std::lock_guard<std::mutex> lk(st.mutex);
  auto it = st.index.find(key);
  return it == st.index.end() ? 0 : st.records[it->second].hits;
}

uint32_t unsupported_count(Subsystem s) {
  State& st = state();
  std::lock_guard<std::mutex> lk(st.mutex);
  uint32_t n = 0;
  for (const auto& r : st.records) {
    if (r.subsys == s) n++;
  }
  return n;
}

uint32_t format_report(char* out, uint32_t max) {
  if (!out || max == 0) return 0;
  State& st = state();
  std::lock_guard<std::mutex> lk(st.mutex);
  uint32_t used = 0;
  auto emit = [&](const char* fmt, ...) {
    if (used + 1 >= max) return;
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(out + used, max - used, fmt, ap);
    va_end(ap);
    if (n > 0) used += static_cast<uint32_t>(n);
  };
  emit("=== unsupported API report (%zu distinct) ===\n", st.records.size());
  for (Subsystem s : {Subsystem::Kernel, Subsystem::Memory, Subsystem::FileSystem,
                      Subsystem::Audio, Subsystem::Input, Subsystem::Gpu}) {
    uint32_t n = 0;
    for (const auto& r : st.records) {
      if (r.subsys != s) continue;
      if (n == 0) emit("[%s]\n", subsystem_name(s));
      emit("  x%-6u %s\n", r.hits, r.key);
      n++;
    }
    if (n == 0) emit("[%s] none recorded\n", subsystem_name(s));
  }
  emit("=== end ===\n");
  return used;
}

void install_crash_handler(const std::string& report_dir) {
  if (g_installed.exchange(true)) {
    // already installed: just refresh the directory
  }
  snprintf(g_report_dir, sizeof(g_report_dir), "%s", report_dir.c_str());
  struct sigaction sa{};
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = crash_handler;
  for (int i = 0; i < 6; i++) {
    sigaction(kSignals[i], &sa, &g_old_actions[i]);
  }
  sigaltstack_t ss{};
  static thread_local char altstack[SIGSTKSZ];
  ss.ss_sp = altstack;
  ss.ss_size = sizeof(altstack);
  ss.ss_flags = 0;
  sigaltstack(&ss, nullptr);
  X360LOGI(TAG, "crash handler installed (reports in %s)",
            g_report_dir[0] ? g_report_dir : "<stderr only>");
}

void uninstall_crash_handler() {
  if (!g_installed.load()) return;
  for (int i = 0; i < 6; i++) {
    sigaction(kSignals[i], &g_old_actions[i], nullptr);
  }
  g_installed.store(false);
}

void set_guest_pc(void* thread_handle, uint64_t guest_pc) {
  auto* slot = static_cast<GuestPcSlot*>(thread_handle);
  if (!slot) return;
  slot->pc.store(guest_pc, std::memory_order_relaxed);
}

static GuestPcSlot* claim_slot() {
  const uint32_t idx = g_pc_slot_next.fetch_add(1, std::memory_order_relaxed);
  if (idx >= kMaxThreads) return nullptr;
  g_pc_slots[idx].valid.store(true, std::memory_order_relaxed);
  return &g_pc_slots[idx];
}

GuestPcSlot* register_guest_pc_slot() {
  return claim_slot();
}

void set_guest_mapping(uintptr_t host_base, uint64_t size, uint64_t guest_base) {
  g_mapping.host_base = host_base;
  g_mapping.size = size;
  g_mapping.guest_base = guest_base;
}

void write_crash_report(const char* reason, uintptr_t fault_addr, uint64_t guest_pc) {
  X360LOGE(TAG, "unrecoverable: %s (fault=%p guest_pc=0x%llX)", reason ? reason : "?",
            reinterpret_cast<void*>(fault_addr), static_cast<unsigned long long>(guest_pc));
  siginfo_t si{};
  si.si_addr = reinterpret_cast<void*>(fault_addr);
  crash_handler(SIGABRT, &si, nullptr);
}

const std::string& last_report_path() {
  std::lock_guard<std::mutex> lk(g_report_mutex);
  return g_last_report_path;
}

}  // namespace diag
}  // namespace x360
