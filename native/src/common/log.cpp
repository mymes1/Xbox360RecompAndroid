/**
 * Implementation of the shared logger + ring buffer (see include/x360/log.h).
 */
#include "../../include/x360/log.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sys/time.h>
#include <unistd.h>

namespace x360 {
namespace log {
namespace {

constexpr size_t kSlotSize = 256;
constexpr size_t kRingSlots = 2048;

struct Ring {
  char slots[kRingSlots][kSlotSize];
  std::atomic<uint64_t> seq{0};
  std::atomic<bool> enabled{true};
};

Ring& ring() {
  static Ring r;
  return r;
}

std::atomic<int> g_min_level{static_cast<int>(Level::Info)};
std::atomic<uint64_t> g_dropped{0};
FILE* g_file = nullptr;
std::mutex g_file_mutex;

const char* level_char(Level l) {
  switch (l) {
    case Level::Verbose: return "V";
    case Level::Debug:   return "D";
    case Level::Info:    return "I";
    case Level::Warn:    return "W";
    case Level::Error:   return "E";
    default:             return "F";
  }
}

bool enabled(Level level) {
  return static_cast<int>(level) >= g_min_level.load(std::memory_order_relaxed);
}

void append_file(const char* line) {
  std::lock_guard<std::mutex> lk(g_file_mutex);
  if (g_file) {
    fputs(line, g_file);
    // Flushing keeps the tail useful when the process dies without unwinding.
    fflush(g_file);
  }
}

}  // namespace

void set_min_level(Level level) {
  g_min_level.store(static_cast<int>(level), std::memory_order_relaxed);
}

Level min_level() {
  return static_cast<Level>(g_min_level.load(std::memory_order_relaxed));
}

bool set_file(const char* path) {
  std::lock_guard<std::mutex> lk(g_file_mutex);
  if (g_file) {
    fclose(g_file);
    g_file = nullptr;
  }
  if (!path || !*path) return true;  // disabling is always fine
  g_file = fopen(path, "a");
  return g_file != nullptr;
}

void set_ring_enabled(bool enabled_flag) {
  ring().enabled.store(enabled_flag, std::memory_order_relaxed);
}

void write(Level level, const char* tag, const char* fmt, ...) {
  if (!enabled(level)) {
    g_dropped.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  char line[kSlotSize];
  int n = 0;

  struct timeval tv{};
  gettimeofday(&tv, nullptr);
  struct tm tmv{};
  localtime_r(&tv.tv_sec, &tmv);

  n = snprintf(line, sizeof(line), "%02d:%02d:%02d.%03ld %s/%s: ", tmv.tm_hour,
               tmv.tm_min, tmv.tm_sec, static_cast<long>(tv.tv_usec / 1000),
               level_char(level), tag ? tag : "?");
  if (n < 0) n = 0;
  if (static_cast<size_t>(n) < sizeof(line) - 1) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line + n, sizeof(line) - static_cast<size_t>(n), fmt, ap);
    va_end(ap);
  }

  if (ring().enabled.load(std::memory_order_relaxed)) {
    Ring& r = ring();
    const uint64_t seq = r.seq.fetch_add(1, std::memory_order_relaxed);
    memcpy(r.slots[seq % kRingSlots], line, strlen(line) + 1);
  }

  append_file(line);

#ifndef __ANDROID__
  // Host-side tools (unit tests, x360tool) and device console output.
  fprintf(stderr, "%s\n", line);
#endif
}

size_t snapshot(char* out, size_t max) {
  if (!out || max == 0) return 0;
  Ring& r = ring();
  const uint64_t total = r.seq.load(std::memory_order_relaxed);
  const uint64_t count = total < kRingSlots ? total : kRingSlots;
  const uint64_t start = total - count;

  size_t used = 0;
  out[0] = '\0';
  for (uint64_t i = 0; i < count; i++) {
    const char* slot = r.slots[(start + i) % kRingSlots];
    const size_t len = strlen(slot);
    if (used + len + 1 >= max) break;
    memcpy(out + used, slot, len);
    used += len;
    out[used++] = '\n';
    out[used] = '\0';
  }
  return used;
}

uint64_t dropped_count() {
  return g_dropped.load(std::memory_order_relaxed);
}

}  // namespace log
}  // namespace x360
