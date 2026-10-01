/**
 * Unified logging + runtime diagnostics ring buffer.
 *
 * Why this exists:
 *  - Every subsystem used its own `#define LOG_TAG` + `__android_log_print`, so
 *    there was no way to (a) capture what happened when a title fails to boot,
 *    or (b) build any native file for host testing (the unit tests and the
 *    `x360tool` offline analyser both need a non-Android logging path).
 *  - Per-title (Xenoverse) testing needs "first unsupported kernel / GPU / FS /
 *    audio call" to be recoverable *after* a crash, not only in logcat.
 *
 * The ring buffer is lock-free-ish (single atomic write index + fixed slots) and
 * is intentionally tiny: it must survive a SIGSEGV, so no allocation happens on
 * the logging path.
 *
 * Behaviour is generic: nothing here is Xenoverse-specific. Titles opt into
 * file logging via x360_log_set_file() (see diagnostics.cpp).
 */
#pragma once
#include <cstdarg>
#include <cstddef>
#include <cstdint>

namespace x360 {
namespace log {

enum class Level : int {
  Verbose = 2,
  Debug   = 3,
  Info    = 4,
  Warn    = 5,
  Error   = 6,
  Fatal   = 7,
};

// Filter: messages below this level are dropped (defaults to Info, matching the
// previous hard-coded behaviour of every subsystem).
void set_min_level(Level level);
Level min_level();

// Optional persistent log file (used by Xenoverse bring-up: the game dies in
// native code long before logcat can be read on a tablet). Passing nullptr
// disables file logging. Returns false if the file could not be opened.
bool set_file(const char* path);

// Enable/disable capture of the most recent messages into the ring buffer that
// crash diagnostics dump. Always on; kept as a knob for performance experiments.
void set_ring_enabled(bool enabled);

// Ring buffer access (used by diagnostics.cpp when writing a crash report).
// Writes up to `max` bytes into `out`, returns the number of bytes written.
size_t snapshot(char* out, size_t max);

// Number of messages dropped because of the level filter (diagnostics output).
uint64_t dropped_count();

// Internal: called by the X360LOG* macros. Not for direct use.
void write(Level level, const char* tag, const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 3, 4)))
#endif
    ;

}  // namespace log
}  // namespace x360

/*
 * Macros. Existing code used LOGI/LOGE/LOGW defined per translation unit; those
 * definitions are kept (they now forward here) so that untouched subsystems keep
 * compiling, and new code should use X360LOG*.
 */
#ifdef __ANDROID__
#include <android/log.h>
#define X360_LOG_IMPL(lvl, tag, ...) \
  ::x360::log::write((lvl), (tag), __VA_ARGS__), \
      __android_log_print(static_cast<int>(lvl), (tag), __VA_ARGS__)
#else
#define X360_LOG_IMPL(lvl, tag, ...) ::x360::log::write((lvl), (tag), __VA_ARGS__)
#endif

#define X360LOGV(tag, ...) X360_LOG_IMPL(::x360::log::Level::Verbose, tag, __VA_ARGS__)
#define X360LOGD(tag, ...) X360_LOG_IMPL(::x360::log::Level::Debug, tag, __VA_ARGS__)
#define X360LOGI(tag, ...) X360_LOG_IMPL(::x360::log::Level::Info, tag, __VA_ARGS__)
#define X360LOGW(tag, ...) X360_LOG_IMPL(::x360::log::Level::Warn, tag, __VA_ARGS__)
#define X360LOGE(tag, ...) X360_LOG_IMPL(::x360::log::Level::Error, tag, __VA_ARGS__)
