/**
 * Per-title profiles (a.k.a. "compat configuration").
 *
 * This is the *only* place where title-specific behaviour may live: the generic
 * runtime (loader / JIT / HLE / GPU) must keep asking "what does the profile
 * say?" instead of hard-coding `if (title == Xenoverse)`.
 *
 * Resolution order (first hit wins):
 *   1. <gameDataDir>/title_profile.json     — user override, per game directory
 *   2. <userDataDir>/titles/<TID>.json      — per-title override shipped by the user
 *   3. built-in table (built-in: Dragon Ball Xenoverse)
 *   4. generic defaults
 *
 * The parser is intentionally tiny: flat objects, string/number/bool values,
 * plus the handful of string arrays we need. Anything it does not understand is
 * ignored (and logged), never fatal — a typo in a user's compat file must not
 * brick an otherwise booting game.
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace x360 {
namespace title {

// What the runtime actually supports for a title today. This is *reported* to
// the UI and the compatibility database; it must never be used to fake support.
enum class Support : uint8_t {
  Untested = 0,   // never executed on real hardware with this runtime
  LoadFail = 1,   // executable cannot even be parsed/mapped
  Noop = 2,       // loads, but required services are missing
  Partial = 3,    // some subsystems functional (e.g. boots to a black screen)
  Playable = 4,   // full game loop verified by a human
};

const char* support_name(Support s);
bool support_from_name(const char* name, Support* out);

struct TitleProfile {
  bool valid = false;             // false => generic defaults, no title matched
  std::string source;             // where the profile came from (for logs)

  uint32_t title_id = 0;          // 0xTTRRNNNN as stored in the XEX execution info
  std::vector<uint32_t> alternate_title_ids;
  std::string name;               // display name
  std::string region = "ANY";     // NTSC-U / NTSC-J / PAL / ANY

  // ── Discovery ───────────────────────────────────────────────────────────────
  std::string executable = "default.xex";   // file to boot inside the game dir / ISO
  std::vector<std::string> search_paths;    // e.g. "default.xex", "Xenoverse/default.xex"
  std::vector<std::string> content_roots;   // extra dir names that may hold game data

  // ── Loader / security workarounds (all default OFF) ─────────────────────────
  bool skip_security_validation = false;    // do not fail on digest/SHA mismatches
  bool allow_unencrypted = true;            // accept XEX2 without AES (dev/loose dumps)
  bool ignore_title_updates = true;         // never silently apply .xexp patch files
  bool require_imports_resolved = false;    // fail load if an import is unknown

  // ── Memory layout hints ─────────────────────────────────────────────────────
  uint32_t override_base_address = 0;       // 0 = use value from the XEX
  uint32_t title_workspace_size = 0;        // XEX_HEADER_TITLE_WORKSPACE_SIZE fallback
  uint32_t extra_memory_size = 0;           // XEX_HEADER_ADDITIONAL_TITLE_MEMORY fallback
  uint32_t stack_size = 65536;              // per-thread guest stack
  uint32_t heap_size = 0;                   // 0 = let the allocator choose

  // ── Runtime behaviour ───────────────────────────────────────────────────────
  uint32_t max_guest_threads = 6;           // Xenon has 6 hardware threads
  uint32_t swap_timeout_ms = 1000;          // 0 = never present without a GPU swap
  uint32_t unsupported_log_limit = 64;      // per subsystem "first N" logging cap
  bool log_unsupported_kernel = true;
  bool log_unsupported_gpu = true;
  bool log_unsupported_fs = true;
  bool log_unsupported_audio = true;
  bool install_crash_handler = true;
  bool write_log_file = true;
  Support support = Support::Untested;
  std::string status_notes;                 // free-form, shown in the UI
  std::vector<std::string> known_issues;

  // True when `tid` matches this profile's title id or one of its alternates.
  bool matches(uint32_t tid) const;

  // Human readable one-liner for the boot banner.
  std::string describe() const;
};

// Built-in profiles. Dragon Ball Xenoverse is the only entry today; adding one
// here does NOT mean the game works — see `support`.
const TitleProfile* builtin_for_title(uint32_t title_id);
const TitleProfile& xenoverse_profile();

// Load <dir>/title_profile.json, or nullptr if absent/invalid.
bool load_profile_from_file(const std::string& path, TitleProfile* out);

// Full resolution: file overrides -> built-in table -> generic default.
TitleProfile resolve(uint32_t title_id, const std::string& game_data_dir,
                     const std::string& user_data_dir);

// Fallback title id computed from the disc/XEX when the header cannot be read,
// so the profile system still has something to key on. Returns 0 if unknown.
uint32_t guess_title_id_from_path(const std::string& path);

}  // namespace title
}  // namespace x360
