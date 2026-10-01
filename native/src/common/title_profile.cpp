/**
 * Per-title profile implementation + a deliberately small JSON subset reader.
 */
#include "../../include/x360/title_profile.h"
#include "../../include/x360/log.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#define TAG "X360:TITLE"

namespace x360 {
namespace title {

const char* support_name(Support s) {
  switch (s) {
    case Support::Untested: return "untested";
    case Support::LoadFail: return "loadfail";
    case Support::Noop:     return "nothing";
    case Support::Partial:  return "partial";
    case Support::Playable: return "playable";
  }
  return "untested";
}

bool support_from_name(const char* name, Support* out) {
  if (!name) return false;
  struct { const char* n; Support s; } map[] = {
      {"untested", Support::Untested}, {"loadfail", Support::LoadFail},
      {"nothing", Support::Noop},      {"boots", Support::Partial},
      {"partial", Support::Partial},   {"ingame", Support::Partial},
      {"playable", Support::Playable},
  };
  for (const auto& e : map) {
    if (!strcasecmp(name, e.n)) {
      if (out) *out = e.s;
      return true;
    }
  }
  return false;
}

bool TitleProfile::matches(uint32_t tid) const {
  if (tid == 0) return false;
  if (title_id == tid) return true;
  return std::find(alternate_title_ids.begin(), alternate_title_ids.end(), tid) !=
         alternate_title_ids.end();
}

std::string TitleProfile::describe() const {
  std::ostringstream os;
  if (!valid) {
    os << "generic profile (no title match)";
    return os.str();
  }
  char tidbuf[16];
  snprintf(tidbuf, sizeof(tidbuf), "%08X", title_id);
  os << "profile '" << (name.empty() ? tidbuf : name) << "' tid=" << tidbuf
     << " region=" << region << " exe=" << executable
     << " support=" << support_name(support);
  if (skip_security_validation) os << " skip_security";
  if (ignore_title_updates) os << " ignore_xexp";
  if (!source.empty()) os << " src=" << source;
  return os.str();
}

// ─── Minimal JSON subset ───────────────────────────────────────────────────────
// Supported: one flat object of {"key": string|number|bool|null} and
// {"key": ["string", ...]}. Nested objects are skipped. Anything malformed
// aborts parsing of the *file*, never of the emulator.

namespace {

struct JsonScanner {
  const std::string& s;
  size_t i = 0;
  explicit JsonScanner(const std::string& str) : s(str) {}

  void ws() {
    while (i < s.size() && (isspace(static_cast<unsigned char>(s[i])) || s[i] == '\r' ||
                            s[i] == '\n' || s[i] == '\t')) {
      i++;
    }
  }
  bool eof() {
    ws();
    return i >= s.size();
  }
  bool consume(char c) {
    ws();
    if (i < s.size() && s[i] == c) {
      i++;
      return true;
    }
    return false;
  }
  bool peek(char c) {
    ws();
    return i < s.size() && s[i] == c;
  }
  // Reads a quoted string (unescaping \" \\ \n \t and \uXXXX as best-effort).
  bool readString(std::string* out) {
    ws();
    if (i >= s.size() || s[i] != '"') return false;
    i++;
    out->clear();
    while (i < s.size()) {
      char c = s[i++];
      if (c == '"') return true;
      if (c == '\\') {
        if (i >= s.size()) return false;
        char e = s[i++];
        switch (e) {
          case 'n': out->push_back('\n'); break;
          case 't': out->push_back('\t'); break;
          case 'r': out->push_back('\r'); break;
          case 'b': out->push_back('\b'); break;
          case 'f': out->push_back('\f'); break;
          case '/': out->push_back('/'); break;
          case '"': out->push_back('"'); break;
          case '\\': out->push_back('\\'); break;
          case 'u': {
            if (i + 4 > s.size()) return false;
            unsigned v = 0;
            for (int k = 0; k < 4; k++) {
              char h = s[i++];
              v <<= 4;
              if (h >= '0' && h <= '9') v |= static_cast<unsigned>(h - '0');
              else if (h >= 'a' && h <= 'f') v |= static_cast<unsigned>(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') v |= static_cast<unsigned>(h - 'A' + 10);
              else return false;
            }
            if (v < 0x80) {
              out->push_back(static_cast<char>(v));
            } else if (v < 0x800) {
              out->push_back(static_cast<char>(0xC0 | (v >> 6)));
              out->push_back(static_cast<char>(0x80 | (v & 0x3F)));
            } else {
              out->push_back(static_cast<char>(0xE0 | (v >> 12)));
              out->push_back(static_cast<char>(0x80 | ((v >> 6) & 0x3F)));
              out->push_back(static_cast<char>(0x80 | (v & 0x3F)));
            }
            break;
          }
          default: return false;
        }
      } else {
        out->push_back(c);
      }
    }
    return false;
  }
  // number | true | false | null
  bool readScalar(std::string* raw) {
    ws();
    size_t start = i;
    while (i < s.size() && strchr("+-0123456789.eE", s[i]) != nullptr) i++;
    if (i > start) {
      raw->assign(s, start, i - start);
      return true;
    }
    for (const char* lit : {"true", "false", "null"}) {
      size_t len = strlen(lit);
      if (s.compare(i, len, lit) == 0) {
        raw->assign(lit);
        i += len;
        return true;
      }
    }
    return false;
  }
  void skipValue() {
    ws();
    if (i >= s.size()) return;
    if (s[i] == '"') {
      std::string tmp;
      readString(&tmp);
      return;
    }
    if (s[i] == '{') {
      int depth = 0;
      while (i < s.size()) {
        if (s[i] == '{') depth++;
        else if (s[i] == '}') {
          depth--;
          if (depth == 0) {
            i++;
            return;
          }
        } else if (s[i] == '"') {
          std::string tmp;
          readString(&tmp);
          continue;
        }
        i++;
      }
      return;
    }
    if (s[i] == '[') {
      int depth = 0;
      while (i < s.size()) {
        if (s[i] == '[') depth++;
        else if (s[i] == ']') {
          depth--;
          if (depth == 0) {
            i++;
            return;
          }
        } else if (s[i] == '"') {
          std::string tmp;
          readString(&tmp);
          continue;
        }
        i++;
      }
      return;
    }
    std::string tmp;
    readScalar(&tmp);
  }

  // Reads one array of strings. Assumes '[' was already consumed.
  bool readStringArray(std::vector<std::string>* out) {
    while (!peek(']')) {
      std::string v;
      if (!readString(&v)) return false;
      out->push_back(v);
      if (!consume(',')) break;
    }
    return consume(']');
  }
};

long parse_long(const std::string& v) {
  if (v == "true") return 1;
  if (v == "false" || v == "null") return 0;
  return strtol(v.c_str(), nullptr, 0);  // accepts 0x... too
}

}  // namespace

bool load_profile_from_file(const std::string& path, TitleProfile* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  std::string text = ss.str();

  JsonScanner sc(text);
  if (!sc.consume('{')) {
    X360LOGW(TAG, "profile %s: expected top-level object, ignored", path.c_str());
    return false;
  }

  out->source = path;
  out->valid = true;
  bool any = false;

  while (!sc.peek('}')) {
    std::string key;
    if (!sc.readString(&key)) break;
    if (!sc.consume(':')) break;
    any = true;

    auto set_string = [&](std::string& dst) {
      std::string v;
      if (sc.readString(&v)) dst = v;
      else sc.skipValue();
    };
    auto set_u32 = [&](uint32_t& dst) {
      std::string v;
      if (sc.readScalar(&v)) dst = static_cast<uint32_t>(parse_long(v));
      else sc.skipValue();
    };
    auto set_bool = [&](bool& dst) {
      size_t save = sc.i;
      std::string v;
      if (sc.readScalar(&v)) {
        dst = parse_long(v) != 0;
      } else {
        sc.i = save;
        sc.skipValue();
      }
    };

    if (key == "title_id") {
      std::string v;
      if (sc.readString(&v)) {
        out->title_id = static_cast<uint32_t>(strtoul(v.c_str(), nullptr, 16));
      } else {
        set_u32(out->title_id);
      }
    } else if (key == "name") {
      set_string(out->name);
    } else if (key == "region") {
      set_string(out->region);
    } else if (key == "executable" || key == "xex") {
      set_string(out->executable);
    } else if (key == "search_paths") {
      if (!sc.consume('[')) sc.skipValue();
      else sc.readStringArray(&out->search_paths);
    } else if (key == "content_roots") {
      if (!sc.consume('[')) sc.skipValue();
      else sc.readStringArray(&out->content_roots);
    } else if (key == "alternate_title_ids") {
      std::vector<std::string> tmp;
      if (!sc.consume('[')) {
        sc.skipValue();
      } else {
        sc.readStringArray(&tmp);
      }
      for (auto& t : tmp) out->alternate_title_ids.push_back(static_cast<uint32_t>(strtoul(t.c_str(), nullptr, 16)));
    } else if (key == "skip_security_validation") {
      set_bool(out->skip_security_validation);
    } else if (key == "allow_unencrypted") {
      set_bool(out->allow_unencrypted);
    } else if (key == "ignore_title_updates") {
      set_bool(out->ignore_title_updates);
    } else if (key == "require_imports_resolved") {
      set_bool(out->require_imports_resolved);
    } else if (key == "override_base_address") {
      set_u32(out->override_base_address);
    } else if (key == "title_workspace_size") {
      set_u32(out->title_workspace_size);
    } else if (key == "extra_memory_size") {
      set_u32(out->extra_memory_size);
    } else if (key == "stack_size") {
      set_u32(out->stack_size);
    } else if (key == "heap_size") {
      set_u32(out->heap_size);
    } else if (key == "max_guest_threads") {
      set_u32(out->max_guest_threads);
    } else if (key == "swap_timeout_ms") {
      set_u32(out->swap_timeout_ms);
    } else if (key == "unsupported_log_limit") {
      set_u32(out->unsupported_log_limit);
    } else if (key == "log_unsupported_kernel") {
      set_bool(out->log_unsupported_kernel);
    } else if (key == "log_unsupported_gpu") {
      set_bool(out->log_unsupported_gpu);
    } else if (key == "log_unsupported_fs") {
      set_bool(out->log_unsupported_fs);
    } else if (key == "log_unsupported_audio") {
      set_bool(out->log_unsupported_audio);
    } else if (key == "install_crash_handler") {
      set_bool(out->install_crash_handler);
    } else if (key == "write_log_file") {
      set_bool(out->write_log_file);
    } else if (key == "support") {
      std::string v;
      if (sc.readString(&v)) support_from_name(v.c_str(), &out->support);
      else sc.skipValue();
    } else if (key == "notes" || key == "status_notes") {
      set_string(out->status_notes);
    } else if (key == "known_issues") {
      out->known_issues.clear();
      if (!sc.consume('[')) sc.skipValue();
      else sc.readStringArray(&out->known_issues);
    } else {
      sc.skipValue();
    }
    if (!sc.consume(',')) break;
  }
  sc.consume('}');

  if (!any) {
    X360LOGW(TAG, "profile %s: no recognised keys, ignored", path.c_str());
    out->valid = false;
    return false;
  }
  if (out->search_paths.empty() && !out->executable.empty()) {
    out->search_paths.push_back(out->executable);
  }
  X360LOGI(TAG, "loaded file profile: %s", out->describe().c_str());
  return true;
}

// ─── Built-in profiles ─────────────────────────────────────────────────────────
/*
 * Dragon Ball Xenoverse (Xbox 360) — the title Xbox 360 games actually use is
 * encoded in the XEX execution info as ASCII hex digits "TTRRNNNN":
 *
 *   4E4D0862  "NM-2146"  Dragon Ball Xenoverse  (publisher 4E4D = Namco)
 *             media ids: USA 423344A1 / Europe 3AEC954D / Japan 28E88B7D
 *   4E4D0841  seen in the wild for Xenoverse add-on/compat content
 *   4D530825  NOT valid for this title (that id space is Microsoft)
 *
 * Sources for the id: Xenia's game-compatibility issue for DBXenoverse is filed
 * as "4E4D0862 - Dragon Ball Xenoverse", and the marketplace product id ends in
 * 66acd000-77fe-1000-9115-d8024e4d0862 (the trailing 8 hex digits are the title
 * id). If a dump reports a different id, the boot log prints the detected value
 * and the generic profile applies — nothing here is required for identification
 * to succeed.
 *
 * IMPORTANT: support = Untested. This entry only enables identification, game
 * data-directory handling and diagnostics. It is NOT a claim that the game runs.
 * (See compat/xenoverse.json and docs/xenoverse.md for the honest status.)
 */
const TitleProfile& xenoverse_profile() {
  static TitleProfile p = [] {
    TitleProfile x;
    x.valid = true;
    x.source = "builtin";
    x.title_id = 0x4E4D0862u;
    x.alternate_title_ids = {0x4E4D0841u};
    x.name = "Dragon Ball Xenoverse";
    x.region = "ANY";   // USA / Europe / Japan discs share the title id
    x.executable = "default.xex";
    x.search_paths = {"default.xex", "Xenoverse/default.xex", "DBXenoverse/default.xex"};
    x.content_roots = {"", "content", "TitleData"};
    x.skip_security_validation = true;   // SHA/perm checks are not implemented yet
    x.allow_unencrypted = true;
    x.ignore_title_updates = true;       // title-update (.xexp) support is not implemented
    x.require_imports_resolved = false;  // continue past unknown imports, but log them
    x.stack_size = 0x10000;
    x.max_guest_threads = 6;
    x.swap_timeout_ms = 1000;
    x.unsupported_log_limit = 256;
    x.support = Support::Untested;
    x.status_notes =
        "Runtime supports XEX parsing/decryption and an ARM64 JIT, but the Xenos "
        "PM4->Vulkan pipeline and XAudio2 are incomplete; expect no graphics or "
        "audio even if the executable maps.";
    x.known_issues = {
        "XMA2 audio not implemented",
        "Xenos shader microcode recompiler is a passthrough stub",
        "eDRAM resolve is a stub",
        "XDVDFS (Xbox DVD) reads need a real disc layout check",
    };
    return x;
  }();
  return p;
}

const TitleProfile* builtin_for_title(uint32_t title_id) {
  const TitleProfile& dbx = xenoverse_profile();
  if (dbx.matches(title_id)) return &dbx;
  return nullptr;
}

uint32_t guess_title_id_from_path(const std::string& path) {
  // Accepts ".../5841120D/..." or ".../Xenoverse-5841120D.iso" style names so a
  // profile still applies when the XEX header cannot be read.
  std::string up = path;
  std::transform(up.begin(), up.end(), up.begin(), ::toupper);
  for (size_t i = 0; i + 8 <= up.size(); i++) {
    bool ok = true;
    for (size_t k = 0; k < 8; k++) {
      const char c = up[i + k];
      const bool hex = (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F');
      if (!hex) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    // Must be delimited (start/end or non-alnum neighbours) to avoid matching
    // random words of exactly 8 hex letters.
    const bool left_ok = (i == 0) || (!isalnum(static_cast<unsigned char>(up[i - 1])) && up[i - 1] != '/');
    const bool right_ok = (i + 8 >= up.size()) || (!isalnum(static_cast<unsigned char>(up[i + 8])));
    if (!left_ok || !right_ok) continue;
    const std::string token = up.substr(i, 8);
    // Real title ids start with a publisher prefix we know about.
    const std::string pre = token.substr(0, 4);
    if (pre == "5841" || pre == "4D53" || pre == "4541" || pre == "5453" || pre == "4E4D" ||
        pre == "5041" || pre == "5351" || pre == "4253" || pre == "4E4D") {
      return static_cast<uint32_t>(strtoul(token.c_str(), nullptr, 16));
    }
  }
  return 0;
}

TitleProfile resolve(uint32_t title_id, const std::string& game_data_dir,
                     const std::string& user_data_dir) {
  TitleProfile result;

  // 1) explicit per-directory override
  if (!game_data_dir.empty()) {
    TitleProfile tmp;
    if (load_profile_from_file(game_data_dir + "/title_profile.json", &tmp)) {
      if (title_id == 0 || tmp.title_id == 0 || tmp.matches(title_id)) return tmp;
      X360LOGI(TAG, "game-dir profile is for tid %08X, want %08X — skipping",
               tmp.title_id, title_id);
    }
  }

  // 2) per-title override in user data
  if (title_id != 0 && !user_data_dir.empty()) {
    char name[32];
    snprintf(name, sizeof(name), "%08X", title_id);
    TitleProfile tmp;
    if (load_profile_from_file(user_data_dir + "/titles/" + name + ".json", &tmp)) {
      return tmp;
    }
  }

  // 3) built-in
  if (title_id != 0) {
    const TitleProfile* b = builtin_for_title(title_id);
    if (b) return *b;
  }

  // 4) generic
  result.valid = false;
  result.title_id = title_id;
  result.source = "default";
  result.search_paths = {result.executable};
  return result;
}

}  // namespace title
}  // namespace x360
