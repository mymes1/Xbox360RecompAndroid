/**
 * Xbox 360 kernel export ordinal tables (xboxkrnl.exe / xam.xex).
 *
 * WHY THIS EXISTS
 *   Before this, the HLE dispatchers switched on *invented* ordinal numbers, so
 *   a guest import could never be routed correctly, and an unimplemented import
 *   could not be named in a log. The tables below give the runtime a trustworthy
 *   ordinal -> name map for every documented export, which is what makes
 *   "first unsupported kernel call" a *log line* instead of a guess.
 *
 * The ordinals are the publicly documented kernel export ordinals (the same
 * numbers Xenia's xboxkrnl_table.inc / xam_table.inc list). Only names/ordinals
 * are recorded here; the implementations live in the hle sources.
 *
 * NOTE on XInput: on Xbox 360 the input entry points are the Xam* family
 * (XamInputGetState / XamInputSetState / XamInputGetCapabilities). Titles that
 * reference an "XInput*" name resolve through those; there is no separate
 * xboxkrnl XInputGetState export in the documented tables.
 *
 * GENERATED FILE - do not edit by hand. Regenerate with
 *   tools/gen_kernel_ordinals.py
 */
#pragma once
#include <cstdint>
#include <string>

namespace x360 {
namespace kernel {

struct OrdinalEntry {
  uint32_t ordinal;
  const char* name;
  bool isVariable;
};

// Canonical module names as they appear in the XEX import string table.
bool is_kernel_module(const char* name);           // "xboxkrnl.exe" / "xam.xex"
const char* name_for_ordinal(const std::string& module, uint32_t ordinal);
bool ordinal_for_name(const char* module, const char* name, uint32_t* out);
bool is_implemented(const std::string& module, uint32_t ordinal);
// Called by the HLE layer when it registers a handler so the "implemented" set
// always reflects reality rather than a hand-maintained list.
void mark_implemented(const char* module, uint32_t ordinal);

uint32_t xboxkrnl_export_count();
uint32_t xam_export_count();

}  // namespace kernel
}  // namespace x360
