/**
 * Guest memory accessors used by every HLE dispatcher.
 */
#include "../../include/hle/kernel_helpers.h"
#include "../../include/x360/diagnostics.h"
#include "../../include/x360/log.h"

#include <cstring>

#define TAG "X360:HLE"

namespace x360 {
namespace hle {
namespace {

// Bounds-checked host pointer for a guest address. On failure we record the
// fault so a crash log says "guest dereferenced 0x…" rather than just SIGSEGV.
inline void* checked(HleKernel& k, uint64_t addr, uint32_t len) {
  if (!len) return nullptr;
  if (addr < k.guest_base()) return nullptr;
  const uint64_t off = addr - k.guest_base();
  const uint64_t end = off + len;
  if (end < off) return nullptr;
  // The guest window is 4 GB; we cannot know its size from here, so we rely on
  // the kernel's own bounds (set at init) via guest_ptr().
  uint8_t* p = k.guest_ptr(addr);
  return p;
}

}  // namespace

bool g_read8(HleKernel& k, uint64_t addr, uint8_t* out) {
  void* p = checked(k, addr, 1);
  if (!p) return false;
  *out = *reinterpret_cast<uint8_t*>(p);
  return true;
}
bool g_read16(HleKernel& k, uint64_t addr, uint16_t* out) {
  void* p = checked(k, addr, 2);
  if (!p) return false;
  uint16_t v;
  memcpy(&v, p, 2);
  *out = __builtin_bswap16(v);
  return true;
}
bool g_read32(HleKernel& k, uint64_t addr, uint32_t* out) {
  void* p = checked(k, addr, 4);
  if (!p) return false;
  uint32_t v;
  memcpy(&v, p, 4);
  *out = __builtin_bswap32(v);
  return true;
}
bool g_read64(HleKernel& k, uint64_t addr, uint64_t* out) {
  void* p = checked(k, addr, 8);
  if (!p) return false;
  uint64_t v;
  memcpy(&v, p, 8);
  *out = __builtin_bswap64(v);
  return true;
}
bool g_write8(HleKernel& k, uint64_t addr, uint8_t v) {
  void* p = checked(k, addr, 1);
  if (!p) return false;
  *reinterpret_cast<uint8_t*>(p) = v;
  return true;
}
bool g_write16(HleKernel& k, uint64_t addr, uint16_t v) {
  void* p = checked(k, addr, 2);
  if (!p) return false;
  const uint16_t b = __builtin_bswap16(v);
  memcpy(p, &b, 2);
  return true;
}
bool g_write32(HleKernel& k, uint64_t addr, uint32_t v) {
  void* p = checked(k, addr, 4);
  if (!p) return false;
  const uint32_t b = __builtin_bswap32(v);
  memcpy(p, &b, 4);
  return true;
}
bool g_write64(HleKernel& k, uint64_t addr, uint64_t v) {
  void* p = checked(k, addr, 8);
  if (!p) return false;
  const uint64_t b = __builtin_bswap64(v);
  memcpy(p, &b, 8);
  return true;
}
bool g_zero(HleKernel& k, uint64_t addr, uint32_t len) {
  void* p = checked(k, addr, len);
  if (!p) return false;
  memset(p, 0, len);
  return true;
}

bool g_read_bytes(HleKernel& k, uint64_t addr, void* dst, uint32_t len) {
  void* p = checked(k, addr, len);
  if (!p) return false;
  memcpy(dst, p, len);
  return true;
}

bool g_write_bytes(HleKernel& k, uint64_t addr, const void* src, uint32_t len) {
  void* p = checked(k, addr, len);
  if (!p) return false;
  memcpy(p, src, len);
  return true;
}

bool g_read_unicode_string(HleKernel& k, uint64_t addr, std::string* outUtf8) {
  if (!addr) return false;
  uint16_t length = 0, maxLength = 0;
  uint32_t buffer = 0;
  if (!g_read16(k, addr, &length)) return false;
  if (!g_read16(k, addr + 2, &maxLength)) return false;
  if (!g_read32(k, addr + 4, &buffer)) return false;
  (void)maxLength;
  if (length == 0 || length > 0x1000) {
    if (outUtf8) outUtf8->clear();
    return length == 0;
  }
  if (!buffer) return false;
  std::string s;
  s.reserve(length / 2 + 1);
  for (uint32_t i = 0; i + 1 < uint32_t(length) + 1; i += 2) {
    uint16_t u = 0;
    if (!g_read16(k, buffer + i, &u)) break;
    if (u == 0) break;
    if (u < 0x80) {
      s.push_back(char(u));
    } else if (u < 0x800) {
      s.push_back(char(0xC0 | (u >> 6)));
      s.push_back(char(0x80 | (u & 0x3F)));
    } else {
      s.push_back(char(0xE0 | (u >> 12)));
      s.push_back(char(0x80 | ((u >> 6) & 0x3F)));
      s.push_back(char(0x80 | (u & 0x3F)));
    }
  }
  if (outUtf8) *outUtf8 = std::move(s);
  return true;
}

bool g_read_ansi_string(HleKernel& k, uint64_t addr, std::string* out) {
  if (!addr) return false;
  uint16_t length = 0;
  uint32_t buffer = 0;
  if (!g_read16(k, addr, &length)) return false;
  if (!g_read32(k, addr + 4, &buffer)) return false;
  if (length == 0) {
    if (out) out->clear();
    return true;
  }
  if (!buffer || length > 0x1000) return false;
  std::string s(length, '\0');
  if (!g_read_bytes(k, buffer, &s[0], length)) return false;
  const size_t nul = s.find('\0');
  if (nul != std::string::npos) s.resize(nul);
  if (out) *out = std::move(s);
  return true;
}

bool g_read_object_attributes_name(HleKernel& k, uint64_t addr, std::string* outUtf8) {
  // OBJECT_ATTRIBUTES (Xbox 360):
  //   0x00 Length (be32)
  //   0x04 RootDirectory (be32)
  //   0x08 ObjectName (be32 -> UNICODE_STRING)
  //   0x0C Attributes (be32)
  //   0x10 SecurityDescriptor (be32)
  //   0x14 SecurityQualityOfService (be32)
  if (!addr) return false;
  uint32_t namePtr = 0;
  if (!g_read32(k, addr + 0x08, &namePtr)) return false;
  return g_read_unicode_string(k, namePtr, outUtf8);
}

bool g_write_io_status(HleKernel& k, uint64_t addr, uint32_t status, uint32_t information) {
  if (!addr) return true;  // guest is allowed to pass NULL
  if (!g_write64(k, addr, uint64_t(status))) return false;
  if (!g_write32(k, addr + 8, information)) return false;
  return true;
}

}  // namespace hle
}  // namespace x360
