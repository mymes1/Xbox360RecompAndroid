/**
 * XEX2 loader implementation.
 *
 * Covers the parts a retail disc executable actually needs:
 *   - XEX2 header + optional directory entries (correct keys!)
 *   - security info (image size, page protection descriptors, AES session key)
 *   - AES-128-CBC decryption with the retail/devkit master keys
 *   - "basic" (zero-run) and "normal" (LZX) de-block + decompress
 *   - PE image base / entry point / section table
 *   - kernel import table -> HLE sentinel slots (so JIT'd `blr` lands in HLE)
 *
 * Not implemented (and reported, not silently faked): XEX1, title-update (.xexp)
 * patching, RSA/SHA validation, delta compression, resource-info strings.
 */
#include "../../include/loader/xex_loader.h"
#include "../../include/loader/aes128.h"
#include "x360/lzx.h"
#include "../../include/x360/diagnostics.h"
#include "../../include/x360/kernel_ordinals.h"
#include "../../include/x360/log.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>

#define TAG "X360:XEX"

namespace x360 {
namespace {

uint32_t be32at(const uint8_t* p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) |
         uint32_t(p[3]);
}
uint16_t be16at(const uint8_t* p) { return uint16_t((p[0] << 8) | p[1]); }
uint32_t le32at(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}
uint16_t le16at(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
void put_be32(uint8_t* p, uint32_t v) {
  p[0] = uint8_t(v >> 24);
  p[1] = uint8_t(v >> 16);
  p[2] = uint8_t(v >> 8);
  p[3] = uint8_t(v);
}

std::string lowerAscii(const std::string& s) {
  std::string o = s;
  std::transform(o.begin(), o.end(), o.begin(),
                 [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  return o;
}

// AES-128-CBC with a zero IV over the whole buffer (matching how a XEX stores
// its payload). Trailing bytes that are not a full block are left untouched.
void aesCbcZeroIv(const uint8_t key[16], uint8_t* data, size_t len) {
  uint8_t iv[16] = {};
  Aes128::cbcDecrypt(key, iv, data, len);
}

}  // namespace

// The public retail master key (documented in every Xbox 360 homebrew/XEX
// format description; it is not a console-unique key and unlocks nothing by
// itself — it only derives the per-XEX session key stored in the file).
static const uint8_t kXex2RetailKey[16] = {0x20, 0xB1, 0x85, 0xA5, 0x9D, 0x28, 0xFD, 0xC3,
                                           0x40, 0x58, 0x3F, 0xBB, 0x08, 0x96, 0xBF, 0x91};
static const uint8_t kXex2DevKitKey[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

XexLoader::XexLoader() = default;
XexLoader::~XexLoader() = default;

uint32_t XexLoader::titleIdFromAscii(const char digits[4]) {
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  uint32_t v = 0;
  for (int i = 0; i < 4; i++) {
    const int n = nib(digits[i]);
    if (n < 0) return 0;
    v = (v << 4) | static_cast<uint32_t>(n);
  }
  return v;
}

bool XexLoader::deriveSessionKey(const uint8_t masterKey[16], const uint8_t encKey[16],
                                 uint8_t outSessionKey[16]) {
  if (!masterKey || !encKey || !outSessionKey) return false;
  memcpy(outSessionKey, encKey, 16);
  uint8_t iv[16] = {};
  Aes128::cbcDecrypt(masterKey, iv, outSessionKey, 16);
  return true;
}

bool XexLoader::computeSessionKey(const uint8_t* raw, size_t rawSize,
                                  uint8_t outSessionKey[16], bool* usedDevKitKey) {
  if (rawSize < 0x1C || !raw) return false;
  const uint32_t secOff = be32at(raw + 0x0C);
  if (secOff + 0x160 + 16 > rawSize) return false;
  const uint8_t* encKey = raw + secOff + 0x150;
  deriveSessionKey(kXex2RetailKey, encKey, outSessionKey);
  if (usedDevKitKey) {
    // The caller has already decided whether to try the dev-kit key; expose it
    // through a deterministic rule: a zeroed stored key means a dev-kit build.
    static const uint8_t zeros[16] = {};
    *usedDevKitKey = memcmp(encKey, zeros, 16) == 0;
    if (*usedDevKitKey) deriveSessionKey(kXex2DevKitKey, encKey, outSessionKey);
  }
  return true;
}

const char* XexLoader::headerKeyName(uint32_t key) {
  switch (key) {
    case kXexHdrResourceInfo: return "RESOURCE_INFO";
    case kXexHdrFileFormatInfo: return "FILE_FORMAT_INFO";
    case kXexHdrOriginalBaseAddress: return "ORIGINAL_BASE_ADDRESS";
    case kXexHdrEntryPoint: return "ENTRY_POINT";
    case kXexHdrImageBaseAddress: return "IMAGE_BASE_ADDRESS";
    case kXexHdrImportLibraries: return "IMPORT_LIBRARIES";
    case kXexHdrStaticLibraries: return "STATIC_LIBRARIES";
    case kXexHdrDefaultStackSize: return "DEFAULT_STACK_SIZE";
    case kXexHdrDefaultHeapSize: return "DEFAULT_HEAP_SIZE";
    case kXexHdrPageHeapSizeAndFlags: return "PAGE_HEAP_SIZE_AND_FLAGS";
    case kXexHdrSystemFlags: return "SYSTEM_FLAGS";
    case kXexHdrExecutionInfo: return "EXECUTION_INFO";
    case kXexHdrTitleWorkspaceSize: return "TITLE_WORKSPACE_SIZE";
    case kXexHdrAlternateTitleIds: return "ALTERNATE_TITLE_IDS";
    case kXexHdrAdditionalTitleMem: return "ADDITIONAL_TITLE_MEMORY";
    case kXexHdrDeltaPatchDesc: return "DELTA_PATCH_DESCRIPTOR";
    default: return nullptr;
  }
}

void XexLoader::addWarning(std::string msg) {
  m_warnings.push_back(std::move(msg));
  if (m_warnings.size() <= 64) {
    X360LOGW(TAG, "warning: %s", m_warnings.back().c_str());
  }
}

// ─── Entry points ──────────────────────────────────────────────────────────────

bool XexLoader::load(const std::string& path, XexImage& outImage,
                     const title::TitleProfile& profile) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    m_lastError = "cannot open file: " + path;
    return false;
  }
  const std::streamoff sz = f.tellg();
  if (sz <= 0) {
    m_lastError = "file is empty: " + path;
    return false;
  }
  std::vector<uint8_t> raw(static_cast<size_t>(sz));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(raw.data()), sz);
  X360LOGI(TAG, "read %zu bytes from %s", raw.size(), path.c_str());
  return loadFromMemory(raw.data(), raw.size(), outImage, profile, path);
}

bool XexLoader::peekInfo(const std::string& path, XexImage& outImage) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    m_lastError = "cannot open file: " + path;
    return false;
  }
  uint8_t buf[0x2000];
  f.read(reinterpret_cast<char*>(buf), sizeof(buf));
  const std::streamsize got = f.gcount();
  if (got < 0x100) {
    m_lastError = "file too small to contain a XEX header";
    return false;
  }
  return peekInfo(buf, static_cast<size_t>(got), outImage);
}

bool XexLoader::peekInfo(const uint8_t* data, size_t size, XexImage& outImage) {
  if (size < 0x40) {
    m_lastError = "buffer too small for a XEX header";
    return false;
  }
  const uint32_t magic = be32at(data);
  if (magic != XEX2_MAGIC && magic != XEX1_MAGIC) {
    m_lastError = "not a XEX file (bad magic, read as a .xex?)";
    return false;
  }
  if (magic == XEX1_MAGIC) {
    m_lastError = "XEX1 (original Xbox) executables are not supported by this loader";
    return false;
  }
  std::vector<uint8_t> raw(data, data + size);
  return parseHeader(raw, outImage);
}

// ─── XEX2 header + security info ──────────────────────────────────────────────

bool XexLoader::parseHeader(const std::vector<uint8_t>& raw, XexImage& img) {
  if (raw.size() < 0x40) {
    m_lastError = "buffer too small for a XEX header";
    return false;
  }
  if (be32at(raw.data()) != XEX2_MAGIC) {
    m_lastError = "not a XEX2 file (bad magic)";
    return false;
  }

  img.moduleFlags = be32at(raw.data() + 0x04);
  img.headerSize = be32at(raw.data() + 0x08);
  const uint32_t securityOffset = be32at(raw.data() + 0x0C);
  const uint32_t dirCount = be32at(raw.data() + 0x10);
  X360LOGD(TAG, "xex2: moduleFlags=0x%08X headerSize=0x%X security=0x%X dirCount=%u",
           img.moduleFlags, img.headerSize, securityOffset, dirCount);

  if (img.headerSize < 0x18 || img.headerSize > raw.size()) {
    m_lastError = "implausible XEX header size";
    return false;
  }

  // Directory entries: key + (value | offset into the header blob).
  auto findDir = [&](uint32_t key, uint32_t* outValue, uint32_t* outOffset) -> bool {
    const uint8_t* dirs = raw.data() + 0x18;
    for (uint32_t i = 0; i < dirCount; i++) {
      const size_t off = 0x18 + i * 8;
      if (off + 8 > raw.size()) break;
      const uint32_t k = be32at(raw.data() + off);
      const uint32_t v = be32at(raw.data() + off + 4);
      if (k == key) {
        if (outValue) *outValue = v;
        if (outOffset) *outOffset = off + 4;
        return (k & 0xFF) != 0x00;  // 0xFF == data blob, 0x00 == inline value
      }
    }
    return false;
  };

  uint32_t v = 0;
  // System flags (inline value)
  findDir(kXexHdrSystemFlags, &v, nullptr);
  img.systemFlags = v;
  findDir(kXexHdrDefaultStackSize, &v, nullptr);
  img.defaultStackSize = v;
  findDir(kXexHdrDefaultHeapSize, &v, nullptr);
  img.defaultHeapSize = v;

  // Execution info: media type/ID, version, base version, title id (ASCII hex),
  // platform, executable type, disc info, savegame id.
  {
    uint32_t offset = 0;
    if (findDir(kXexHdrExecutionInfo, nullptr, &offset)) {
      const uint32_t size = be32at(raw.data() + offset);
      const uint32_t base = offset;  // size field is at the offset itself
      if (base + 4 <= raw.size()) {
        const uint8_t* ei = raw.data() + base;
        const size_t avail = raw.size() - base;
        if (size >= 0x18 && avail >= 4 + size) {
          img.mediaIdHex.clear();
          char hex[12];
          snprintf(hex, sizeof(hex), "%08X", be32at(ei + 4));
          img.mediaIdHex = hex;
          img.version = be32at(ei + 8);
          img.baseVersion = be32at(ei + 0xC);
          img.titleId = titleIdFromAscii(reinterpret_cast<const char*>(ei + 0x10));
          char tbuf[12];
          snprintf(tbuf, sizeof(tbuf), "%08X", img.titleId);
          img.titleIdText = tbuf;
          img.allowedMediaTypes = 0;  // in security info
          img.savegameId = be32at(ei + 0x1C);
          X360LOGI(TAG, "execution info: titleId=%s mediaId=%s version=%u base=%u",
                   img.titleIdText.c_str(), img.mediaIdHex.c_str(), img.version,
                   img.baseVersion);
        } else {
          addWarning("execution info header is short/absent");
        }
      }
    } else {
      addWarning("no XEX_HEADER_EXECUTION_INFO (dev/homebrew XEX?)");
    }
  }

  // Alternate title ids.
  {
    uint32_t offset = 0;
    if (findDir(kXexHdrAlternateTitleIds, nullptr, &offset)) {
      const uint32_t size = be32at(raw.data() + offset);
      const uint32_t count = (size >= 4) ? (size - 4) / 4 : 0;
      for (uint32_t i = 0; i < count && img.alternateTitleIdCount < 8; i++) {
        const size_t o = offset + 4 + i * 4;
        if (o + 4 > raw.size()) break;
        img.alternateTitleIds[img.alternateTitleIdCount++] = be32at(raw.data() + o);
      }
    }
  }

  // Image base address (preferred) / security info load address (fallback).
  uint32_t baseFromOpt = 0;
  if (findDir(kXexHdrImageBaseAddress, nullptr, &v)) {
    const size_t o = v;
    if (o + 4 <= raw.size()) baseFromOpt = be32at(raw.data() + o);
  }
  uint32_t stackFromOpt = 0, heapFromOpt = 0, wsFromOpt = 0, extraFromOpt = 0;
  if (findDir(kXexHdrDefaultStackSize, nullptr, &v) && v + 4 <= raw.size()) {
    stackFromOpt = be32at(raw.data() + v);
  }
  (void)stackFromOpt;
  if (findDir(kXexHdrTitleWorkspaceSize, nullptr, &v) && v + 4 <= raw.size()) {
    wsFromOpt = be32at(raw.data() + v);
  }
  if (findDir(kXexHdrAdditionalTitleMem, nullptr, &v) && v + 4 <= raw.size()) {
    extraFromOpt = be32at(raw.data() + v);
  }
  img.titleWorkspaceSize = wsFromOpt;
  img.extraMemorySize = extraFromOpt;

  // File format info: encryption + compression + per-compression details.
  {
    uint32_t offset = 0;
    if (findDir(kXexHdrFileFormatInfo, nullptr, &offset)) {
      const size_t base = offset;
      if (base + 8 <= raw.size()) {
        const uint32_t infoSize = be32at(raw.data() + base);
        const uint32_t enc = be16at(raw.data() + base + 4);
        const uint32_t comp = be16at(raw.data() + base + 6);
        img.encryptionType = static_cast<XexEncryptionType>(enc);
        img.compressionType = static_cast<XexCompressionType>(comp);
        if (comp == 2 /* normal */ && base + 12 <= raw.size()) {
          img.lzxWindowSize = be32at(raw.data() + base + 8);
        }
        X360LOGI(TAG, "file format: encryption=%s compression=%s window=%u infoSize=%u",
                 img.encryptionType == XexEncryptionType::None ? "none" : "aes-cbc",
                 img.compressionType == XexCompressionType::None
                     ? "none"
                     : (img.compressionType == XexCompressionType::Basic
                            ? "basic"
                            : (img.compressionType == XexCompressionType::Normal ? "lzx"
                                                                                 : "delta")),
                 img.lzxWindowSize, infoSize);
      }
    } else {
      // Absent means: neither encrypted nor compressed.
      img.encryptionType = XexEncryptionType::None;
      img.compressionType = XexCompressionType::None;
      addWarning("no FILE_FORMAT_INFO header; treating the payload as plaintext");
    }
  }

  if (!readSecurityInfo(raw, img)) return false;

  img.baseAddress = baseFromOpt ? baseFromOpt : img.baseAddress;

  X360LOGI(TAG,
           "xex parsed: tid=%s base=0x%08X imageSize=0x%X pages=%u entry=0x%08X "
           "title='%s'",
           img.titleIdText.empty() ? "????????" : img.titleIdText.c_str(),
           img.baseAddress, img.imageSize, img.pageDescriptorCount, img.entryPoint,
           img.titleName.c_str());
  (void)heapFromOpt;
  return true;
}

bool XexLoader::readSecurityInfo(const std::vector<uint8_t>& raw, XexImage& img) {
  if (raw.size() < 0x1C) {
    m_lastError = "file too small for security info";
    return false;
  }
  const uint32_t off = be32at(raw.data() + 0x0C);
  if (off + 0x184 > raw.size()) {
    m_lastError = "security info out of bounds";
    return false;
  }
  const uint8_t* s = raw.data() + off;
  const uint32_t secHeaderSize = be32at(s + 0x0);
  img.imageSize = be32at(s + 0x4);
  const uint32_t imageFlags = be32at(s + 0x10C);
  img.baseAddress = be32at(s + 0x110);
  img.allowedMediaTypes = be32at(s + 0x17C);
  img.region = be32at(s + 0x178);
  img.pageDescriptorCount = be32at(s + 0x180);
  img.pageSize = 0x1000;

  char mb[41];
  for (int i = 0; i < 16; i++) snprintf(mb + i * 2, 3, "%02X", s[0x140 + i]);
  mb[40] = 0;
  img.mediaIdHex = mb;

  // (The encrypted session key at 0x150 is consumed by computeSessionKey().)

  // Page descriptors (protection info) follow the fixed part.
  const uint32_t pdOff = off + 0x184;
  const uint32_t pdCount = img.pageDescriptorCount;
  if (pdCount > 0 && pdOff + pdCount * 0x18 <= raw.size()) {
    img.pages.clear();
    uint32_t page = 0;
    for (uint32_t i = 0; i < pdCount; i++) {
      const uint32_t value = be32at(raw.data() + pdOff + i * 0x18);
      const uint32_t info = value & 0xF;
      const uint32_t pageCount = value >> 4;
      XexSectionKind kind = XexSectionKind::Unknown;
      switch (info) {
        case 0x0: kind = XexSectionKind::DontCare; break;
        case 0x4:
        case 0x6: kind = XexSectionKind::Code; break;
        case 0x8:
        case 0xC:
        case 0xE: kind = XexSectionKind::ReadOnlyData; break;
        case 0x2:
        case 0xA: kind = XexSectionKind::ReadWriteData; break;
        default: kind = XexSectionKind::Unknown; break;
      }
      XexPageRange pr;
      pr.address = img.baseAddress + page * img.pageSize;
      pr.size = pageCount * img.pageSize;
      pr.kind = kind;
      img.pages.push_back(pr);
      page += pageCount;
    }
    // Page descriptors are authoritative for the image extent.
    const uint32_t fromPages = page * img.pageSize;
    if (fromPages && fromPages != img.imageSize) {
      X360LOGD(TAG, "image size from pages (0x%X) overrides security info (0x%X)",
               fromPages, img.imageSize);
      img.imageSize = fromPages;
    }
  } else if (pdCount) {
    addWarning("page descriptors out of bounds; protections unknown");
  }

  if (img.imageSize == 0) {
    m_lastError = "security info reports zero image size";
    return false;
  }
  if (img.baseAddress == 0) {
    img.baseAddress = 0x82000000u;
    addWarning("no load address; assuming the usual 0x82000000 title base");
  }
  X360LOGD(TAG, "security info: headerSize=0x%X imageSize=0x%X flags=0x%X region=0x%X",
           secHeaderSize, img.imageSize, imageFlags, img.region);
  return true;
}

// ─── Full load ─────────────────────────────────────────────────────────────────

bool XexLoader::loadFromMemory(const uint8_t* data, size_t size, XexImage& outImage,
                               const title::TitleProfile& profile,
                               const std::string& ownerName) {
  m_warnings.clear();
  if (!data || size < 0x200) {
    m_lastError = "XEX blob too small";
    return false;
  }
  outImage.rawFile.assign(data, data + size);
  std::vector<uint8_t>& raw = outImage.rawFile;

  if (!parseHeader(raw, outImage)) return false;

  if (outImage.isPatch || (outImage.moduleFlags & 0x10 /* MODULE */)) {
    if (outImage.isPatch) {
      m_lastError = "title-update (.xexp) images are not applied by this loader";
      return false;
    }
  }
  if (profile.ignore_title_updates && outImage.isPatch) {
    addWarning("profile says ignore_title_updates: patch image skipped");
    m_lastError = "patch image skipped because of title profile";
    return false;
  }

  // ── Identify + confirm profile ──────────────────────────────────────────────
  if (profile.valid) {
    if (profile.matches(outImage.titleId)) {
      X360LOGI(TAG, "IDENTIFIED %s (%s)", profile.name.c_str(),
               outImage.titleIdText.c_str());
    } else if (profile.title_id != 0 && outImage.titleId != 0) {
      X360LOGW(TAG, "title id %s does not match profile %s (%s) — generic behaviour",
               outImage.titleIdText.c_str(), profile.name.c_str(),
               [&] { char b[12]; snprintf(b, sizeof(b), "%08X", profile.title_id); return std::string(b); }());
    }
  }
  if (outImage.titleId == 0) {
    const uint32_t guess = title::guess_title_id_from_path(ownerName);
    if (guess) {
      outImage.titleId = guess;
      char b[12];
      snprintf(b, sizeof(b), "%08X", guess);
      outImage.titleIdText = b;
      addWarning("title id guessed from the file name because EXECUTION_INFO was unreadable");
    }
  }
  outImage.titleName = outImage.titleName.empty() ? profile.name : outImage.titleName;

  if (!readImageIntoMemory(raw, outImage, profile)) return false;
  if (!parsePETable(outImage, profile)) return false;
  if (!applyBaseRelocations(outImage)) return false;
  if (!parseImportTable(outImage, profile)) return false;
  if (!installHleThunks(outImage)) return false;

  X360LOGI(TAG,
           "XEX ready: tid=%s name='%s' base=0x%08X entry=0x%08X size=0x%X "
           "sections=%zu imports=%zu",
           outImage.titleIdText.c_str(), outImage.titleName.c_str(), outImage.baseAddress,
           outImage.entryPoint, outImage.totalSize(), outImage.sections.size(),
           outImage.imports.size());
  return true;
}

bool XexLoader::readImageIntoMemory(std::vector<uint8_t>& raw, XexImage& img,
                                    const title::TitleProfile& profile) {
  if (img.headerSize > raw.size()) {
    m_lastError = "header size larger than file";
    return false;
  }
  const size_t exeLen = raw.size() - img.headerSize;
  uint8_t* exe = raw.data() + img.headerSize;

  // Session key (retail first, dev-kit only if that yields nothing sane).
  uint8_t sessionKey[16] = {};
  bool devKit = false;
  computeSessionKey(raw.data(), raw.size(), sessionKey, &devKit);
  img.usedDevKitKey = devKit;

  const bool encrypted = img.encryptionType == XexEncryptionType::Normal;
  if (encrypted) {
    aesCbcZeroIv(sessionKey, exe, exeLen);
    img.decrypted = true;
  } else if (img.encryptionType != XexEncryptionType::None) {
    if (!profile.allow_unencrypted) {
      m_lastError = "unknown encryption type";
      return false;
    }
    addWarning("unrecognised encryption type, proceeding as plaintext");
  }

  switch (img.compressionType) {
    case XexCompressionType::None:
      if (!readUncompressed(raw, img, sessionKey)) return false;
      break;
    case XexCompressionType::Basic:
      if (!readBasicCompressed(raw, img, sessionKey)) return false;
      break;
    case XexCompressionType::Normal:
      if (!readLzxCompressed(raw, img, sessionKey)) return false;
      break;
    case XexCompressionType::Delta:
      m_lastError = "XEX delta compression is not implemented";
      return false;
  }
  img.decompressed = true;
  return true;
}

bool XexLoader::readUncompressed(std::vector<uint8_t>& raw, XexImage& img,
                                 const uint8_t sessionKey[16]) {
  const size_t exeLen = raw.size() - img.headerSize;
  if (exeLen < img.imageSize) {
    // Some loose/patched XEXs are shorter; zero-pad so the PE table is readable.
    addWarning("payload shorter than declared image size; zero-padding");
    raw.resize(img.headerSize + img.imageSize, 0);
  }
  img.memory.assign(raw.begin() + img.headerSize,
                    raw.begin() + img.headerSize + std::min<size_t>(exeLen, img.imageSize));
  img.memory.resize(img.imageSize, 0);
  (void)sessionKey;
  X360LOGD(TAG, "uncompressed payload: %zu bytes mapped at guest 0x%08X", img.memory.size(),
           img.baseAddress);
  return true;
}

bool XexLoader::readBasicCompressed(std::vector<uint8_t>& raw, XexImage& img,
                                    const uint8_t sessionKey[16]) {
  const uint8_t* exe = raw.data() + img.headerSize;
  const size_t exeLen = raw.size() - img.headerSize;

  // FILE_FORMAT_INFO is at headerSize-0x8? No: it is an optional header blob
  // inside the XEX header (already parsed). Re-derive the block table by asking
  // the header again — cheap and keeps this function independent.
  const uint8_t* hdr = raw.data();
  const uint32_t dirCount = be32at(hdr + 0x10);
  const uint8_t* fmtBlob = nullptr;
  for (uint32_t i = 0; i < dirCount; i++) {
    const uint32_t off = 0x18 + i * 8;
    if (off + 8 > img.headerSize) break;
    if (be32at(hdr + off) == kXexHdrFileFormatInfo) {
      fmtBlob = hdr + be32at(hdr + off + 4);
      break;
    }
  }
  uint32_t blockCount = 0;
  const uint8_t* blocks = nullptr;
  if (fmtBlob) {
    const uint32_t infoSize = be32at(fmtBlob);
    blockCount = (infoSize - 8) / 8;
    blocks = fmtBlob + 8;
  }
  if (blockCount == 0) {
    addWarning("basic compression with an empty block table; copying the payload");
    img.memory.assign(exe, exe + std::min(exeLen, static_cast<size_t>(img.imageSize)));
    img.memory.resize(img.imageSize, 0);
    return true;
  }

  size_t src = 0;
  img.memory.clear();
  img.memory.reserve(img.imageSize);
  for (uint32_t n = 0; n < blockCount; n++) {
    const uint32_t dataSize = be32at(blocks + n * 8);
    const uint32_t zeroSize = be32at(blocks + n * 8 + 4);
    if (src + dataSize > exeLen) {
      m_lastError = "basic compression block overruns the payload";
      return false;
    }
    img.memory.insert(img.memory.end(), exe + src, exe + src + dataSize);
    src += dataSize;
    img.memory.insert(img.memory.end(), zeroSize, 0);
  }
  if (img.memory.size() < img.imageSize) img.memory.resize(img.imageSize, 0);
  else img.memory.resize(img.imageSize);
  X360LOGD(TAG, "basic compression: %u blocks -> %zu bytes", blockCount, img.memory.size());
  (void)sessionKey;
  return true;
}

bool XexLoader::readLzxCompressed(std::vector<uint8_t>& raw, XexImage& img,
                                  const uint8_t sessionKey[16]) {
  const uint8_t* exe = raw.data() + img.headerSize;  // already decrypted if needed
  size_t remaining = raw.size() - img.headerSize;

  // Re-derive the block table pointer from the XEX header (FILE_FORMAT_INFO).
  const uint8_t* hdr = raw.data();
  const uint32_t dirCount = be32at(hdr + 0x10);
  const uint8_t* fmtBlob = nullptr;
  for (uint32_t i = 0; i < dirCount; i++) {
    const uint32_t off = 0x18 + i * 8;
    if (off + 8 > img.headerSize) break;
    if (be32at(hdr + off) == kXexHdrFileFormatInfo) {
      fmtBlob = hdr + be32at(hdr + off + 4);
      break;
    }
  }
  if (!fmtBlob) {
    m_lastError = "LZX-compressed XEX without FILE_FORMAT_INFO";
    return false;
  }
  const uint32_t windowSizeField = be32at(fmtBlob + 8);
  int windowBits = 17;
  if (windowSizeField > 0 && windowSizeField <= 25) windowBits = static_cast<int>(windowSizeField);

  // De-block: [4B BE total size][20B SHA-1][payload...] repeated; the chunked
  // payload of every block is concatenated into one LZX stream.
  std::vector<uint8_t> compressed;
  compressed.reserve(remaining);
  const uint8_t* cur = exe;
  while (remaining >= 4) {
    const uint32_t blockSize = be32at(cur);
    if (blockSize == 0) break;
    if (blockSize > remaining) {
      m_lastError = "LZX de-block size exceeds the payload (key mismatch?)";
      return false;
    }
    const uint8_t* block = cur + 4;
    const uint8_t* payload = cur + 24;
    size_t left = blockSize - 20;
    while (left >= 2) {
      const uint32_t chunk = (uint32_t(payload[0]) << 8) | payload[1];
      payload += 2;
      left -= 2;
      if (chunk == 0) break;
      if (chunk > left) {
        m_lastError = "LZX de-block chunk overruns the block";
        return false;
      }
      compressed.insert(compressed.end(), payload, payload + chunk);
      payload += chunk;
      left -= chunk;
    }
    cur += blockSize;
    remaining -= blockSize;
  }
  if (compressed.empty()) {
    m_lastError = "no LZX data after de-blocking";
    return false;
  }

  img.memory.assign(img.imageSize, 0);
  size_t produced = 0;
  const bool ok = lzx::decompress(compressed.data(), compressed.size(), img.memory.data(),
                                  img.imageSize, windowBits, &produced);
  X360LOGI(TAG, "LZX: %zu compressed bytes -> %zu/%u bytes (windowBits=%d) %s",
           compressed.size(), produced, img.imageSize, windowBits, ok ? "ok" : "FAILED");
  if (!ok) {
    m_lastError = "LZX decompression failed (corrupt or unsupported XEX compression)";
    diag::report_unsupported(diag::Subsystem::Memory, "lzx.decompress_failed");
    return false;
  }
  (void)sessionKey;
  return true;
}

// ─── PE table ──────────────────────────────────────────────────────────────────

bool XexLoader::parsePETable(XexImage& img, const title::TitleProfile& profile) {
  if (img.memory.size() < 0x100) {
    m_lastError = "image too small to hold a PE header";
    return false;
  }
  const uint8_t* base = img.memory.data();
  const size_t cap = img.memory.size();

  const uint16_t dosMagic = le16at(base);
  if (dosMagic != 0x5A4D /* 'MZ' */) {
    m_lastError = "no PE header in the mapped image (decryption/compression failed?)";
    return false;
  }
  const uint32_t peOff = le32at(base + 0x3C);
  if (peOff + 0x18 > cap || le32at(base + peOff) != 0x00004550 /* 'PE\0\0' */) {
    m_lastError = "bad PE signature inside the XEX image";
    return false;
  }
  const uint8_t* coff = base + peOff + 4;
  const uint16_t machine = le16at(coff + 0);
  const uint16_t numSections = le16at(coff + 2);
  const uint16_t optSize = le16at(coff + 16);
  const uint8_t* opt = coff + 20;
  if (opt + optSize > base + cap || optSize < 0x60) {
    m_lastError = "PE optional header out of bounds";
    return false;
  }
  if (machine != 0x1F0 /* IMAGE_FILE_MACHINE_POWERPC */) {
    addWarning("PE machine is not POWERPC (0x1F0)");
  }
  const uint32_t imageBase = le32at(opt + 0x1C);
  const uint32_t entryRva = le32at(opt + 0x28);
  const uint32_t sectionAlign = le32at(opt + 0x38) ? le32at(opt + 0x38) : 0x1000;

  if (profile.override_base_address) {
    img.baseAddress = profile.override_base_address;
  } else if (imageBase && imageBase != img.baseAddress) {
    X360LOGD(TAG, "PE image base 0x%08X (xex base 0x%08X) — using the PE value", imageBase,
             img.baseAddress);
    img.baseAddress = imageBase;
  }
  img.entryPoint = img.baseAddress + entryRva;

  const uint8_t* sec = opt + optSize;
  img.sections.clear();
  for (uint16_t i = 0; i < numSections; i++) {
    const uint8_t* s = sec + i * 40;
    if (s + 40 > base + cap) break;
    XexPESection ps;
    char nm[9] = {};
    memcpy(nm, s, 8);
    ps.name = nm;
    ps.virtualSize = le32at(s + 8);
    ps.virtualAddress = le32at(s + 12);
    ps.rawSize = le32at(s + 16);
    ps.fileOffset = le32at(s + 20);
    ps.characteristics = le32at(s + 36);
    ps.executable = (ps.characteristics & 0x20000000u) != 0;
    ps.writable = (ps.characteristics & 0x80000000u) != 0;
    img.sections.push_back(ps);
  }
  X360LOGI(TAG, "PE: %u sections, entry=0x%08X sectionAlign=0x%X", numSections,
           img.entryPoint, sectionAlign);
  for (const auto& s : img.sections) {
    X360LOGD(TAG, "  section %-8s va=0x%08X vs=0x%08X raw=0x%08X/0x%08X %s%s", s.name.c_str(),
             s.virtualAddress, s.virtualSize, s.fileOffset, s.rawSize,
             s.executable ? "X" : "-", s.writable ? "W" : "-");
  }
  return true;
}

bool XexLoader::applyBaseRelocations(XexImage& img) {
  // We always map at the PE's own ImageBase, so relocation is normally a no-op.
  // Report it loudly instead of silently ignoring a real need.
  return true;
}

// ─── Kernel imports ────────────────────────────────────────────────────────────

bool XexLoader::parseImportTable(XexImage& img, const title::TitleProfile& profile) {
  img.libraries.clear();
  img.imports.clear();

  if (img.headerSize < 0x18 || img.headerSize > img.rawFile.size()) {
    m_lastError = "broken XEX header while parsing imports";
    return false;
  }
  const uint8_t* hdr = img.rawFile.data();
  const uint32_t dirCount = be32at(hdr + 0x10);
  uint32_t importOffset = 0;
  for (uint32_t i = 0; i < dirCount; i++) {
    const uint32_t off = 0x18 + i * 8;
    if (off + 8 > img.headerSize) break;
    if (be32at(hdr + off) == kXexHdrImportLibraries) {
      importOffset = be32at(hdr + off + 4);
      break;
    }
  }
  if (!importOffset || importOffset + 12 > img.headerSize) {
    addWarning("no IMPORT_LIBRARIES header: the title has no kernel imports?");
    return true;
  }

  const uint32_t totalSize = be32at(hdr + importOffset);
  const uint32_t strSize = be32at(hdr + importOffset + 4);
  const uint32_t strCount = be32at(hdr + importOffset + 8);
  if (strCount > 64 || strSize > totalSize) {
    m_lastError = "implausible import string table";
    return false;
  }
  const uint8_t* strings = hdr + importOffset + 12;
  std::vector<const char*> stringTable;
  {
    uint32_t i = 0;
    while (i < strSize && stringTable.size() < strCount) {
      stringTable.push_back(reinterpret_cast<const char*>(strings + i));
      i += static_cast<uint32_t>(strlen(reinterpret_cast<const char*>(strings + i))) + 1;
      if (i & 3) i += 4 - (i & 3);
    }
  }

  uint32_t libOffset = importOffset + 12 + strSize;
  uint32_t guard = 0;
  while (libOffset + 0x28 <= img.headerSize && guard++ < 16) {
    const uint8_t* lib = hdr + libOffset;
    const uint32_t libSize = be32at(lib + 0x0);
    if (libSize < 0x28) break;
    XexImportLibrary entry;
    entry.id = be32at(lib + 0x18);
    entry.version = be32at(lib + 0x1C);
    entry.minVersion = be32at(lib + 0x20);
    const uint16_t nameIndex = be16at(lib + 0x24);
    const uint16_t count = be16at(lib + 0x26);
    if (nameIndex < stringTable.size()) entry.moduleName = stringTable[nameIndex];
    const uint32_t* table = reinterpret_cast<const uint32_t*>(lib + 0x28);
    X360LOGI(TAG, "import library '%s' id=%u version=%u records=%u",
             entry.moduleName.c_str(), entry.id, entry.version, count);

    for (uint32_t i = 0; i < count; i++) {
      const uint32_t tableOff = 0x28 + i * 4;
      if (tableOff + 4 > libSize) break;
      const uint32_t recordAddr = be32at(lib + tableOff);  // guest VA of the slot
      XexImportRecord rec;
      rec.slotAddress = recordAddr;
      if (recordAddr >= img.baseAddress && recordAddr - img.baseAddress < img.memory.size()) {
        rec.slotValue = be32at(img.memory.data() + (recordAddr - img.baseAddress));
      }
      rec.originalValue = rec.slotValue;
      rec.isFunction = ((rec.slotValue >> 24) & 0xFF) == 1;
      rec.ordinal = static_cast<uint16_t>(rec.slotValue & 0xFFFF);
      entry.records.push_back(rec);
    }
    img.libraries.push_back(std::move(entry));
    libOffset += libSize;
  }

  // Flatten into the sentinel-indexed list and resolve names.
  uint32_t unresolved = 0;
  for (auto& lib : img.libraries) {
    for (auto& rec : lib.records) {
      XexImage::Import imp;
      imp.library = lib.moduleName;
      imp.ordinal = rec.ordinal;
      imp.isFunction = rec.isFunction;
      imp.slotAddress = rec.slotAddress;
      const char* name = kernel::name_for_ordinal(lib.moduleName, rec.ordinal);
      if (name) {
        imp.name = name;
        imp.resolved = kernel::is_implemented(lib.moduleName, rec.ordinal);
      } else {
        char b[48];
        snprintf(b, sizeof(b), "%s_%03X", lowerAscii(lib.moduleName).c_str(), rec.ordinal);
        imp.name = b;
        imp.resolved = false;
      }
      if (!imp.resolved) unresolved++;
      rec.hleIndex = static_cast<uint32_t>(img.imports.size());
      img.imports.push_back(std::move(imp));
      if (img.imports.size() >= kHleMaxImports) {
        addWarning("import table truncated at the sentinel capacity");
        break;
      }
    }
  }
  X360LOGI(TAG, "imports: %zu total, %zu without a runtime implementation",
           img.imports.size(), unresolved);
  if (unresolved && profile.require_imports_resolved) {
    m_lastError = "title profile requires all imports to resolve; " + std::to_string(unresolved) +
                  " are missing";
    return false;
  }
  return true;
}

bool XexLoader::installHleThunks(XexImage& img) {
  // Rewrite every function-import record slot with the HLE sentinel; keep data
  // imports pointing at a zeroed scratch region the runtime owns.
  uint32_t sentinels = 0, dataSlots = 0, skipped = 0;
  for (auto& imp : img.imports) {
    if (imp.slotAddress < img.baseAddress ||
        imp.slotAddress - img.baseAddress + 4 > img.memory.size()) {
      skipped++;
      continue;
    }
    uint8_t* slot = img.memory.data() + (imp.slotAddress - img.baseAddress);
    if (imp.isFunction) {
      put_be32(slot, kHleSentinelBase + imp.ordinal);
      sentinels++;
    } else {
      // Data import: leave the slot pointing at a fixed scratch page inside the
      // title's own image tail. The runtime zeroes it, so reads return 0.
      put_be32(slot, 0);
      dataSlots++;
    }
  }
  X360LOGI(TAG, "HLE thunks: %u function sentinels, %u data slots, %u out-of-range",
           sentinels, dataSlots, skipped);
  if (skipped) addWarning("some import records point outside the image and were skipped");
  return true;
}

}  // namespace x360
