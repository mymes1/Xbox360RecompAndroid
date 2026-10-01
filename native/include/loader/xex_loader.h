/**
 * XEX2 loader — parses, decrypts (AES-128-CBC with the public retail/devkit
 * key), decompresses (none / basic / LZX) and maps an Xbox 360 executable,
 * including its PE sections, page protections, and the kernel import table.
 *
 * Two-stage load, because the runtime needs the title id *before* it can pick a
 * per-title profile, and needs imports resolved before the JIT can run:
 *    peekInfo()      -> header + execution id only (title id, name, media type)
 *    loadFromMemory()-> everything, ready to install into guest memory
 *
 * Everything here is title-agnostic; Xenoverse-specific choices arrive through
 * the TitleProfile passed in (see x360/title_profile.h).
 *
 * Format knowledge comes from publicly documented XEX2 layout (same structure
 * Xenia documents in xex2_info.h); no proprietary source is used.
 */
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "x360/title_profile.h"

namespace x360 {

// XEX2 magic + sentinel encodings
static constexpr uint32_t XEX2_MAGIC = 0x58455832;  // 'XEX2'
static constexpr uint32_t XEX1_MAGIC = 0x58455831;  // 'XEX1'

/**
 * HLE import sentinel range.
 *
 * How kernel imports work on a real Xbox 360 XEX: the PE section contains 4-byte
 * "import record" slots whose value is `type<<24 | ordinal` before load, and the
 * kernel rewrites them to addresses of thunks/variables at load time. A title
 * then does `lwz rX, 0(rY)` + `mtlr rX` + `blr`, or `bl` directly at the slot
 * (the slot value is *also* the code address for a "thunked" call).
 *
 * We encode the resolved import as `kHleSentinelBase + importIndex`. The JIT
 * dispatch loop recognises PC in [0x8FFE0000, 0x8FFE0000 + 0x10000) and calls
 * the HLE handler table instead of compiling guest code.
 */
static constexpr uint32_t kHleSentinelBase = 0x8FFE0000u;
static constexpr uint32_t kHleSentinelMask = 0xFFFF0000u;
static constexpr uint32_t kHleMaxImports   = 0x10000u;

// Legacy sentinel used before this rewrite (module addresses in 0x80000000+);
// still honoured so a hand-made test XEX keeps working.
static constexpr uint32_t kLegacyHleBase = 0x80000000u;

// XEX2 header keys we care about (public, from the XEX2 format docs).
enum Xex2HeaderKey : uint32_t {
  kXexHdrResourceInfo        = 0x000002FF,
  kXexHdrFileFormatInfo      = 0x000003FF,
  kXexHdrOriginalBaseAddress = 0x00010001,
  kXexHdrEntryPoint          = 0x00010100,
  kXexHdrImageBaseAddress    = 0x00010201,
  kXexHdrImportLibraries     = 0x000103FF,
  kXexHdrStaticLibraries     = 0x000200FF,
  kXexHdrDefaultStackSize    = 0x00020200,
  kXexHdrDefaultHeapSize     = 0x00020401,
  kXexHdrPageHeapSizeAndFlags= 0x00028002,
  kXexHdrSystemFlags         = 0x00030000,
  kXexHdrExecutionInfo       = 0x00040006,
  kXexHdrTitleWorkspaceSize  = 0x00040201,
  kXexHdrAlternateTitleIds   = 0x000407FF,
  kXexHdrAdditionalTitleMem  = 0x00040801,
  kXexHdrDeltaPatchDesc      = 0x000005FF,
};

// System flags (subset relevant to loading decisions).
enum XexSystemFlag : uint32_t {
  kXexSysRequireFes         = 0x00000001,
  kXexSysDisableUserModeCode= 0x00000004,
  kXexSysTitle              = 0x00000080,
  kXexSysModule             = 0x00000100,
  kXexSysHardDiskHide       = 0x00001000,
  kXexSysDiscNoInstall        = 0x00004000,
  kXexSysMoveToDataPartition= 0x00020000,
};

enum class XexCompressionType : uint32_t {
  None = 0,
  Basic = 1,
  Normal = 2,   // LZX
  Delta = 3,
};

enum class XexEncryptionType : uint32_t {
  None = 0,
  Normal = 1,   // AES-128-CBC, key derived from the security info
};

enum class XexSectionKind : uint8_t {
  DontCare = 0,
  Code = 1,
  ReadOnlyData = 2,
  ReadWriteData = 4,
  Unknown = 7,
};

struct XexImportRecord {
  uint32_t slotAddress = 0;    // guest VA of the record slot (rewritten to a sentinel)
  uint32_t slotValue = 0;      // original value: type<<24 | ordinal
  uint32_t originalValue = 0;  // for round-trip debugging
  bool     isFunction = true;  // false => data import
  uint16_t ordinal = 0;
  uint32_t hleIndex = 0xFFFFFFFF;  // index into XexImage::imports (sentinel payload)
};

struct XexImportLibrary {
  std::string moduleName;      // e.g. "xboxkrnl.exe", "xam.xex"
  uint32_t id = 0;
  uint32_t version = 0;
  uint32_t minVersion = 0;
  std::vector<XexImportRecord> records;
};

struct XexPESection {
  std::string name;
  uint32_t virtualAddress = 0;
  uint32_t virtualSize = 0;
  uint32_t fileOffset = 0;
  uint32_t rawSize = 0;
  uint32_t characteristics = 0;
  bool executable = false;
  bool writable = false;
};

struct XexPageRange {
  uint32_t address = 0;
  uint32_t size = 0;
  XexSectionKind kind = XexSectionKind::Unknown;
};

struct XexImage {
  // ── identity ────────────────────────────────────────────────────────────────
  uint32_t titleId = 0;              // 0xTTRRNNNN (execution info)
  std::string titleIdText;           // "%08X" of the above
  std::string titleName;             // best-effort display name
  uint32_t alternateTitleIds[8] = {};
  uint32_t alternateTitleIdCount = 0;
  uint32_t version = 0;              // execution info version
  uint32_t baseVersion = 0;
  uint32_t allowedMediaTypes = 0;
  uint32_t region = 0;
  uint32_t savegameId = 0;
  std::string mediaIdHex;
  std::string internalName;          // ORIGINAL_PE_NAME if present

  // ── layout ──────────────────────────────────────────────────────────────────
  uint32_t moduleFlags = 0;
  uint32_t systemFlags = 0;
  uint32_t baseAddress = 0;          // normally 0x82000000
  uint32_t entryPoint = 0;           // absolute guest VA
  uint32_t imageSize = 0;            // from page descriptors
  uint32_t defaultStackSize = 0;
  uint32_t defaultHeapSize = 0;
  uint32_t defaultFibersSize = 0;
  uint32_t titleWorkspaceSize = 0;
  uint32_t extraMemorySize = 0;
  uint32_t pageDescriptorCount = 0;
  uint32_t pageSize = 0x1000;
  XexCompressionType compressionType = XexCompressionType::None;
  XexEncryptionType encryptionType = XexEncryptionType::None;
  uint32_t lzxWindowSize = 0;
  uint32_t headerSize = 0;
  bool isPatch = false;
  bool usedDevKitKey = false;
  bool decrypted = false;
  bool decompressed = false;

  std::vector<XexPESection> sections;
  std::vector<XexPageRange> pages;
  std::vector<XexImportLibrary> libraries;

  // Flattened import list in sentinel order: index == (sentinel - base).
  struct Import {
    std::string library;      // "xboxkrnl.exe"
    std::string name;         // resolved from the ordinal tables, "" if unknown
    uint32_t ordinal = 0;
    bool isFunction = true;
    bool resolved = false;    // runtime has an implementation for it
    uint32_t slotAddress = 0;
    uint32_t value = 0;       // for data imports: address of the HLE variable
  };
  std::vector<Import> imports;

  // Linear image buffer, addressed as buffer[guestAddress - baseAddress].
  std::vector<uint8_t> memory;
  // Kept so callers can re-run the PE step after applying a title update.
  std::vector<uint8_t> rawFile;

  uint32_t totalSize() const { return static_cast<uint32_t>(memory.size()); }
  bool contains(uint32_t guestAddr) const {
    if (guestAddr < baseAddress) return false;
    return (guestAddr - baseAddress) < memory.size();
  }
  uint8_t* hostFor(uint32_t guestAddr) {
    return contains(guestAddr) ? memory.data() + (guestAddr - baseAddress) : nullptr;
  }
  const uint8_t* hostFor(uint32_t guestAddr) const {
    return contains(guestAddr) ? memory.data() + (guestAddr - baseAddress) : nullptr;
  }
};

class XexLoader {
 public:
  XexLoader();
  ~XexLoader();

  // Load from a filesystem path (extracted .xex / loose file).
  bool load(const std::string& path, XexImage& outImage,
            const title::TitleProfile& profile);

  // Load from an in-memory XEX blob (e.g. read out of an ISO without touching
  // the disk). `ownerName` is only used for logs (e.g. "disc.iso").
  bool loadFromMemory(const uint8_t* data, size_t size, XexImage& outImage,
                      const title::TitleProfile& profile, const std::string& ownerName);

  // Header-only pass: enough to identify a title and pick a profile.
  bool peekInfo(const uint8_t* data, size_t size, XexImage& outImage);
  bool peekInfo(const std::string& path, XexImage& outImage);

  const std::string& lastError() const { return m_lastError; }
  // Non-fatal problems (missing optional headers, unresolvable imports, ...).
  const std::vector<std::string>& warnings() const { return m_warnings; }

  // ── Helpers exposed for the offline analyser (x360tool) and tests ──────────
  static const char* headerKeyName(uint32_t key);
  static uint32_t titleIdFromAscii(const char digits[4]);
  // Derives the AES session key from the security info using a given master key.
  static bool deriveSessionKey(const uint8_t masterKey[16], const uint8_t encKey[16],
                               uint8_t outSessionKey[16]);

 private:
  bool parseHeader(const std::vector<uint8_t>& raw, XexImage& img);
  bool readSecurityInfo(const std::vector<uint8_t>& raw, XexImage& img);
  bool readImageIntoMemory(std::vector<uint8_t>& raw, XexImage& img,
                           const title::TitleProfile& profile);
  bool readUncompressed(std::vector<uint8_t>& raw, XexImage& img,
                        const uint8_t sessionKey[16]);
  bool readBasicCompressed(std::vector<uint8_t>& raw, XexImage& img,
                           const uint8_t sessionKey[16]);
  bool readLzxCompressed(std::vector<uint8_t>& raw, XexImage& img,
                         const uint8_t sessionKey[16]);
  bool parsePETable(XexImage& img, const title::TitleProfile& profile);
  bool applyBaseRelocations(XexImage& img);
  bool parseImportTable(XexImage& img, const title::TitleProfile& profile);
  bool installHleThunks(XexImage& img);
  void addWarning(std::string msg);

  // Derives the per-image AES session key from the (plaintext) security info.
  static bool computeSessionKey(const uint8_t* raw, size_t rawSize,
                                uint8_t outSessionKey[16], bool* usedDevKitKey);

  std::string m_lastError;
  std::vector<std::string> m_warnings;
};

}  // namespace x360
