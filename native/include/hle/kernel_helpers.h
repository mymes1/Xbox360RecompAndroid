/**
 * Shared helpers for the HLE dispatchers (guest memory access + NTSTATUS).
 *
 * The guest is big-endian and its addresses are 32-bit, so *every* read/write
 * crossing into guest memory must go through these: the previous code had
 * half-accesses (a 32-bit read to fetch a 16-bit UTF-16 unit) and unguarded
 * pointer arithmetic, which produced wrong paths and SIGSEGVs that looked like
 * "the game crashed" rather than "the HLE mis-read a string".
 */
#pragma once
#include <cstdint>
#include <string>

#include "hle/hle_kernel.h"

namespace x360 {
namespace hle {

// NTSTATUS values (subset the runtime actually returns).
enum NtStatus : uint32_t {
  kStatusSuccess = 0x00000000,
  kStatusWait0 = 0x00000000,
  kStatusTimeout = 0x00000102,
  kStatusPending = 0x00000103,
  kStatusAlerted = 0x00000101,
  kStatusBufferOverflow = 0x80000023,
  kStatusInvalidHandle = 0xC0000008,
  kStatusInvalidParameter = 0xC000000D,
  kStatusNoSuchFile = 0xC000000F,
  kStatusUnsuccessful = 0xC00000BB,
  kStatusNoMemory = 0xC0000017,
  kStatusInvalidDeviceRequest = 0xC0000010,
  kStatusEndOfFile = 0xC0000011,
  kStatusNotSupported = 0xC00000BB,
  kStatusObjectNameNotFound = 0xC0000034,
  kStatusObjectPathNotFound = 0xC000003A,
  kStatusAccessDenied = 0xC0000022,
  kStatusSharingViolation = 0xC0000043,
  kStatusFileIsADirectory = 0xC00000BA,
  kStatusDirectoryNotEmpty = 0xC0000101,
  kStatusInfoLengthMismatch = 0xC0000004,
  kStatusCantOpenFile = 0xC0000105,
  kStatusNoMoreEntries = 0x8000001A,
};

// Xbox (Win32-style) error values.
enum XError : uint32_t {
  kErrorSuccess = 0,
  kErrorFileNotFound = 2,
  kErrorAccessDenied = 5,
  kErrorInvalidHandle = 6,
  kErrorNotEnoughMemory = 8,
  kErrorInvalidParameter = 87,
  kErrorNoMoreFiles = 18,
  kErrorDeviceNotConnected = 0x48F,
  kERROR_SUCCESS = kErrorSuccess,
};

// CreateDisposition values for NtCreateFile.
enum FileDisposition : uint32_t {
  kSupersedeExisting = 0,
  kOpenInitialFile = 1,
  kCreateNew = 2,
  kCreateFile = 3,
  kOpenIf = 4,
  kOverwriteIf = 5,
};

// Access flags (generic subset).
enum FileAccess : uint32_t {
  kGenericRead = 0x80000000,
  kGenericWrite = 0x40000000,
  kGenericExecute = 0x20000000,
  kGenericAll = 0x10000000,
  kFileReadData = 0x00000001,
  kFileWriteData = 0x00000002,
  kFileReadAttributes = 0x00000080,
  kSynchronize = 0x00100000,
};

// Guest-memory accessors, bounds-checked and big-endian correct.
bool g_read8(HleKernel& k, uint64_t addr, uint8_t* out);
bool g_read16(HleKernel& k, uint64_t addr, uint16_t* out);
bool g_read32(HleKernel& k, uint64_t addr, uint32_t* out);
bool g_read64(HleKernel& k, uint64_t addr, uint64_t* out);
bool g_write8(HleKernel& k, uint64_t addr, uint8_t v);
bool g_write16(HleKernel& k, uint64_t addr, uint16_t v);
bool g_write32(HleKernel& k, uint64_t addr, uint32_t v);
bool g_write64(HleKernel& k, uint64_t addr, uint64_t v);
bool g_zero(HleKernel& k, uint64_t addr, uint32_t len);

// UNICODE_STRING { be16 length; be16 max_length; be32 buffer; } at `addr`.
bool g_read_unicode_string(HleKernel& k, uint64_t addr, std::string* outUtf8);
// ANSI_STRING { be16 length; be16 max_length; be32 buffer; }
bool g_read_ansi_string(HleKernel& k, uint64_t addr, std::string* out);
// OBJECT_ATTRIBUTES { length, root, name*, attributes, sd, se }  (Xbox 360 layout)
bool g_read_object_attributes_name(HleKernel& k, uint64_t addr, std::string* outUtf8);
// IO_STATUS_BLOCK { be64 status; be32 information; } — the guest passes a pointer.
bool g_write_io_status(HleKernel& k, uint64_t addr, uint32_t status, uint32_t information);

// Copy helper.
bool g_read_bytes(HleKernel& k, uint64_t addr, void* dst, uint32_t len);
bool g_write_bytes(HleKernel& k, uint64_t addr, const void* src, uint32_t len);

}  // namespace hle
}  // namespace x360
