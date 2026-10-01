/**
 * HLE Memory — NtAllocateVirtualMemory / NtFreeVirtualMemory / NtProtectVirtualMemory
 * and the XMemAlloc physical-pool allocator, on top of the flat 4 GB guest mapping.
 *
 * What changed vs. the previous version (and why it mattered for Xenoverse):
 *   - the old bump allocator handed out addresses by bumping a single pointer and
 *     never reused memory, so a title that allocates/frees in a loop (all of them)
 *     ran out of address space; it also memset() every allocation, which for the
 *     multi-hundred-megabyte pools Xenoverse requests is slow enough to look like a hang.
 *   - allocations were not checked against the loaded image, so a title could be
 *     handed an address inside its own code.
 *
 * This is a plain sorted free-list allocator (largest-fit) over [heapBase, heapEnd)
 * of the guest window. Zero-on-free replaces zero-on-alloc: freshly mmap'd memory is
 * already zero and we keep that invariant, so allocation is O(log n) and no longer
 * touches the pages.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>

namespace x360 {
namespace hle {

class GuestHeap {
 public:
  // hostBase: start of the flat guest mapping; guestBase: guest VA of hostBase.
  void init(uint8_t* hostBase, uint64_t hostSize, uint64_t guestBase,
            uint64_t heapBegin, uint64_t heapEnd);
  void reset();

  // Reserve `size` bytes (rounded up to 64 KB granularity) at/above hint (0 = any).
  // Returns 0 on failure. `committed` regions are simply part of the flat mapping.
  uint64_t allocate(uint64_t hint, uint64_t size, uint32_t protect, uint64_t* outSize);
  bool free(uint64_t address);
  // Query used by NtQueryVirtualMemory / diagnostics.
  bool query(uint64_t address, uint64_t* outBase, uint64_t* outSize, uint32_t* outProtect);
  bool protect(uint64_t address, uint64_t size, uint32_t newProtect);

  // XMemAlloc-style physical pool (Xenoverse allocates most of its buffers here).
  uint64_t physicalAlloc(uint64_t size, uint32_t alignment);
  void physicalFree(uint64_t address);

  // Kernel-owned scratch region (import variable values, small structures the
  // kernel hands out pointers to). Returns a guest VA, zero-initialised.
  uint64_t allocKernelVariable(uint32_t size);

  // Mark a range as already occupied (used to reserve the loaded XEX image and
  // the thread stacks so the heap never hands them out).
  void reserve(uint64_t address, uint64_t size, uint32_t protect);

  uint64_t bytes_free() const;
  uint64_t bytes_committed() const;
  uint64_t heap_base() const { return heapBegin_; }
  uint64_t heap_end() const { return heapEnd_; }

  // Human-readable state for crash logs.
  void describe(char* out, size_t max) const;

  static constexpr uint64_t kGranularity = 0x10000;  // 64 KB, as on real Xenon

 private:

  uint8_t* host_ = nullptr;
  uint64_t host_size_ = 0;
  uint64_t guest_base_ = 0;
  uint64_t heapBegin_ = 0;
  uint64_t heapEnd_ = 0;
  uint64_t kernelScratch_ = 0;      // cursor into the reserved kernel area
  uint64_t kernelScratchEnd_ = 0;

 public:
  struct RegionInfo {
    uint64_t size = 0;
    uint32_t protect = 0;
  };

 private:
  mutable std::mutex mutex_;
  std::map<uint64_t, RegionInfo> committed_;   // address -> region
  std::map<uint64_t, uint64_t> free_;      // address -> size (free blocks)

  bool carve(uint64_t address, uint64_t size, uint64_t align, uint64_t* outAddr);
  void insertFree(uint64_t address, uint64_t size);
  void coalesceFree(uint64_t address);
  uint8_t* host_of(uint64_t guest) const {
    if (guest < guest_base_) return nullptr;
    const uint64_t off = guest - guest_base_;
    return off < host_size_ ? host_ + off : nullptr;
  }
};

}  // namespace hle
}  // namespace x360
