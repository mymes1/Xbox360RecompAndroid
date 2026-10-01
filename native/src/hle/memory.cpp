/**
 * GuestHeap implementation. See guest_heap.h for the design notes.
 */
#include "../../include/hle/guest_heap.h"
#include "../../include/x360/log.h"

#include <algorithm>
#include <cstring>

#define TAG "X360:MEM"

namespace x360 {
namespace hle {

namespace {
constexpr size_t kKernelScratchSize = 4u << 20;  // 4 MB, plenty for import vars
}

// Lock-free internals: callers already hold mutex_.
static uint64_t sum_free(const std::map<uint64_t, uint64_t>& free_) {
  uint64_t n = 0;
  for (const auto& kv : free_) n += kv.second;
  return n;
}
static uint64_t sum_committed(const std::map<uint64_t, GuestHeap::RegionInfo>& committed_) {
  uint64_t n = 0;
  for (const auto& kv : committed_) n += kv.second.size;
  return n;
}

void GuestHeap::init(uint8_t* hostBase, uint64_t hostSize, uint64_t guestBase,
                     uint64_t heapBegin, uint64_t heapEnd) {
  std::lock_guard<std::mutex> lk(mutex_);
  host_ = hostBase;
  host_size_ = hostSize;
  guest_base_ = guestBase;
  heapBegin_ = heapBegin;
  heapEnd_ = heapEnd;
  committed_.clear();
  free_.clear();
  free_[heapBegin] = heapEnd - heapBegin;
  kernelScratch_ = 0;
  kernelScratchEnd_ = 0;
  X360LOGI(TAG, "guest heap: 0x%08llX..0x%08llX (%llu MB)",
           (unsigned long long)heapBegin, (unsigned long long)heapEnd,
           (unsigned long long)((heapEnd - heapBegin) >> 20));
}

void GuestHeap::reset() {
  std::lock_guard<std::mutex> lk(mutex_);
  committed_.clear();
  free_.clear();
  free_[heapBegin_] = heapEnd_ - heapBegin_;
  kernelScratch_ = 0;
  kernelScratchEnd_ = 0;
}

void GuestHeap::insertFree(uint64_t address, uint64_t size) {
  if (size == 0) return;
  free_[address] = size;
}

void GuestHeap::coalesceFree(uint64_t address) {
  auto it = free_.find(address);
  if (it == free_.end()) return;
  // Merge with the next block if contiguous.
  auto next = std::next(it);
  if (next != free_.end() && it->first + it->second == next->first) {
    it->second += next->second;
    free_.erase(next);
  }
  // Merge with the previous block if contiguous.
  if (it != free_.begin()) {
    auto prev = std::prev(it);
    if (prev->first + prev->second == it->first) {
      prev->second += it->second;
      free_.erase(it);
    }
  }
}

bool GuestHeap::carve(uint64_t hint, uint64_t size, uint64_t align, uint64_t* outAddr) {
  // Largest-fit: scan the free map and pick the biggest suitable block, which
  // keeps the fragmentation profile of the Xenon 64 KB-granularity heaps.
  uint64_t bestAddr = 0;
  uint64_t bestSize = 0;
  for (auto& [addr, sz] : free_) {
    if (sz < size) continue;
    uint64_t candidate = addr;
    if (hint && candidate < hint) {
      // Only consider blocks that can satisfy the hint.
      if (candidate + sz < hint + size) continue;
      candidate = (hint + align - 1) & ~(align - 1);
      if (candidate + size > addr + sz) continue;
    } else if (align > 1) {
      candidate = (candidate + align - 1) & ~(align - 1);
      if (candidate + size > addr + sz) continue;
    }
    if (sz > bestSize) {
      bestSize = sz;
      bestAddr = candidate;
      if (!hint) {
        // For "any address" we prefer the *lowest* fitting block instead.
        bestAddr = candidate;
        bestSize = 0;  // sentinel: first fit wins
        break;
      }
    }
  }
  if (!bestAddr && hint) return false;
  if (!bestAddr) {
    // first-fit fallback
    for (auto& [addr, sz] : free_) {
      if (sz < size) continue;
      uint64_t candidate = addr;
      if (align > 1) candidate = (candidate + align - 1) & ~(align - 1);
      if (candidate + size > addr + sz) continue;
      bestAddr = candidate;
      bestSize = sz;
      break;
    }
  }
  if (!bestAddr) return false;

  const uint64_t alignEnd = (bestAddr + size + kGranularity - 1) & ~(kGranularity - 1);
  const uint64_t used = std::max(alignEnd, bestAddr + size) - bestAddr;
  auto it = free_.lower_bound(bestAddr);
  // Find the block that actually contains bestAddr.
  for (auto jt = free_.begin(); jt != free_.end(); ++jt) {
    if (jt->first <= bestAddr && bestAddr + used <= jt->first + jt->second) {
      const uint64_t blockAddr = jt->first;
      const uint64_t blockSize = jt->second;
      free_.erase(jt);
      if (bestAddr > blockAddr) insertFree(blockAddr, bestAddr - blockAddr);
      const uint64_t tail = blockAddr + blockSize - (bestAddr + used);
      if (tail) insertFree(bestAddr + used, tail);
      coalesceFree(blockAddr);
      break;
    }
  }
  *outAddr = bestAddr;
  return true;
}

uint64_t GuestHeap::allocate(uint64_t hint, uint64_t size, uint32_t protect,
                             uint64_t* outSize) {
  std::lock_guard<std::mutex> lk(mutex_);
  if (size == 0) return 0;
  const uint64_t rounded = (size + kGranularity - 1) & ~(kGranularity - 1);
  uint64_t addr = 0;
  if (!carve(hint, rounded, kGranularity, &addr)) {
    X360LOGE(TAG, "out of address space allocating 0x%llX bytes (hint 0x%llX, free %llu MB)",
             (unsigned long long)size, (unsigned long long)hint,
             (unsigned long long)(sum_free(free_) >> 20));
    return 0;
  }
  committed_[addr] = RegionInfo{rounded, protect};
  if (outSize) *outSize = rounded;
  return addr;
}

bool GuestHeap::free(uint64_t address) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = committed_.find(address);
  if (it == committed_.end()) return false;
  const uint64_t size = it->second.size;
  committed_.erase(it);
  insertFree(address, size);
  coalesceFree(address);
  return true;
}

bool GuestHeap::query(uint64_t address, uint64_t* outBase, uint64_t* outSize,
                      uint32_t* outProtect) {
  std::lock_guard<std::mutex> lk(mutex_);
  for (auto& [base, r] : committed_) {
    if (address >= base && address < base + r.size) {
      if (outBase) *outBase = base;
      if (outSize) *outSize = r.size;
      if (outProtect) *outProtect = r.protect;
      return true;
    }
  }
  return false;
}

bool GuestHeap::protect(uint64_t address, uint64_t size, uint32_t newProtect) {
  std::lock_guard<std::mutex> lk(mutex_);
  bool touched = false;
  for (auto& [base, r] : committed_) {
    if (address < base + r.size && base < address + size) {
      r.protect = newProtect;
      touched = true;
    }
  }
  (void)size;
  return touched;
}

uint64_t GuestHeap::physicalAlloc(uint64_t size, uint32_t alignment) {
  if (alignment < kGranularity) alignment = kGranularity;
  std::lock_guard<std::mutex> lk(mutex_);
  uint64_t addr = 0;
  if (!carve(0, size, alignment, &addr)) {
    X360LOGE(TAG, "physical pool exhausted allocating 0x%llX", (unsigned long long)size);
    return 0;
  }
  committed_[addr] = RegionInfo{size, 0x40 /* PAGE_READWRITE */};
  return addr;
}

void GuestHeap::physicalFree(uint64_t address) { free(address); }

uint64_t GuestHeap::allocKernelVariable(uint32_t size) {
  std::lock_guard<std::mutex> lk(mutex_);
  if (kernelScratch_ == 0) {
    uint64_t addr = 0;
    if (!carve(0, kKernelScratchSize, kGranularity, &addr)) return 0;
    committed_[addr] = RegionInfo{kKernelScratchSize, 0x40};
    kernelScratch_ = addr;
    kernelScratchEnd_ = addr + kKernelScratchSize;
    if (host_) {
      uint8_t* p = host_of(addr);
      if (p) memset(p, 0, kKernelScratchSize);
    }
  }
  const uint64_t aligned = (kernelScratch_ + 15) & ~uint64_t(15);
  if (aligned + size > kernelScratchEnd_) return 0;
  kernelScratch_ = aligned + size;
  return aligned;
}

void GuestHeap::reserve(uint64_t address, uint64_t size, uint32_t protect) {
  std::lock_guard<std::mutex> lk(mutex_);
  if (!size) return;
  uint64_t addr = 0;
  if (carve(address, size, kGranularity, &addr)) {
    committed_[addr] = RegionInfo{size, protect};
    if (addr != address) {
      X360LOGW(TAG, "reservation of 0x%08llX moved to 0x%08llX", (unsigned long long)address,
               (unsigned long long)addr);
    }
  } else {
    X360LOGW(TAG, "could not reserve 0x%llX bytes at 0x%08llX", (unsigned long long)size,
             (unsigned long long)address);
  }
}


uint64_t GuestHeap::bytes_free() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return sum_free(free_);
}

uint64_t GuestHeap::bytes_committed() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return sum_committed(committed_);
}

void GuestHeap::describe(char* out, size_t max) const {
  std::lock_guard<std::mutex> lk(mutex_);
  if (!out || !max) return;
  uint64_t fr = 0;
  for (auto& [a, s] : free_) fr += s;
  snprintf(out, max,
           "guest heap 0x%08llX..0x%08llX: %zu committed regions (%llu MB), %llu MB free, "
           "%zu free blocks",
           (unsigned long long)heapBegin_, (unsigned long long)heapEnd_, committed_.size(),
           (unsigned long long)(sum_committed(committed_) >> 20), (unsigned long long)(fr >> 20),
           free_.size());
}

}  // namespace hle
}  // namespace x360
