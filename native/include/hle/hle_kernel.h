/**
 * HLE kernel surface: objects, memory, threading, filesystem, input, GPU doorbell.
 *
 * Everything a title can call lands here. The dispatch is driven by the *real*
 * kernel ordinals (see x360/kernel_ordinals.h), so an unimplemented entry point
 * is named in the log instead of being a silent no-op.
 */
#pragma once
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hle/guest_heap.h"
#include "loader/xex_loader.h"

namespace x360 {

// Forward declaration of PPCContext from jit namespace
namespace jit { struct PPCContext; }

// Forward declaration of the disc mount (loader layer).
class IsoParser;

namespace hle {

// ─── Kernel Object Types ───────────────────────────────────────────────────────
enum class KernelObjectType : uint32_t {
  Thread = 0x01,
  Mutex = 0x02,
  Semaphore = 0x03,
  Event = 0x04,
  Timer = 0x05,
  File = 0x06,
  IoCompletion = 0x07,
  Section = 0x08,
  SymbolicLink = 0x09,
};

struct KernelObject {
  uint32_t handle = 0;
  KernelObjectType type = KernelObjectType::Thread;
  std::string name;
  virtual ~KernelObject() = default;
};

struct KernelThread : KernelObject {
  uint64_t startPc = 0;
  uint64_t startParam = 0;
  uint64_t stackBase = 0;
  uint32_t stackSize = 0;
  int xenonThreadIndex = -1;
  bool running = false;
  bool suspended = false;
  bool exited = false;
  uint32_t exitCode = 0;
  std::mutex m;
  std::condition_variable cv;
};

struct KernelMutex : KernelObject {
  std::recursive_mutex m;
  uint32_t ownerThread = 0;
  int recursionCount = 0;
};

struct KernelSemaphore : KernelObject {
  int count = 0;
  int maxCount = 0;
  std::mutex m;
  std::condition_variable cv;
};

struct KernelEvent : KernelObject {
  bool signaled = false;
  bool manualReset = false;
  std::mutex m;
  std::condition_variable cv;
};

struct KernelTimer : KernelObject {
  std::chrono::steady_clock::time_point due{};
  bool periodic = false;
  int64_t periodMs = 0;
  bool signaled = false;
  std::mutex m;
  std::condition_variable cv;
};

struct KernelFile : KernelObject {
  FILE* hostFile = nullptr;
  std::string guestPath;
  std::string hostPath;
  bool isDirectory = false;
  // Disc-backed file (read-only, no host FILE*): served from the mounted ISO.
  const void* isoEntry = nullptr;   // opaque IsoFileEntry*
  uint64_t isoOffset = 0;
  uint64_t isoSize = 0;
};

// What the guest's GPU command stream has told us so far (read by the engine).
struct GpuDoorbell {
  std::atomic<uint64_t> ringBufferGuestAddr{0};
  std::atomic<uint32_t> ringBufferSize{0};
  std::atomic<uint64_t> lastWriteGuestAddr{0};
  std::atomic<uint64_t> swapRequests{0};
};

class HleKernel {
 public:
  HleKernel();
  ~HleKernel();

  bool init(uint8_t* guestMemory, uint64_t guestBase, const std::string& gameDir,
             const std::string& saveDir);
  void shutdown();

  // ── Loaded image + guest-side helpers ──────────────────────────────────────
  void attachImage(const XexImage* img) { m_image = img; }
  const XexImage* image() const { return m_image; }
  // Install the runtime-owned thunks (HLE call table) for a loaded image.
  bool installImportTable(std::vector<XexImage::Import>& imports);

  // ── Memory ─────────────────────────────────────────────────────────────────
  GuestHeap& heap() { return m_heap; }
  uint64_t allocVirtualMemory(uint64_t baseAddr, uint64_t size, uint32_t type, uint32_t protect);
  void freeVirtualMemory(uint64_t baseAddr);
  uint64_t xMemAlloc(uint64_t size, uint64_t params);
  void xMemFree(uint64_t baseAddr);

  // ── Threading ──────────────────────────────────────────────────────────────
  uint32_t createThread(uint64_t startAddress, uint64_t startParam, uint32_t stackSize,
                        int priority, bool createSuspended);
  void terminateThread(uint32_t handle, uint32_t exitCode);
  uint32_t createMutex(bool initialOwner);
  void releaseMutex(uint32_t handle);
  uint32_t createSemaphore(int initialCount, int maxCount);
  void releaseSemaphore(uint32_t handle, int releaseCount);
  uint32_t createEvent(bool manualReset, bool initialState);
  void setEvent(uint32_t handle);
  void resetEvent(uint32_t handle);
  void pulseEvent(uint32_t handle);
  uint32_t waitForObject(uint32_t handle, bool alertable, uint64_t timeoutMs);
  uint32_t waitForMultiple(const std::vector<uint32_t>& handles, bool waitAll, bool alertable,
                            uint64_t timeoutMs);

  // ── File System ────────────────────────────────────────────────────────────
  // `disc` may be null; when set, paths under the disc mount are served from it.
  void mountDisc(IsoParser* disc) { m_disc = disc; }
  IsoParser* disc() const { return m_disc; }
  std::string resolveGuestPath(const std::string& guestPath) const;
  uint32_t createFile(const std::string& guestPath, uint32_t access, uint32_t shareMode,
                      uint32_t createDisposition);
  bool readFile(uint32_t handle, void* buffer, uint32_t size, uint32_t* bytesRead);
  bool writeFile(uint32_t handle, const void* buffer, uint32_t size, uint32_t* bytesWritten);
  bool readFileAt(uint32_t handle, uint64_t offset, void* buffer, uint32_t size,
                  uint32_t* bytesRead);
  void closeHandle(uint32_t handle);
  uint64_t getFileSize(uint32_t handle);
  bool setFilePointer(uint32_t handle, int64_t distanceToMove, uint32_t moveMethod);
  // Directory enumeration used by NtQueryDirectoryFile.
  struct DirEntry {
    std::string name;
    uint64_t size = 0;
    bool isDirectory = false;
  };
  bool enumerateDirectory(const std::string& guestPath, std::vector<DirEntry>* out);

  // ── Input ──────────────────────────────────────────────────────────────────
  void setInputState(int pad, uint32_t buttons, int16_t lx, int16_t ly, int16_t rx,
                     int16_t ry, uint8_t lt, uint8_t rt);
  // Writes a big-endian XINPUT_STATE into guest memory; returns an Xbox error code.
  uint32_t getInputState(int pad, void* outState);
  void setRumble(int pad, uint16_t lowFreq, uint16_t highFreq);
  uint16_t rumbleLow(int pad) const;
  uint16_t rumbleHigh(int pad) const;

  // ── GPU ────────────────────────────────────────────────────────────────────
  GpuDoorbell& gpuDoorbell() { return m_gpu; }
  // Called by the engine when the guest submits a PM4 buffer.
  std::function<void(uint64_t guestAddr, uint32_t sizeDwords)> onPm4Submit;
  std::function<void()> onSwap;

  // ── Kernel dispatch (called from the JIT via the HLE sentinel) ──────────────
  void dispatchKernelCall(const std::string& module, uint32_t ordinal,
                          jit::PPCContext& ctx);
  // Resolve one import (called at load time by the loader/bridge).
  bool resolveImport(XexImage::Import& imp);

  // Guest memory access helpers (big-endian, bounds-checked, fault-logging).
  uint8_t* guest_ptr(uint64_t addr) const;
  bool read_u32(uint64_t addr, uint32_t* out) const;
  bool write_u32(uint64_t addr, uint32_t value) const;
  bool read_u64(uint64_t addr, uint64_t* out) const;
  bool write_u64(uint64_t addr, uint64_t value) const;
  bool read_string(uint64_t addr, uint32_t maxLen, std::string* out) const;      // ASCII
  bool read_utf16(uint64_t addr, uint32_t byteLen, std::string* out) const;       // -> UTF-8
  // OBJECT_ATTRIBUTES (guest) -> path
  bool read_object_attributes_path(uint64_t objAttrAddr, std::string* out) const;

  const std::string& lastError() const { return m_lastError; }
  uint64_t guest_base() const { return m_guestBase; }
  uint8_t* guest_memory() const { return m_guestMemory; }

  // Diagnostics summary (heap + objects) for crash reports / UI.
  void describe(char* out, size_t max) const;

 private:
  uint32_t allocHandle();
  KernelObject* getObject(uint32_t handle);
  void destroyObject(uint32_t handle);

  uint8_t* m_guestMemory = nullptr;
  uint64_t m_guestBase = 0;
  uint64_t m_guestSize = 0;
  std::string m_gameDir;
  std::string m_saveDir;
  const XexImage* m_image = nullptr;
  IsoParser* m_disc = nullptr;
  GuestHeap m_heap;

  std::unordered_map<uint32_t, std::unique_ptr<KernelObject>> m_objects;
  std::mutex m_objectMutex;
  uint32_t m_nextHandle = 0x0100;

  std::unordered_map<std::string, std::string> m_pathMappings;

  struct XInputState {
    uint32_t buttons = 0;
    int16_t lx = 0, ly = 0, rx = 0, ry = 0;
    uint8_t lt = 0, rt = 0;
    uint16_t rumbleLow = 0, rumbleHigh = 0;
    std::mutex m;
  } m_inputState[4];

  GpuDoorbell m_gpu;
  std::string m_lastError;
};

}  // namespace hle
}  // namespace x360
