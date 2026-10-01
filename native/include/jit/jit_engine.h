#pragma once
/**
 * JIT engine: PowerPC Xenon -> ARM64 translation, block cache, dispatch loop and
 * the interpreter that backs it up.
 *
 * What this replaced (all of these were fatal for a real title):
 *  - guest state mapped onto host registers (PPC r0..r27 -> x0..x27) with no
 *    register save/restore, so any HLE call or FP use clobbered guest state;
 *  - loads treating PPC rA as a host pointer instead of a guest address;
 *  - an unimplemented opcode compiled to `nop` (silent wrong behaviour) and an
 *    empty block making the dispatcher advance PC by 4 forever;
 *  - a fixed 64 MB code arena with a non-atomic bump allocator and no bounds
 *    check on the returned pointer;
 *  - kernel imports recognised by "PC >= 0x80000000", which is the guest *code*
 *    range too (XEX base is 0x82000000) — i.e. real game code was treated as an
 *    HLE call, and real HLE sentinels were not.
 *
 * Now: PPCContext is the only architectural state; the ARM64 backend is
 * "context memory-operand" (load from [x28], compute, store back), it declines
 * anything it cannot prove and the interpreter executes that instead; kernel
 * calls go through the loader's HLE sentinels; the arena grows in chunks with a
 * bounds-checked allocator; and no generated code may fault because every memory
 * access is range-checked against the guest window.
 */
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ppc_decode.h"

namespace x360 {
class XexImage;
namespace hle {
class HleKernel;
}
namespace jit {

// PowerPC register file per-thread context.
//
// The ARM64 backend encodes LDR/STR against offsetof(PPCContext, ...) and only
// uses 12-bit unsigned offsets, so the scalar state must stay in the first
// 3 KB of this struct (static_assert in arm64_backend.cpp enforces it). The VMX
// file lives beyond that window and is reached through a dedicated base register.
struct PPCContext {
  uint64_t gpr[32];      // General purpose registers r0-r31 (full 64-bit PPC)
  double fpr[32];        // Floating point registers f0-f31
  uint32_t cr;           // Condition register (field 0 in the most significant nibble)
  uint32_t xer;          // Fixed point exception register (SO/OV/CA)
  uint32_t vscr;         // VMX status and control register
  uint32_t fpscr;        // Floating point status/control
  uint64_t ctr;          // Count register
  uint64_t lr;           // Link register
  uint64_t pc;           // Next guest PC to execute
  uint32_t msr;          // Machine state register
  uint32_t threadIndex;  // Xenon hardware thread index (0-5)
  uint32_t reserveAddr;  // guest address of the lwarx/ldarx reservation
  uint32_t reserveValid;  // 1 when a reservation is armed
  uint32_t exitCode;     // BlockExit written by generated code / the interpreter
  uint32_t hleIndex;     // Pending HLE import index (sentinel payload)
  uint32_t haltReason;
  uint32_t pad0;
  uint8_t vmx[128][16];  // VMX128 vector registers vr0-vr127

  // Host-side view of the guest address space (identical for all threads).
  uint8_t* hostMem;
  uint64_t guestBase;
  uint64_t guestSize;
};

// Host-side parameters the code generator needs to address guest memory.
struct MemCtx {
  uint8_t* guestMemory = nullptr;
  uint64_t guestBase = 0;
  uint64_t guestSize = 0;
  class JitEngine* engine = nullptr;
};

// What a compiled block (or the interpreter) reports back.
enum BlockExit : uint32_t {
  kExitContinue = 0,  // fall through: ctx.pc already advanced
  kExitReturn = 1,    // blr to the guest-return sentinel -> thread exits
  kExitHle = 2,       // ctx.hleIndex holds the import index to dispatch
  kExitError = 3,     // unrecoverable (bad PC, bad access, unimplemented op)
  kExitYield = 4,     // interpreter exhausted its instruction budget
};

// HLE import sentinels: the loader stores `kHleSentinelBase + importIndex` in
// each unresolved/resolved import slot, so a guest branch to that address is how
// a kernel call is recognised. This range is outside every real XEX image (they
// load at 0x80000000..0x9FFFFFFF but a 4 GB window only maps what the title
// allocated; the sentinel page is never written with real code).
static constexpr uint32_t kHleSentinelBase = 0x8FFE0000u;  // == loader's constant
static constexpr uint32_t kHleMaxImports = 0x10000;         // (asserted in the engine)

// Guest-resident thunks installed by the runtime so that "return from HLE" has
// a real address, and a guest `blr` into a kernel module never reaches memory.
static constexpr uint32_t kGuestReturnSentinel = 0x8FFFF000u;
static constexpr uint32_t kHleThunkBase = 0x8FFFF100u;  // 8 bytes per import
static constexpr uint32_t kHleThunkMax = 0x400;

enum class JitState { Stopped, Running, Paused, Error };

// Handler for one resolved kernel import.
using HleHandler = std::function<void(PPCContext& ctx, uint32_t importIndex)>;

class JitEngine {
 public:
  JitEngine();
  ~JitEngine();

  bool init(uint8_t* guestMemory, uint64_t guestMemorySize, uint64_t guestBase,
            uint64_t entryPoint);

  void setKernel(hle::HleKernel* kernel) { m_kernel = kernel; }
  hle::HleKernel* kernel() const { return m_kernel; }
  // Called by the kernel when the guest wants a thread to run a function.
  void setThreadStarter(std::function<bool(uint64_t pc, uint64_t r3, uint64_t stackTop)> f) {
    m_threadStarter = std::move(f);
  }
  // Called when a guest thread finishes waiting on a kernel object.
  void setWaitHandler(std::function<uint32_t(PPCContext&, uint32_t, uint64_t)> f) {
    m_waitHandler = std::move(f);
  }

  bool start(int threadCount = 1);
  void pause();
  void resume();
  void stop();

  // Thread 0 boot: pc = XEX entry point, r3/r4 = the standard loader arguments.
  bool startThread0(uint64_t entryPc, uint64_t stackTop);
  // Run a guest function on a fresh thread (ExCreateThread). Returns false if no
  // slot is free. `slotOut` receives the slot index for termination.
  bool runGuestFunction(uint64_t pc, uint64_t r3, uint64_t stackTop, int* slotOut);
  void requestStop(int slot);

  JitState state() const { return m_state.load(); }
  bool isRunning() const { return m_state.load() == JitState::Running; }

  // HLE import table. `index` is the sentinel offset chosen by the loader, so
  // registration order must match XexImage::imports order (the loader does this).
  bool registerHleImport(uint32_t index, const std::string& module, uint32_t ordinal,
                         HleHandler handler);
  void registerHle(const std::string& module, uint32_t ordinal, HleHandler handler,
                   const std::string& name = {});
  bool hasImport(uint32_t index) const;
  bool isImportResolved(uint32_t index) const;
  const std::string* importName(uint32_t index) const;
  // Guest address of the import slot's *value* (data imports) / thunk (funcs).
  uint32_t importThunkAddress(uint32_t index) const;
  // Called at boot: writes the runtime thunks into the guest image and installs
  // the handlers for every import in the loaded image.
  bool installImportThunks(XexImage* image);

  void invalidateRange(uint64_t guestAddr, size_t size);

  uint64_t blocksCompiled() const { return m_blocksCompiled.load(); }
  uint64_t instructionsExecuted() const { return m_instrExecuted.load(); }
  uint64_t interpreterBlocks() const { return m_interpBlocks.load(); }
  uint64_t unimplementedOpcodes() const { return m_unimplementedOpcodes.load(); }
  double currentFps() const { return m_fps.load(); }
  double frameTimeMs() const { return m_frameTimeMs.load(); }
  void noteSwap() { m_swapTick.fetch_add(1, std::memory_order_relaxed); }
  double jitCoverage() const;
  size_t arenaBytesUsed() const { return m_arenaUsed.load(); }
  size_t arenaBytesTotal() const;
  uint32_t threadCount() const { return uint32_t(m_activeThreads.load()); }
  std::string threadStatus() const;
  // Whether the ARM64 backend is usable in this process (false on non-ARM64
  // hosts and in "interp only" debug mode).
  bool jitEnabled() const { return m_jitEnabled; }
  void setJitEnabled(bool on) { m_jitEnabled = on; }

 private:
  struct ArenaChunk {
    uint8_t* base = nullptr;
    size_t size = 0;
    std::atomic<size_t> used{0};
  };

  struct CompiledBlock {
    uint64_t guestPc = 0;
    uint8_t* code = nullptr;
    size_t codeSize = 0;
    uint32_t guestBytes = 0;   // span of guest code covered (for invalidation)
    uint32_t instrCount = 0;
    uint32_t guestHash = 0;
    uint64_t generation = 0;
    bool usable = false;       // false => always interpreter
  };

  struct ThreadSlot {
    PPCContext ctx;
    std::thread host;
    std::atomic<bool> busy{false};
    std::atomic<bool> requestExit{false};
    uint32_t kernelHandle = 0;
  };

  uint8_t* allocCode(size_t bytes, ArenaChunk** outChunk);
  void flushICache(uint8_t* start, size_t size);

  void threadLoop(int slot);
  CompiledBlock* findBlock(uint64_t guestPc) const;
  CompiledBlock* compileBlock(uint64_t guestPc, PPCContext& ctx);
  void dispatchHle(PPCContext& ctx, uint32_t importIndex);
  int acquireSlot();
  void releaseSlot(int slot);

  uint8_t* m_guestMemory = nullptr;
  uint64_t m_guestMemorySize = 0;
  uint64_t m_guestBase = 0;
  uint64_t m_entryPoint = 0;
  hle::HleKernel* m_kernel = nullptr;
  std::function<bool(uint64_t, uint64_t, uint64_t)> m_threadStarter;
  std::function<uint32_t(PPCContext&, uint32_t, uint64_t)> m_waitHandler;
  std::unique_ptr<MemCtx> m_mem;

  std::unordered_map<uint64_t, std::unique_ptr<CompiledBlock>> m_blockCache;
  std::mutex m_cacheMutex;
  uint64_t m_generation = 1;

  std::vector<std::unique_ptr<ArenaChunk>> m_arena;
  std::atomic<size_t> m_arenaUsed{0};
  std::mutex m_arenaMutex;
  size_t m_chunkSize = 2 * 1024 * 1024;
  size_t m_maxTotal = 256ull * 1024 * 1024;

  struct ImportEntry {
    HleHandler handler;
    std::string module;
    std::string name;
    uint32_t ordinal = 0;
    bool resolved = false;
  };
  std::vector<ImportEntry> m_imports;
  std::unordered_map<uint64_t, uint32_t> m_importIndexByKey;
  std::mutex m_hleMutex;

  std::vector<std::unique_ptr<ThreadSlot>> m_slots;
  std::atomic<uint32_t> m_activeThreads{0};
  std::atomic<JitState> m_state{JitState::Stopped};
  std::atomic<bool> m_pauseRequested{false};
  // Default off: see JitEngine::init. Set X360_XBOX_JIT=1 or the `debug.x360.jit`
  // system property to translate guest blocks to ARM64.
  bool m_jitEnabled = false;

  std::atomic<uint64_t> m_blocksCompiled{0};
  std::atomic<uint64_t> m_instrExecuted{0};
  std::atomic<uint64_t> m_interpBlocks{0};
  std::atomic<uint64_t> m_unimplementedOpcodes{0};
  std::atomic<uint64_t> m_swapTick{0};
  std::atomic<double> m_fps{0.0};
  std::atomic<double> m_frameTimeMs{0.0};
};

// ─── Interpreter + backend contracts (jit_interpreter.cpp / arm64_backend.cpp) ─
// Execute a single guest instruction at ctx.pc. Returns false and sets
// ctx.exitCode when the encoding is unknown or the access was out of range.
bool stepOneGuestInstruction(PPCContext& ctx);
// Interpret up to `maxInstrs` instructions, stopping on HLE/error/return.
uint32_t interpretRange(PPCContext& ctx, uint32_t maxInstrs, uint64_t stopPc);
// True when `guestPc` is an HLE import sentinel (defined in ppc_decode.cpp; the
// convention is owned by loader/xex_loader.h).
bool isHleSentinel(uint64_t guestPc, uint32_t* outImportIndex);
// True when the ARM64 backend can execute this class natively.
bool backendSupports(PpcClass cls);
// Emit ARM64 code for the guest range starting at guestPc. Returns the number of
// bytes written (0 => the block must run interpreted).
using BlockFn = uint32_t (*)(PPCContext*);
// The backend writes a 4-byte prefix (instruction count) then the code, so
// `out` must have room for 4 extra bytes; outGuestBytes receives the span.
size_t jitCompileBlock(uint64_t guestPc, uint8_t* out, size_t maxBytes, MemCtx& mem,
                       uint32_t* outInstrCount, uint32_t* outGuestBytes);

}  // namespace jit
}  // namespace x360
