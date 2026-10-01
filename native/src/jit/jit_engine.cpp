/**
 * JIT Engine — dispatch loop, code arena, block cache, HLE import table,
 * Xenon thread slots, and FPS accounting.
 *
 * See include/jit/jit_engine.h for the design notes (and for what the previous
 * version got wrong). Everything guest-visible goes through PPCContext, so the
 * interpreter and the generated code share one state model.
 */
#include "../../include/jit/jit_engine.h"

#include "../../include/hle/hle_kernel.h"
#include "../../include/loader/xex_loader.h"
#include "../../include/x360/diagnostics.h"
#include "../../include/x360/kernel_ordinals.h"
#include "../../include/x360/log.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <sys/mman.h>
#if defined(__ANDROID__)
#include <sys/system_properties.h>  // PROP_VALUE_MAX, __system_property_get
#endif

static_assert(x360::jit::kHleSentinelBase == x360::kHleSentinelBase,
              "the JIT and the XEX loader must agree on the HLE sentinel window");
static_assert(x360::jit::kHleMaxImports == x360::kHleMaxImports, "import capacity mismatch");

#define TAG "X360:JIT"

namespace x360 {
namespace jit {

namespace {
constexpr int kMaxSlots = 8;  // 6 hardware threads + headroom for HLE worker threads
}

JitEngine::JitEngine() {
  m_mem = std::make_unique<MemCtx>();
  for (int i = 0; i < kMaxSlots; i++) m_slots.push_back(std::make_unique<ThreadSlot>());
}

JitEngine::~JitEngine() {
  stop();
  for (auto& c : m_arena) {
    if (c && c->base) munmap(c->base, c->size);
  }
  m_arena.clear();
}

bool JitEngine::init(uint8_t* guestMemory, uint64_t guestMemorySize, uint64_t guestBase,
                     uint64_t entryPoint) {
  m_guestMemory = guestMemory;
  m_guestMemorySize = guestMemorySize;
  m_guestBase = guestBase;
  m_entryPoint = entryPoint;
  m_mem->guestMemory = guestMemory;
  m_mem->guestBase = guestBase;
  m_mem->guestSize = guestMemorySize;
  m_mem->engine = this;
  m_state.store(JitState::Stopped);
  m_pauseRequested.store(false);

  // Code generation is opt-in until it has been validated by *executing* on real
  // hardware. `tools/check_arm64_asm.py` proves the emitter writes the instructions it
  // means, and the interpreter is the reference for semantics, but only a device can
  // prove the generated code runs correctly. The interpreter stays available either way
  // (every declined instruction is interpreted), so the default is the path whose
  // behaviour we can stand behind, and a divergence is always reproducible by turning
  // the JIT on or off:
  //   adb shell setprop debug.x360.jit 1      (or X360_XBOX_JIT=1 for host runs)
  const char* jitEnv = getenv("X360_XBOX_JIT");
  bool wantJit = jitEnv && (!strcmp(jitEnv, "1") || !strcmp(jitEnv, "on") ||
                            !strcmp(jitEnv, "yes"));
#if defined(__ANDROID__)
  char prop[PROP_VALUE_MAX] = {0};
  if (__system_property_get("debug.x360.jit", prop) > 0) {
    wantJit = prop[0] == '1' || !strcmp(prop, "on") || !strcmp(prop, "yes");
  }
#endif
  m_jitEnabled = wantJit;
  X360LOGI(TAG, "initialised: guest 0x%08llX..0x%08llX entry 0x%08llX jit=%s",
           (unsigned long long)guestBase, (unsigned long long)(guestBase + guestMemorySize),
           (unsigned long long)entryPoint, m_jitEnabled ? "on" : "off (interpreter only)");
  if (!m_jitEnabled) return true;
#if !defined(__aarch64__)
  X360LOGW(TAG, "host is not ARM64 — the code generator is disabled, running interpreted");
#endif
  return true;
}

// ─── Executable arena ─────────────────────────────────────────────────────────
uint8_t* JitEngine::allocCode(size_t bytes, ArenaChunk** outChunk) {
  std::lock_guard<std::mutex> lk(m_arenaMutex);
  if (bytes == 0 || bytes > m_chunkSize / 2) return nullptr;
  // Try the newest chunk first; allocate a new one when it is full.
  while (true) {
    ArenaChunk* last = m_arena.empty() ? nullptr : m_arena.back().get();
    if (last) {
      size_t off = last->used.fetch_add(bytes);
      if (off + bytes <= last->size) {
        m_arenaUsed.fetch_add(bytes, std::memory_order_relaxed);
        *outChunk = last;
        return last->base + off;
      }
      last->used.store(last->size);  // exhausted: stop handing out of it
    }
    if (m_arenaUsed.load() + m_chunkSize > m_maxTotal) {
      X360LOGE(TAG, "code arena exhausted (%zu MB of %zu MB) — continuing interpreted",
               m_arenaUsed.load() >> 20, m_maxTotal >> 20);
      return nullptr;
    }
    auto chunk = std::make_unique<ArenaChunk>();
    chunk->size = m_chunkSize;
    chunk->base = static_cast<uint8_t*>(
        mmap(nullptr, chunk->size, PROT_READ | PROT_WRITE | PROT_EXEC,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (chunk->base == MAP_FAILED) {
      X360LOGE(TAG, "mmap of a %zu MB executable chunk failed: %s", chunk->size, strerror(errno));
      chunk->base = nullptr;
      return nullptr;
    }
    ArenaChunk* raw = chunk.get();
    m_arena.push_back(std::move(chunk));
    *outChunk = raw;
    size_t off = raw->used.fetch_add(bytes);
    if (off + bytes <= raw->size) {
      m_arenaUsed.fetch_add(bytes, std::memory_order_relaxed);
      return raw->base + off;
    }
  }
}

size_t JitEngine::arenaBytesTotal() const {
  size_t n = 0;
  for (const auto& c : m_arena) n += c ? c->size : 0;
  return n;
}

void JitEngine::flushICache(uint8_t* start, size_t size) {
#if defined(__aarch64__)
  __builtin___clear_cache(reinterpret_cast<char*>(start),
                          reinterpret_cast<char*>(start + size));
#else
  (void)start;
  (void)size;
#endif
}

double JitEngine::jitCoverage() const {
  const uint64_t j = m_blocksCompiled.load();
  const uint64_t i = m_interpBlocks.load();
  return (j + i) ? double(j) / double(j + i) : 0.0;
}

// ─── Block cache ──────────────────────────────────────────────────────────────
JitEngine::CompiledBlock* JitEngine::findBlock(uint64_t guestPc) const {
  auto it = m_blockCache.find(guestPc);
  if (it == m_blockCache.end()) return nullptr;
  return it->second.get();
}

JitEngine::CompiledBlock* JitEngine::compileBlock(uint64_t guestPc, PPCContext& ctx) {
  (void)ctx;
  auto block = std::make_unique<CompiledBlock>();
  block->guestPc = guestPc;
  block->generation = m_generation;

  const uint64_t off = guestPc - m_guestBase;
  if (off >= m_guestMemorySize) {
    block->usable = false;
    block->guestBytes = 4;
    CompiledBlock* raw = block.get();
    std::lock_guard<std::mutex> lk(m_cacheMutex);
    m_blockCache[guestPc] = std::move(block);
    return raw;
  }

  if (m_jitEnabled) {
    // Reserve the maximum a block can need, then trim. One instruction per guest
    // word for the supported subset, plus prologue/epilogue.
    constexpr size_t kMaxBytesPerInstr = 48;
    constexpr size_t kPrefix = 16;
    const size_t budget = kPrefix + kMaxBytesPerInstr * 64;
    ArenaChunk* chunk = nullptr;
    uint8_t* slot = allocCode(budget, &chunk);
    if (slot) {
      uint32_t instrCount = 0, guestBytes = 0;
      const size_t n = jitCompileBlock(guestPc, slot + kPrefix, budget - kPrefix, *m_mem,
                                       &instrCount, &guestBytes);
      if (n >= 4 && instrCount > 0) {
        // Prefix: [0]=instruction count, [4]=code size, code starts at +16 so the
        // entry address stays 16-byte aligned.
        memcpy(slot, &instrCount, 4);
        memcpy(slot + 4, &n, 4);
        block->code = slot + kPrefix;
        block->codeSize = n;
        block->instrCount = instrCount;
        block->guestBytes = guestBytes ? guestBytes : instrCount * 4;
        block->usable = true;
        flushICache(slot, kPrefix + n);
        uint32_t h = 2166136261u;
        for (uint32_t i = 0; i < block->guestBytes; i += 4) {
          uint32_t word;
          memcpy(&word, m_guestMemory + off + i, 4);
          h = (h ^ word) * 16777619u;
        }
        block->guestHash = h;
        m_blocksCompiled.fetch_add(1, std::memory_order_relaxed);
      } else {
        // Roll back the reservation: the arena is a bump allocator, so returning
        // the space keeps it from filling with rejected blocks.
        if (chunk) {
          size_t want = chunk->used.load();
          chunk->used.compare_exchange_weak(want, want - budget);
        }
        m_arenaUsed.fetch_sub(budget, std::memory_order_relaxed);
      }
    }
  }
  if (!block->usable) {
    // Remember that this PC is interpreter-only, so the dispatcher does not retry
    // compilation on every pass (that was a 100%-CPU hot loop in the old code).
    block->guestBytes = 4;
  }
  CompiledBlock* raw = block.get();
  std::lock_guard<std::mutex> lk(m_cacheMutex);
  m_blockCache[guestPc] = std::move(block);
  return raw;
}

void JitEngine::invalidateRange(uint64_t guestAddr, size_t size) {
  std::lock_guard<std::mutex> lk(m_cacheMutex);
  m_generation++;
  const uint64_t lo = guestAddr;
  const uint64_t hi = guestAddr + size;
  for (auto it = m_blockCache.begin(); it != m_blockCache.end();) {
    const uint64_t bLo = it->first;
    const uint64_t bHi = bLo + it->second->guestBytes;
    if (bLo < hi && bHi > lo)
      it = m_blockCache.erase(it);
    else
      ++it;
  }
}

// ─── HLE import table ─────────────────────────────────────────────────────────
bool JitEngine::registerHleImport(uint32_t index, const std::string& module, uint32_t ordinal,
                                  HleHandler handler) {
  std::lock_guard<std::mutex> lk(m_hleMutex);
  if (index >= kHleMaxImports) return false;
  if (m_imports.size() <= index) m_imports.resize(index + 1);
  ImportEntry e;
  e.handler = std::move(handler);
  e.module = module;
  e.ordinal = ordinal;
  e.resolved = true;
  e.name = kernel::name_for_ordinal(module.c_str(), ordinal);
  m_imports[index] = std::move(e);
  m_importIndexByKey[(uint64_t(std::hash<std::string>{}(module)) << 16) ^ ordinal] = index;
  return true;
}

void JitEngine::registerHle(const std::string& module, uint32_t ordinal, HleHandler handler,
                            const std::string& name) {
  std::lock_guard<std::mutex> lk(m_hleMutex);
  const uint64_t key = (uint64_t(std::hash<std::string>{}(module)) << 16) ^ ordinal;
  auto it = m_importIndexByKey.find(key);
  uint32_t idx;
  if (it != m_importIndexByKey.end()) {
    idx = it->second;
  } else {
    idx = uint32_t(m_imports.size());
    m_importIndexByKey[key] = idx;
  }
  if (m_imports.size() <= idx) m_imports.resize(idx + 1);
  ImportEntry e;
  e.handler = std::move(handler);
  e.module = module;
  e.ordinal = ordinal;
  e.resolved = true;
  e.name = name.empty() ? kernel::name_for_ordinal(module.c_str(), ordinal) : name;
  m_imports[idx] = std::move(e);
  kernel::mark_implemented(module.c_str(), ordinal);
  X360LOGI(TAG, "HLE handler registered: %s #%u ordinal 0x%X (%s)", module.c_str(), idx,
           ordinal, e.name.c_str());
}

bool JitEngine::hasImport(uint32_t index) const {
  return index < m_imports.size() && m_imports[index].handler != nullptr;
}
bool JitEngine::isImportResolved(uint32_t index) const {
  return index < m_imports.size() && m_imports[index].resolved;
}
const std::string* JitEngine::importName(uint32_t index) const {
  if (index >= m_imports.size()) return nullptr;
  return &m_imports[index].name;
}
uint32_t JitEngine::importThunkAddress(uint32_t index) const {
  return index < kHleThunkMax ? (kHleThunkBase + index * 8) : 0;
}

bool JitEngine::installImportThunks(XexImage* image) {
  if (!image || !m_guestMemory) return false;
  // 1) every import slot in the PE image already holds its sentinel address
  //    (written by the loader); we only need to make the *code* side safe: a
  //    guest `mtlr; blr` into a sentinel must not fetch memory at 0x8FFE....
  // 2) install a `blr` thunk for each import so that even a hand-written or
  //    patched XEX that calls 0x8FFFF1xx works.
  const uint32_t blr = __builtin_bswap32(0x4E800020u);
  for (uint32_t i = 0; i < kHleThunkMax; i++) {
    const uint32_t guest = kHleThunkBase + i * 8;
    if (guest >= m_guestBase && guest + 8 <= m_guestBase + m_guestMemorySize) {
      memcpy(m_guestMemory + (guest - m_guestBase), &blr, 4);
      const uint32_t zero = 0;
      memcpy(m_guestMemory + (guest - m_guestBase) + 4, &zero, 4);
    }
  }
  // 3) bind import index -> module/ordinal so the dispatcher can name them
  size_t bound = 0;
  for (size_t i = 0; i < image->imports.size() && i < kHleMaxImports; i++) {
    const auto& imp = image->imports[i];
    std::lock_guard<std::mutex> lk(m_hleMutex);
    if (m_imports.size() <= i) m_imports.resize(i + 1);
    m_imports[i].module = imp.library;
    m_imports[i].ordinal = imp.ordinal;
    if (m_imports[i].name.empty()) {
      m_imports[i].name = imp.name.empty()
                              ? (kernel::name_for_ordinal(imp.library.c_str(), imp.ordinal))
                              : imp.name;
    }
    if (m_imports[i].handler) {
      m_imports[i].resolved = true;
      bound++;
    }
  }
  X360LOGI(TAG, "%zu imports installed, %zu with a runtime handler", image->imports.size(), bound);
  return true;
}

void JitEngine::dispatchHle(PPCContext& ctx, uint32_t importIndex) {
  HleHandler handler;
  std::string moduleName;
  uint32_t ordinal = 0;
  {
    std::lock_guard<std::mutex> lk(m_hleMutex);
    if (importIndex < m_imports.size()) {
      handler = m_imports[importIndex].handler;
      moduleName = m_imports[importIndex].module;
      ordinal = m_imports[importIndex].ordinal;
    }
  }
  const uint64_t callSite = ctx.pc;  // PC of the call, not the sentinel
  if (!handler) {
    char buf[192];
    const char* name = importIndex < m_imports.size() && !m_imports[importIndex].name.empty()
                           ? m_imports[importIndex].name.c_str()
                           : kernel::name_for_ordinal(moduleName.empty() ? "xboxkrnl.exe"
                                                                         : moduleName.c_str(),
                                                      ordinal);
    snprintf(buf, sizeof(buf), "%s #%u 0x%04X %s (r3=0x%llX)", moduleName.c_str(), importIndex,
             ordinal, name, (unsigned long long)ctx.gpr[3]);
    if (diag::report_unsupported(diag::Subsystem::Kernel, buf)) {
      X360LOGE(TAG, "UNSUPPORTED KERNEL CALL: %s", buf);
    }
    // Policy: unknown imports return ERROR_SUCCESS so the title keeps running and
    // the *first* offender is what shows up in the log. A silent success is exactly
    // the kind of thing that hides bugs, so it is logged once per call site.
    ctx.gpr[3] = 0;
    ctx.pc = ctx.lr;
    return;
  }
  handler(ctx, importIndex);
  // The handler either set ctx.pc (e.g. to LR) or left it at the call site; make
  // sure we never re-enter the sentinel forever.
  if ((ctx.pc & 0xFFFF0000u) == kHleSentinelBase || ctx.pc == callSite) ctx.pc = ctx.lr;
  if (ctx.pc == kGuestReturnSentinel) ctx.exitCode = kExitReturn;
}

// ─── Threads ──────────────────────────────────────────────────────────────────
int JitEngine::acquireSlot() {
  for (size_t i = 0; i < m_slots.size(); i++) {
    bool expected = false;
    if (m_slots[i]->busy.compare_exchange_strong(expected, true)) {
      m_activeThreads.fetch_add(1, std::memory_order_relaxed);
      return int(i);
    }
  }
  return -1;
}

void JitEngine::releaseSlot(int slot) {
  if (slot < 0 || size_t(slot) >= m_slots.size()) return;
  if (m_slots[slot]->busy.exchange(false)) {
    m_activeThreads.fetch_sub(1, std::memory_order_relaxed);
  }
}

bool JitEngine::startThread0(uint64_t entryPc, uint64_t stackTop) {
  if (m_state.load() != JitState::Stopped) stop();
  m_state.store(JitState::Running);
  m_pauseRequested.store(false);
  const int slot = acquireSlot();
  if (slot < 0) {
    m_state.store(JitState::Error);
    return false;
  }
  ThreadSlot* ts = m_slots[slot].get();
  PPCContext& ctx = ts->ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.threadIndex = 0;
  ctx.pc = entryPc;
  ctx.gpr[1] = stackTop;
  ctx.gpr[3] = 0xFFFFFFFFu;  // "no savegame / not from dashboard"
  ctx.hostMem = m_guestMemory;
  ctx.guestBase = m_guestBase;
  ctx.guestSize = m_guestMemorySize;
  ctx.msr = 0x8002u;
  X360LOGI(TAG, "thread 0 -> guest PC 0x%08llX, r1(stack)=0x%08llX",
           (unsigned long long)entryPc, (unsigned long long)stackTop);
  ts->host = std::thread([this, slot] { threadLoop(slot); });
  return true;
}

bool JitEngine::runGuestFunction(uint64_t pc, uint64_t r3, uint64_t stackTop, int* slotOut) {
  const int slot = acquireSlot();
  if (slot < 0) return false;
  ThreadSlot* ts = m_slots[slot].get();
  PPCContext& ctx = ts->ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.threadIndex = slot;
  ctx.pc = pc;
  ctx.gpr[1] = stackTop;
  ctx.gpr[3] = r3;
  ctx.lr = kGuestReturnSentinel;
  ctx.hostMem = m_guestMemory;
  ctx.guestBase = m_guestBase;
  ctx.guestSize = m_guestMemorySize;
  if (slotOut) *slotOut = slot;
  ts->requestExit.store(false);
  ts->host = std::thread([this, slot] { threadLoop(slot); });
  return true;
}

void JitEngine::requestStop(int slot) {
  if (slot < 0 || size_t(slot) >= m_slots.size()) return;
  m_slots[slot]->requestExit.store(true);
}

bool JitEngine::start(int threadCount) {
  if (m_state.load() != JitState::Stopped) return false;
  // The Xenon has 6 SMT threads; extra "threads" in this runtime are guest
  // threads scheduled on the slots above, started by ExCreateThread.
  const int n = std::max(1, std::min(6, threadCount));
  m_state.store(JitState::Running);
  X360LOGI(TAG, "starting (%d Xenon threads requested; slots are driven by ExCreateThread)", n);
  return true;
}

void JitEngine::pause() { m_pauseRequested.store(true); }
void JitEngine::resume() { m_pauseRequested.store(false); }

void JitEngine::stop() {
  if (m_state.load() == JitState::Stopped) return;
  m_state.store(JitState::Stopped);
  for (auto& ts : m_slots) {
    ts->requestExit.store(true);
    if (ts->host.joinable()) {
      // A guest thread blocked in an HLE wait must be woken; give it a chance,
      // then detach rather than hanging the app on exit.
      for (int i = 0; i < 50 && ts->host.joinable(); i++) {
        if (ts->host.joinable() && i == 49) {
          X360LOGW(TAG, "thread slot still busy after 500 ms — detaching");
          ts->host.detach();
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
      if (ts->host.joinable()) ts->host.join();
    }
    ts->busy.store(false);
  }
  m_activeThreads.store(0);
  X360LOGI(TAG, "stopped: %llu blocks, %llu guest instructions, arena %zu/%zu MB",
           (unsigned long long)m_blocksCompiled.load(),
           (unsigned long long)m_instrExecuted.load(), arenaBytesUsed() >> 20,
           arenaBytesTotal() >> 20);
}

std::string JitEngine::threadStatus() const {
  std::string s;
  char buf[192];
  for (size_t i = 0; i < m_slots.size(); i++) {
    const ThreadSlot* ts = m_slots[i].get();
    if (!ts->busy.load()) continue;
    snprintf(buf, sizeof(buf), "t%zu pc=0x%08llX r1=0x%llX %s%s | ", i,
             (unsigned long long)ts->ctx.pc, (unsigned long long)ts->ctx.gpr[1],
             ts->requestExit.load() ? "exiting" : "running", "");
    s += buf;
  }
  if (s.empty()) s = "(no active guest threads)";
  return s;
}

// ─── Dispatch loop ────────────────────────────────────────────────────────────
void JitEngine::threadLoop(int slot) {
  ThreadSlot* ts = m_slots[slot].get();
  PPCContext& ctx = ts->ctx;
  ctx.exitCode = kExitContinue;

  auto lastFpsTime = std::chrono::steady_clock::now();
  uint64_t lastSwaps = m_swapTick.load();
  uint64_t interpretedRun = 0;

  while (m_state.load() == JitState::Running && !ts->requestExit.load()) {
    if (m_pauseRequested.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    const uint64_t pc = ctx.pc & 0xFFFFFFFFull;
    if (pc == 0) {
      // Nothing to run yet (idle Xenon thread). The kernel owns wakeups.
      std::this_thread::sleep_for(std::chrono::microseconds(200));
      continue;
    }
    if (pc == kGuestReturnSentinel) break;  // thread finished its guest function

    // HLE entry points.
    uint32_t importIndex = 0;
    if (isHleSentinel(pc, &importIndex)) {
      dispatchHle(ctx, importIndex);
      continue;
    }
    if (pc >= kHleThunkBase && pc < kHleThunkBase + kHleThunkMax * 8 && ((pc - kHleThunkBase) % 8) == 0) {
      dispatchHle(ctx, uint32_t((pc - kHleThunkBase) / 8));
      continue;
    }
    if (pc < m_guestBase || pc - m_guestBase >= m_guestMemorySize) {
      char buf[160];
      snprintf(buf, sizeof(buf), "guest PC 0x%08llX outside the mapped window",
               (unsigned long long)pc);
      diag::report_unsupported(diag::Subsystem::Memory, buf);
      X360LOGE(TAG, "%s (t%d, lr=0x%llX)", buf, slot, (unsigned long long)ctx.lr);
      ctx.exitCode = kExitError;
      ctx.haltReason = 8;
      diag::write_crash_report("bad guest PC", 0, pc);
      break;
    }

    CompiledBlock* block = nullptr;
    {
      std::lock_guard<std::mutex> lk(m_cacheMutex);
      block = findBlock(pc);
    }
    if (block && block->usable) {
      const uint64_t gen = block->generation;
      auto fn = reinterpret_cast<BlockFn>(block->code);
      const uint32_t exit = fn(&ctx);
      ctx.exitCode = exit;
      m_instrExecuted.fetch_add(block->instrCount, std::memory_order_relaxed);
      if (exit == kExitHle) {
        dispatchHle(ctx, ctx.hleIndex);
        continue;
      }
      if (exit == kExitError) {
        diag::write_crash_report("error inside JIT'd block", 0, pc);
        break;
      }
      if (exit == kExitReturn) break;
      // If another thread invalidated this block while we were running it, drop
      // back to the dispatcher so the next fetch sees the fresh cache state.
      if (gen != block->generation) continue;
      if (ctx.pc == pc + block->guestBytes) continue;  // fast path: straight on
      continue;
    }

    // Interpreter path: run until a branch target leaves this block, bounded so a
    // long straight-line region cannot starve the dispatcher.
    interpretedRun++;
    const uint32_t exit = interpretRange(ctx, 256, 0);
    m_interpBlocks.fetch_add(1, std::memory_order_relaxed);
    if (exit == kExitHle) {
      dispatchHle(ctx, ctx.hleIndex);
      continue;
    }
    if (exit == kExitError) {
      diag::write_crash_report("error in interpreted guest code", 0, ctx.pc);
      break;
    }
    if (exit == kExitReturn) break;
    if (interpretedRun % 4096 == 0 && block == nullptr && m_jitEnabled) {
      // Retry compilation occasionally: a block may have been rejected because
      // the arena was full, and the code may since have been patched.
      std::lock_guard<std::mutex> lk(m_cacheMutex);
      m_blockCache.erase(pc);
    }

    // FPS from GPU swaps (the title's frame cadence, not our loop count).
    const uint64_t swaps = m_swapTick.load();
    if (swaps != lastSwaps) {
      const auto now = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(now - lastFpsTime).count();
      const uint64_t frames = swaps - lastSwaps;
      if (ms > 500 && frames) {
        m_fps.store(double(frames) * 1000.0 / ms);
        m_frameTimeMs.store(ms / double(frames));
        lastFpsTime = now;
        lastSwaps = swaps;
      }
    }
  }
  X360LOGI(TAG, "slot %d left the dispatch loop (state=%d exit=%u pc=0x%08llX)", slot,
           int(m_state.load()), ctx.exitCode, (unsigned long long)ctx.pc);
  releaseSlot(slot);
}

}  // namespace jit
}  // namespace x360
