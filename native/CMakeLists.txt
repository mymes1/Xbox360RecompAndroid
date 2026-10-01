cmake_minimum_required(VERSION 3.22)
project(xbox360recomp CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Compiler flags ────────────────────────────────────────────────────────────
add_compile_options(
    -Wall -Wextra -Wpedantic
    -fno-strict-aliasing    # Required for JIT memory aliasing
    -ffast-math
    -funroll-loops
)

# Release: aggressive optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -flto=thin -DNDEBUG)
endif()

# ─── Source files ──────────────────────────────────────────────────────────────
set(SOURCES_LOADER
    src/loader/xex_loader.cpp
    src/loader/stfs_parser.cpp
    src/loader/iso_parser.cpp
    src/loader/aes128.cpp
    src/loader/lzx_decompress.cpp
)

set(SOURCES_JIT
    src/jit/ppc_decode.cpp        # table-driven PowerPC/Xenon decoder (+ ppc_tables.inc)
    src/jit/jit_interpreter.cpp   # reference semantics for every decoded class
    src/jit/arm64_backend.cpp     # ARM64 codegen for the verified instruction subset
    src/jit/jit_engine.cpp        # threads, block cache, HLE dispatch
    src/jit/memory_guard.cpp      # W^X code buffers + SMC invalidation
)

# The IR pipeline (ppc_decoder/ir_builder/ir_optimizer/block_cache) was replaced by the
# decode -> interpreter -> ARM64-backend path above; those files had no callers.

# Guest floating point must follow PowerPC semantics exactly. -ffast-math (set globally
# below) would let the compiler flush denormals, assume there are no NaN/Inf, and fold
# mul+add into an FMA - each of which silently changes game state. Same care for the
# byte-level image parsers. The global -fno-strict-aliasing stays: code buffers are
# written as data and executed as code.
set_source_files_properties(${SOURCES_JIT} src/loader/xex_loader.cpp src/loader/aes128.cpp
    src/loader/iso_parser.cpp src/loader/lzx_decompress.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-fno-unsafe-math-optimizations;-ffp-contract=off")

set(SOURCES_HLE
    src/hle/memory.cpp
    src/hle/filesystem.cpp
    src/hle/threading.cpp
    src/hle/input.cpp
    src/hle/audio.cpp
    src/hle/xboxkrnl.cpp
    src/hle/xam.cpp
)

set(SOURCES_GPU
    src/gpu/pm4_parser.cpp
    src/gpu/shader_recompiler.cpp
    src/gpu/texture_manager.cpp
    src/gpu/vulkan_renderer.cpp
    src/gpu/edram_simulator.cpp
    src/gpu/driver_loader.cpp
)

set(SOURCES_BRIDGE
    src/bridge/jni_bridge.cpp
    src/bridge/engine.cpp
    src/bridge/surface_bridge.cpp
    src/bridge/metrics.cpp
)

# ─── Main shared library ───────────────────────────────────────────────────────
add_library(xbox360recomp SHARED
    ${SOURCES_LOADER}
    ${SOURCES_JIT}
    ${SOURCES_HLE}
    ${SOURCES_GPU}
    ${SOURCES_BRIDGE}
)

target_include_directories(xbox360recomp PRIVATE
    include/
    ${ANDROID_NDK}/sysroot/usr/include
)

# ─── Android system libs ───────────────────────────────────────────────────────
find_library(log-lib log)
find_library(android-lib android)
find_library(vulkan-lib vulkan)
find_library(aaudio-lib aaudio)
find_library(opensl-lib OpenSLES)
find_library(nativewindow-lib nativewindow)

target_link_libraries(xbox360recomp PRIVATE
    ${log-lib}
    ${android-lib}
    ${vulkan-lib}
    ${aaudio-lib}
    ${opensl-lib}
    ${nativewindow-lib}
    atomic
    dl
)

# ─── AdrenoTools (optional) ────────────────────────────────────────────────────
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/adrenotools/CMakeLists.txt")
    add_subdirectory(third_party/adrenotools)
    target_link_libraries(xbox360recomp PRIVATE adrenotools)
    target_compile_definitions(xbox360recomp PRIVATE HAVE_ADRENOTOOLS=1)
endif()

# ─── SPIRV-Headers (header-only) ──────────────────────────────────────────────
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/SPIRV-Headers/include")
    target_include_directories(xbox360recomp PRIVATE third_party/SPIRV-Headers/include)
    target_compile_definitions(xbox360recomp PRIVATE HAVE_SPIRV_HEADERS=1)
endif()

# ─── Host-side encoder probe (optional; NOT part of the APK) ─────────────────────
# `cmake -DX360_HOST_PROBES=ON ..` builds a small program that prints every ARM64
# word the backend's encoder emits, so tools/check_arm64_asm.py can disassemble them
# with Capstone on a development machine. Disabled by default and skipped on Android
# so it cannot affect the shipped library.
option(X360_HOST_PROBES "Build the host ARM64 encoding probe" OFF)
if(X360_HOST_PROBES AND NOT ANDROID)
    add_executable(arm64_asm_probe ${CMAKE_CURRENT_SOURCE_DIR}/../tests/arm64_asm_probe.cpp)
    target_include_directories(arm64_asm_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(arm64_asm_probe PRIVATE -Wall -Wextra -fno-fast-math)
endif()
