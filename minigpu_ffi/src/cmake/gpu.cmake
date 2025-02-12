set(EXT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external" CACHE STRING "Path to the local source directory")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/download.cmake")



download_repository("dawn" "https://github.com/AnswerDotAI/gpu.cpp" "${EXT_SOURCE_DIR}/gpu")

set(GPU_INCLUDE_DIR "${EXT_SOURCE_DIR}/gpu")

# Add sources
set(GPU_SOURCES
    "${GPU_INCLUDE_DIR}/gpu.cpp"
    "${GPU_INCLUDE_DIR}/numeric_types/half.cpp"
)

# Add headers
set(GPU_HEADERS
    "${GPU_INCLUDE_DIR}/gpu.hpp"
    "${GPU_INCLUDE_DIR}/utils/logging.hpp"
    "${GPU_INCLUDE_DIR}/utils/array_utils.hpp"
    "${GPU_INCLUDE_DIR}/numeric_types/half.hpp"
)

# Emscripten includes a header automatically
if(EMSCRIPTEN)
    file(REMOVE "${GPU_INCLUDE_DIR}/webgpu/webgpu.h")
else()
    list(APPEND GPU_HEADERS "${GPU_INCLUDE_DIR}/third_party/headers/webgpu/webgpu.h")
endif()

add_library(gpu STATIC ${GPU_SOURCES} ${GPU_HEADERS})
target_include_directories(gpu PUBLIC "${GPU_INCLUDE_DIR}")
target_include_directories(gpu PUBLIC "${GPU_INCLUDE_DIR}/third_party/headers")
