set(GPU_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/include/gpu")

set(GPU_H_URL "https://raw.githubusercontent.com/AnswerDotAI/gpu.cpp/main/gpu.h")
set(GPU_CPP_URL "https://raw.githubusercontent.com/AnswerDotAI/gpu.cpp/main/gpu.cpp")
set(LOGGING_H_URL "https://raw.githubusercontent.com/AnswerDotAI/gpu.cpp/main/utils/logging.h")
set(ARRAY_UTILS_H_URL "https://raw.githubusercontent.com/AnswerDotAI/gpu.cpp/main/utils/array_utils.h")
set(HALF_CPP_URL "https://raw.githubusercontent.com/AnswerDotAI/gpu.cpp/main/numeric_types/half.cpp")
set(HALF_H_URL "https://raw.githubusercontent.com/AnswerDotAI/gpu.cpp/main/numeric_types/half.h")
set(WEBGPU_H_URL "https://raw.githubusercontent.com/AnswerDotAI/gpu.cpp/main/third_party/headers/webgpu/webgpu.h")

set(GPU_H_PATH "${GPU_INCLUDE_DIR}/gpu.h")
set(GPU_CPP_PATH "${GPU_INCLUDE_DIR}/gpu.cpp")
set(LOGGING_H_PATH "${GPU_INCLUDE_DIR}/utils/logging.h")
set(ARRAY_UTILS_H_PATH "${GPU_INCLUDE_DIR}/utils/array_utils.h")
set(HALF_CPP_PATH "${GPU_INCLUDE_DIR}/numeric_types/half.cpp")
set(HALF_H_PATH "${GPU_INCLUDE_DIR}/numeric_types/half.h")
set(WEBGPU_H_PATH "${GPU_INCLUDE_DIR}/webgpu/webgpu.h")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/download.cmake")
download_files(GPU_H LOGGING_H HALF_CPP HALF_H ARRAY_UTILS_H GPU_CPP WEBGPU_H)

set(GPU_SOURCES
    "${GPU_INCLUDE_DIR}/gpu.cpp"
    "${GPU_INCLUDE_DIR}/numeric_types/half.cpp"
)

set(GPU_HEADERS
    "${GPU_INCLUDE_DIR}/gpu.h"
    "${GPU_INCLUDE_DIR}/utils/logging.h"
    "${GPU_INCLUDE_DIR}/utils/array_utils.h"
    "${GPU_INCLUDE_DIR}/numeric_types/half.h"
)

if(EMSCRIPTEN)
    file(REMOVE "${GPU_INCLUDE_DIR}/webgpu/webgpu.h")
else()
    list(APPEND GPU_HEADERS "${GPU_INCLUDE_DIR}/webgpu/webgpu.h")
endif()

add_library(gpu STATIC ${GPU_SOURCES} ${GPU_HEADERS})
target_include_directories(gpu PUBLIC "${GPU_INCLUDE_DIR}")
