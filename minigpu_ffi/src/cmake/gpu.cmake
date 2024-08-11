set(GPU_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/include/gpu")

set(GPU_H_URL "https://raw.githubusercontent.com/AnswerDotAI/gpu.cpp/main/gpu.h")
set(GPU_CPP_URL "https://raw.githubusercontent.com/AnswerDotAI/gpu.cpp/main/gpu.cpp")
set(LOGGING_H_URL "https://raw.githubusercontent.com/AnswerDotAI/gpu.cpp/main/utils/logging.h")
set(ARRAY_UTILS_H_URL "https://raw.githubusercontent.com/AnswerDotAI/gpu.cpp/main/utils/array_utils.h")
set(HALF_CPP_URL "https://raw.githubusercontent.com/AnswerDotAI/gpu.cpp/main/numeric_types/half.cpp")
set(HALF_H_URL "https://raw.githubusercontent.com/AnswerDotAI/gpu.cpp/main/numeric_types/half.h")

set(GPU_H_PATH "${GPU_INCLUDE_DIR}/gpu.h")
set(GPU_CPP_PATH "${GPU_INCLUDE_DIR}/gpu.cpp")
set(LOGGING_H_PATH "${GPU_INCLUDE_DIR}/utils/logging.h")
set(ARRAY_UTILS_H_PATH "${GPU_INCLUDE_DIR}/utils/array_utils.h")
set(HALF_CPP_PATH "${GPU_INCLUDE_DIR}/numeric_types/half.cpp")
set(HALF_H_PATH "${GPU_INCLUDE_DIR}/numeric_types/half.h")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/download.cmake")
download_files(GPU_H LOGGING_H HALF_CPP HALF_H ARRAY_UTILS_H GPU_CPP)

add_library(gpu INTERFACE)
target_include_directories(gpu SYSTEM INTERFACE "${GPU_INCLUDE_DIR}")