cmake_minimum_required(VERSION 3.5)
project(minigpu_ffi VERSION 1.0.0 LANGUAGES C)

# Configure the find_root.cmake script
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_root.cmake")

# Call the function to find the project root
find_project_root("${CMAKE_CURRENT_SOURCE_DIR}" "${FILENAME}" TARGET_FILE_PATH)

# Check if the target file path was found
if(TARGET_FILE_PATH)
    message(STATUS "Project root found at ${TARGET_FILE_PATH}")
else()
    message(FATAL_ERROR "File ${FILENAME} not found within 8 levels up from ${CMAKE_CURRENT_SOURCE_DIR}")
endif()


include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/dawn.cmake")
#include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu.cmake")
