cmake_minimum_required(VERSION 3.14)

include(ExternalProject)
include(FetchContent)

# include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/print_target.cmake")


# Setup directories and basic paths
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(DAWN_DIR           "${FETCHCONTENT_BASE_DIR}/dawn" CACHE INTERNAL "Dawn source directory")

# For Emscripten builds (if desired)
set(EM_SDK_DIR         $ENV{EMSDK} CACHE INTERNAL "")
set(EMSCRIPTEN_DIR     "${EM_SDK_DIR}/upstream/emscripten" CACHE INTERNAL "")

# Decide where to build Dawn’s build files.
if(EMSCRIPTEN)
  set(DAWN_BUILD_DIR "${DAWN_DIR}/build_web" CACHE INTERNAL "web build directory" FORCE)
elseif(WIN32)
  set(DAWN_BUILD_DIR "${DAWN_DIR}/build_win" CACHE INTERNAL "windows build directory" FORCE)
elseif(IOS)
  set(DAWN_BUILD_DIR "${DAWN_DIR}/build_ios" CACHE INTERNAL "ios build directory" FORCE)
elseif(APPLE)
  set(DAWN_BUILD_DIR "${DAWN_DIR}/build_mac" CACHE INTERNAL "mac build directory" FORCE)
elseif(ANDROID)
  set(DAWN_BUILD_DIR "${DAWN_DIR}/build_android" CACHE INTERNAL "android build directory" FORCE)
else()
  set(DAWN_BUILD_DIR "${DAWN_DIR}/build_unix" CACHE INTERNAL "linux build directory" FORCE)
endif()

# Add Dawn header include directories so that they are available later.
include_directories(BEFORE PUBLIC 
  "${DAWN_BUILD_DIR}/src/dawn/native/"
  "${DAWN_BUILD_DIR}/src/dawn/native/Debug"
  "${DAWN_BUILD_DIR}/src/dawn/native/Release"
)


# Optionally try to find an existing Dawn build.
option(ENABLE_DAWN_FIND "Attempt to find an existing Dawn build" OFF)
set(DAWN_BUILD_FOUND OFF CACHE BOOL "Dawn build found" FORCE)

if(ENABLE_DAWN_FIND)
  if(WIN32)
    find_library(WEBGPU_DAWN_DEBUG NAMES webgpu_dawn HINTS "${DAWN_BUILD_DIR}/src/dawn/native/Debug")
    find_library(WEBGPU_DAWN_RELEASE NAMES webgpu_dawn HINTS "${DAWN_BUILD_DIR}/src/dawn/native/Release")
    if(WEBGPU_DAWN_DEBUG OR WEBGPU_DAWN_RELEASE)
      set(DAWN_BUILD_FOUND ON)
    endif()
  elseif(NOT EMSCRIPTEN AND NOT WIN32)
    find_library(WEBGPU_DAWN_LIB NAMES webgpu_dawn.so PATHS "${DAWN_BUILD_DIR}/src/dawn/native")
    if(WEBGPU_DAWN_LIB)
      set(DAWN_BUILD_FOUND ON)
    endif()
  else()
    set(DAWN_BUILD_FOUND ON)
  endif()
endif()


# Pre-build Dawn at configuration time if not already built.
if(NOT DAWN_BUILD_FOUND)
  message(STATUS "Dawn build not found – pre-building Dawn using execute_process.")

  # Force Dawn build options.
  set(DAWN_ALWAYS_ASSERT           OFF CACHE INTERNAL "Always assert in Dawn" FORCE)
  set(DAWN_BUILD_MONOLITHIC_LIBRARY ON CACHE INTERNAL "Build Dawn monolithically" FORCE)
  set(DAWN_BUILD_EXAMPLES          OFF CACHE INTERNAL "Build Dawn examples" FORCE)
  set(DAWN_BUILD_SAMPLES           OFF CACHE INTERNAL "Build Dawn samples" FORCE)
  set(DAWN_BUILD_TESTS             OFF CACHE INTERNAL "Build Dawn tests" FORCE)
  set(DAWN_ENABLE_INSTALL          OFF  CACHE INTERNAL "Enable Dawn installation" FORCE)
  set(DAWN_FETCH_DEPENDENCIES      ON  CACHE INTERNAL "Fetch Dawn dependencies" FORCE)
  set(TINT_BUILD_TESTS             OFF CACHE INTERNAL "Build Tint Tests" FORCE)
  set(TINT_BUILD_IR_BINARY         OFF CACHE INTERNAL "Build Tint IR binary" FORCE)
  set(TINT_BUILD_CMD_TOOLS         OFF CACHE INTERNAL "Build Tint command line tools" FORCE)
  set(DAWN_EMSCRIPTEN_TOOLCHAIN    ${EMSCRIPTEN_DIR} CACHE INTERNAL "Emscripten toolchain" FORCE)

  # Fetch the Dawn repository if not already present.
  FetchContent_Declare(
    dawn
    DOWNLOAD_DIR ${DAWN_DIR}
    SOURCE_DIR   ${DAWN_DIR}
    SUBBUILD_DIR ${DAWN_BUILD_DIR}/tmp
    BINARY_DIR   ${DAWN_BUILD_DIR}
    DOWNLOAD_COMMAND
      cd ${DAWN_DIR} &&
      git init &&
      git fetch --depth=1 https://dawn.googlesource.com/dawn &&
      git reset --hard FETCH_HEAD
  )
  FetchContent_MakeAvailable(dawn)

  set(CMAKE_INCLUDE_PATH "${CMAKE_INCLUDE_PATH};${DAWN_DIR}/src" CACHE INTERNAL "")

  set(DAWN_CONFIG_GENERATOR ${CMAKE_GENERATOR} CACHE INTERNAL "Dawn CMake generator" FORCE)

  # Build unified configuration arguments based on target platform
  # Default generator is Ninja.
  set(DAWN_CONFIG_ARGS
      -DDAWN_ALWAYS_ASSERT=${DAWN_ALWAYS_ASSERT}
      -DDAWN_BUILD_MONOLITHIC_LIBRARY=${DAWN_BUILD_MONOLITHIC_LIBRARY}
      -DDAWN_BUILD_EXAMPLES=${DAWN_BUILD_EXAMPLES}
      -DDAWN_BUILD_SAMPLES=${DAWN_BUILD_SAMPLES}
      -DDAWN_BUILD_TESTS=${DAWN_BUILD_TESTS}
      -DDAWN_ENABLE_INSTALL=${DAWN_ENABLE_INSTALL}
      -DDAWN_FETCH_DEPENDENCIES=${DAWN_FETCH_DEPENDENCIES}
      -DTINT_BUILD_TESTS=${TINT_BUILD_TESTS}
      -DTINT_BUILD_IR_BINARY=${TINT_BUILD_IR_BINARY}
      -DTINT_BUILD_CMD_TOOLS=${TINT_BUILD_CMD_TOOLS}
  )
  if(EMSCRIPTEN)
    list(APPEND DAWN_CONFIG_ARGS -DDAWN_EMSCRIPTEN_TOOLCHAIN=${EMSCRIPTEN_DIR})
  endif()

  if(ANDROID)
    list(APPEND DAWN_CONFIG_ARGS
      -DCMAKE_SYSTEM_NAME=Android
      -DCMAKE_ANDROID_ARCH_ABI=${CMAKE_ANDROID_ARCH_ABI}
      -DCMAKE_ANDROID_NDK=${ANDROID_NDK}
      -DCMAKE_THREAD_PREFER_PTHREAD=True
      -DTHREADS_PREFER_PTHREAD_FLAG=True
      -DCMAKE_CXX_FLAGS="-llog"
    )
  elseif(IOS)
    # For iOS, you need to specify a toolchain file.
    if(NOT CMAKE_TOOLCHAIN_FILE)
      message(FATAL_ERROR "iOS build requires CMAKE_TOOLCHAIN_FILE to be set (e.g., /path/to/ios/toolchain.cmake)")
    endif()
    list(APPEND DAWN_CONFIG_ARGS
      -DCMAKE_SYSTEM_NAME=iOS
      -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
      -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
    )
  endif()


  # Run the configuration step for Dawn.
  execute_process(
    WORKING_DIRECTORY ${DAWN_DIR}
    COMMAND ${CMAKE_COMMAND} -S ${DAWN_DIR} -B ${DAWN_BUILD_DIR} -G ${DAWN_CONFIG_GENERATOR} ${DAWN_CONFIG_ARGS}
    RESULT_VARIABLE config_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT config_result EQUAL 0)
    message(FATAL_ERROR "Failed to configure Dawn")
  endif()

  # Build and install Dawn so that the resulting library file is available.
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${DAWN_BUILD_DIR}
    RESULT_VARIABLE build_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build Dawn")
  endif()

  set(DAWN_BUILD_FOUND ON)
endif()  # End pre-build Dawn

# Create an IMPORTED target for the Dawn library.
# Adjust the expected output name/extension per platform.
if(WIN32)
  if(MSVC)
    if(NOT TARGET webgpu_dawn_debug)
      add_library(webgpu_dawn_debug SHARED IMPORTED)
      set_target_properties(webgpu_dawn_debug PROPERTIES
        IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/Debug/webgpu_dawn.dll")
    endif()
    if(NOT TARGET webgpu_dawn_release)
      add_library(webgpu_dawn_release SHARED IMPORTED)
      set_target_properties(webgpu_dawn_release PROPERTIES
        IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/Release/webgpu_dawn.dll")
    endif()
    if(NOT TARGET webgpu_dawn)
      add_library(webgpu_dawn INTERFACE)
      target_link_libraries(webgpu_dawn INTERFACE
        $<$<CONFIG:Debug>:webgpu_dawn_debug>
        $<$<CONFIG:Release>:webgpu_dawn_release>
      )
    endif()
  else()
    if(NOT TARGET webgpu_dawn)
      add_library(webgpu_dawn SHARED IMPORTED)
      set_target_properties(webgpu_dawn PROPERTIES
        IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/webgpu_dawn.dll")
    endif()
  endif()
elseif(IOS)
  # On iOS, it is common to build a static library.
  if(NOT TARGET webgpu_dawn)
    add_library(webgpu_dawn STATIC IMPORTED)
    set_target_properties(webgpu_dawn PROPERTIES
      IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/webgpu_dawn.a")
  endif()
elseif(APPLE)
  # On macOS (non-iOS), typically a dynamic library (.dylib) is built.
  if(NOT TARGET webgpu_dawn)
    add_library(webgpu_dawn SHARED IMPORTED)
    set_target_properties(webgpu_dawn PROPERTIES
      IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/webgpu_dawn.dylib")
  endif()
elseif(ANDROID)
  if(NOT TARGET webgpu_dawn)
    add_library(webgpu_dawn SHARED IMPORTED)
    set_target_properties(webgpu_dawn PROPERTIES
      IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/webgpu_dawn.so")
  endif()
else()  # For Linux and other Unix-like systems.
  if(NOT TARGET webgpu_dawn)
    add_library(webgpu_dawn SHARED IMPORTED)
    set_target_properties(webgpu_dawn PROPERTIES
      IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/webgpu_dawn.so")
  endif()
endif()

# Make sure the imported target sees Dawn’s include directories.
target_include_directories(webgpu_dawn INTERFACE
  "${DAWN_BUILD_DIR}/src/dawn/native/"
  "${DAWN_BUILD_DIR}/src/dawn/native/Debug"
  "${DAWN_BUILD_DIR}/src/dawn/native/Release"
)