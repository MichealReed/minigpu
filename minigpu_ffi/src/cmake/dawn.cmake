set(CMAKE_BUILD_TYPE  Release CACHE STRING "Choose the type of build: Debug or Release" FORCE)
# Setup directories
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(EM_SDK_DIR $ENV{EMSDK} CACHE INTERNAL "")
set(EMSCRIPTEN_DIR "${FETCHCONTENT_BASE_DIR}/emscripten" CACHE INTERNAL "")
set(DAWN_DIR "${FETCHCONTENT_BASE_DIR}/dawn" CACHE INTERNAL "")
set(DAWN_BUILD_DIR "${DAWN_DIR}/build" CACHE INTERNAL "")

if(EMSCRIPTEN)
    set(DAWN_BUILD_DIR "${DAWN_DIR}/build_web" CACHE INTERNAL "")
endif()

# There's an issue where flutter requires a clean or
# the ephemeral build cannot find the headers
# this speeds up work on shaders because waiting
# for CMake build is slow anyways
set(ENABLE_DAWN_FIND OFF CACHE BOOL "Enable finding Dawn" FORCE)
set(DAWN_BUILD_FOUND OFF CACHE BOOL "Dawn build found" FORCE)
if(ENABLE_DAWN_FIND)
    # find_library, windows adds extra folder
    if(MSVC)
        find_library(WEBGPU_DAWN_MONOLITHIC
        NAMES webgpu_dawn
        HINTS "${DAWN_BUILD_DIR}/src/dawn/native/${CMAKE_BUILD_TYPE}"
        )
        set(DAWN_BUILD_FOUND ON)
    else()
        find_library(WEBGPU_DAWN_MONOLITHIC
        NAMES webgpu_dawn
        PATHS "${DAWN_BUILD_DIR}/src/dawn/native"
        )
        set(DAWN_BUILD_FOUND ON)
    endif()
endif()

# Dawn options for more,
# see https://dawn.googlesource.com/dawn/+/refs/heads/main/CMakeLists.txt
set(DAWN_ALWAYS_ASSERT     OFF CACHE INTERNAL "Always assert in Dawn" FORCE)
set(DAWN_BUILD_MONOLITHIC_LIBRARY ON CACHE INTERNAL "Build Dawn monolithically" FORCE)
set(DAWN_BUILD_EXAMPLES      OFF CACHE INTERNAL "Build Dawn examples" FORCE)
set(DAWN_BUILD_SAMPLES      OFF CACHE INTERNAL "Build Dawn samples" FORCE)
set(DAWN_BUILD_TESTS         OFF CACHE INTERNAL "Build Dawn tests" FORCE)
set(DAWN_ENABLE_INSTALL      OFF  CACHE INTERNAL "Enable Dawn installation" FORCE)
set(DAWN_FETCH_DEPENDENCIES ON  CACHE INTERNAL "Fetch Dawn dependencies" FORCE)
set(TINT_BUILD_TESTS        OFF CACHE INTERNAL "Build Tint Tests" FORCE)
set(TINT_BUILD_IR_BINARY    OFF CACHE INTERNAL "Build Tint IR binary" FORCE)
set(TINT_BUILD_CMD_TOOLS   OFF CACHE INTERNAL "Build Tint command line tools" FORCE)
set(BUILD_SHARED_LIBS       OFF CACHE INTERNAL "Build shared libraries" FORCE)

# There's a issue where only the first flutter build can find headers
# so the first build succeeds, but the second+ fails unless you flutter clean
# including the directories again doesnt seem to work, maybe a problem
# with the flutter tooling for MSVC CMake

if(NOT DAWN_BUILD_FOUND)
    include(FetchContent)
    message("webgpu_dawn not found start building")
    if(EMSCRIPTEN)
        set(EMSCRIPTEN_DIR "${EM_SDK_DIR}/upstream/emscripten" CACHE INTERNAL "" FORCE)
        set(DAWN_EMSCRIPTEN_TOOLCHAIN ${EMSCRIPTEN_DIR} CACHE INTERNAL "" FORCE)
    endif()

    FetchContent_Declare(
        dawn
        DOWNLOAD_DIR ${DAWN_DIR}
        SOURCE_DIR ${DAWN_DIR}
        SUBBUILD_DIR ${DAWN_BUILD_DIR}/tmp
        BINARY_DIR ${DAWN_BUILD_DIR}
        DOWNLOAD_COMMAND
        cd ${DAWN_DIR} &&
        git init &&
        git fetch --depth=1 https://dawn.googlesource.com/dawn &&
        git reset --hard FETCH_HEAD
    )

    # Download the repository and add it as a subdirectory.
    FetchContent_MakeAvailable(dawn)

    # attempt fix flutter rebuilds
    set(CMAKE_INCLUDE_PATH "${CMAKE_INCLUDE_PATH};${DAWN_DIR}/src" CACHE INTERNAL "")
    include_directories("${DAWN_DIR}/src")

    execute_process(
        WORKING_DIRECTORY ${DAWN_DIR}
        COMMAND ${CMAKE_COMMAND} -S ${DAWN_DIR}
            -B ${DAWN_BUILD_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    )

    # Build Dawn
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${DAWN_BUILD_DIR} --config ${CMAKE_BUILD_TYPE}
    )

    if(EMSCRIPTEN)
        include_directories(${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/include)
        include_directories(${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu)
    endif()
    
    # find_library, windows adds extra folder
    if(MSVC)
        find_library(WEBGPU_DAWN_MONOLITHIC
        NAMES webgpu_dawn
        HINTS "${DAWN_BUILD_DIR}/src/dawn/native/${CMAKE_BUILD_TYPE}"
        REQUIRED
        )
        set(DAWN_BUILD_FOUND ON)
    elseif(NOT EMSCRIPTEN)
        find_library(WEBGPU_DAWN_MONOLITHIC
        NAMES webgpu_dawn
        PATHS "${DAWN_BUILD_DIR}/src/dawn/native"
        REQUIRED
        )
        set(DAWN_BUILD_FOUND ON)
    else()
        set(DAWN_BUILD_FOUND ON)
    endif()

endif()