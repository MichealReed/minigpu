set(CMAKE_BUILD_TYPE  Release CACHE STRING "Choose the type of build: Debug or Release" FORCE)

# find_library, windows adds extra folder
if(MSVC)
    find_library(WEBGPU_DAWN_MONOLITHIC
    NAMES webgpu_dawn
    HINTS "${DAWN_BUILD_DIR}/src/dawn/native/${CMAKE_BUILD_TYPE}"
    )
else()
    find_library(WEBGPU_DAWN_MONOLITHIC
    NAMES webgpu_dawn
    PATHS "${DAWN_BUILD_DIR}/src/dawn/native"
    )
endif()

# There's a issue where only the first flutter build can find headers
# so the first build succeeds, but the second+ fails unless you flutter clean
# including the directories again doesnt seem to work, maybe a problem
# with the flutter tooling for MSVC CMake

if(NOT WEBGPU_DAWN_MONOLITHIC)
    message("webgpu_dawn not found start building")
    set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/dawn")

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

    FetchContent_Declare(
        dawn
        DOWNLOAD_COMMAND
        cd ${FETCHCONTENT_BASE_DIR}/dawn-src &&
        git init &&
        git fetch --depth=1 https://dawn.googlesource.com/dawn &&
        git reset --hard FETCH_HEAD
    )

    # Download the repository and add it as a subdirectory.
    FetchContent_MakeAvailable(dawn)

    set(DAWN_SRC_DIR "${FETCHCONTENT_BASE_DIR}/dawn-src" CACHE INTERNAL "")
    set(DAWN_BUILD_DIR "${FETCHCONTENT_BASE_DIR}/dawn-build" CACHE INTERNAL "")
    set(CMAKE_INCLUDE_PATH "${CMAKE_INCLUDE_PATH};${DAWN_SRC_DIR}/src" CACHE INTERNAL "")

    include_directories("${DAWN_SRC_DIR}/src")

    execute_process(
        COMMAND ${CMAKE_COMMAND} -S ${DAWN_SRC_DIR}
            -B ${DAWN_BUILD_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -G "${CMAKE_GENERATOR}"
    )
    # Build Dawn
    execute_process(
        WORKING_DIRECTORY ${DAWN_SRC_DIR}/src
        COMMAND ${CMAKE_COMMAND} --build ${DAWN_BUILD_DIR} --config ${CMAKE_BUILD_TYPE}
    )
endif()