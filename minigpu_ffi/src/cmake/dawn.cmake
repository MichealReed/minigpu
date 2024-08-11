# Set the DAWN_EXT_SOURCE_DIR to the local Dawn directory
set(DAWN_EXT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/dawn" CACHE STRING "Path to the local Dawn source directory")

# Ensure submodules are initialized and updated
execute_process(
    COMMAND git submodule update --init --recursive
    WORKING_DIRECTORY ${DAWN_EXT_SOURCE_DIR}
    RESULT_VARIABLE SUBMODULE_UPDATE_RESULT
)

# Set necessary variables
set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/external/" CACHE INTERNAL "")

# Custom target to fetch Dawn dependencies
add_custom_target(fetch_dawn_dependencies
    COMMAND python3 ${DAWN_EXT_SOURCE_DIR}/tools/fetch_dawn_dependencies.py
    WORKING_DIRECTORY ${DAWN_EXT_SOURCE_DIR}
)

# Set desired CMake arguments
set(DESIRED_CMAKE_ARGS
    -DDAWN_DISABLE_LOGGING=ON
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

# Use ExternalProject to configure and build Dawn
include(ExternalProject)
ExternalProject_Add(
    dawn_project
    PREFIX dawn
    SOURCE_DIR ${DAWN_EXT_SOURCE_DIR}
    CMAKE_ARGS ${DESIRED_CMAKE_ARGS}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
    INSTALL_DIR ${INSTALL_DIR}
    STEP_TARGETS install
    DEPENDS fetch_dawn_dependencies
)

add_custom_target(install_dawn
    DEPENDS dawn_project-install
)