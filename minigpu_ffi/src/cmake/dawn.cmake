set(EXT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external" CACHE STRING "Path to the local source directory")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/download.cmake")

download_repository("dawn" "https://github.com/google/dawn" "${EXT_SOURCE_DIR}/dawn")

execute_process(
    COMMAND python3 ${EXT_SOURCE_DIR}/dawn/tools/fetch_dawn_dependencies.py
    WORKING_DIRECTORY "${EXT_SOURCE_DIR}/dawn"
    RESULT_VARIABLE FETCH_RESULT
)

set(DAWN_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/dawn" CACHE INTERNAL "")

set(DESIRED_CMAKE_ARGS
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

include(ExternalProject)
ExternalProject_Add(
    dawn_project
    PREFIX dawn
    SOURCE_DIR "${EXT_SOURCE_DIR}/dawn"
    CMAKE_ARGS ${DESIRED_CMAKE_ARGS}
    BINARY_DIR "${DAWN_BUILD_DIR}"
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
    INSTALL_COMMAND ""
)

add_library(dawn INTERFACE)
add_dependencies(dawn dawn_project)
target_include_directories(dawn INTERFACE "${EXT_SOURCE_DIR}/dawn/include")
target_link_directories(dawn INTERFACE "${DAWN_BUILD_DIR}/src/dawn")