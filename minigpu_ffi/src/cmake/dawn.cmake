set(EXT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external" CACHE STRING "Path to the local source directory")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/download.cmake")
download_repository("dawn" "https://github.com/google/dawn" "${EXT_SOURCE_DIR}/dawn")

execute_process(
    COMMAND python3 ${EXT_SOURCE_DIR}/dawn/tools/fetch_dawn_dependencies.py
    WORKING_DIRECTORY "${EXT_SOURCE_DIR}/dawn"
    RESULT_VARIABLE FETCH_RESULT
)

set(INSTALL_DIR "${EXT_SOURCE_DIR}" CACHE INTERNAL "")

set(DESIRED_CMAKE_ARGS
    -DDAWN_DISABLE_LOGGING=ON
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

include(ExternalProject)
ExternalProject_Add(
    dawn_project
    PREFIX dawn
    SOURCE_DIR "${EXT_SOURCE_DIR}/dawn"
    CMAKE_ARGS ${DESIRED_CMAKE_ARGS}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
    INSTALL_DIR ${INSTALL_DIR}
    STEP_TARGETS install
)

add_custom_target(install_dawn
    DEPENDS dawn_project-install
)