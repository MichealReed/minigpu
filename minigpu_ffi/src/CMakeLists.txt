cmake_minimum_required(VERSION 3.5)
project(minigpu_ffi VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#enable exports
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

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

#include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/dawn.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu.cmake")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/webgpu.cmake")

if(NOT EMSCRIPTEN)
    set(MAIN_LIB minigpu_ffi)
else()
    set(MAIN_LIB minigpu_web)
    set(MAIN_JS minigpu_web_js)
endif()

set(MAIN_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(MAIN_OUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(WEB_OUTPUT_DIR ${TARGET_FILE_PATH}/minigpu_web/lib/build)

if(NOT EMSCRIPTEN)
    add_library(${MAIN_LIB} SHARED)
else()
    add_executable(${MAIN_LIB})
endif()

file(GLOB_RECURSE MAIN_SOURCES
    ${MAIN_PATH}/src/*.cpp
)


file(GLOB_RECURSE MAIN_INCLUDES LIST_DIRECTORIES TRUE
    ${MAIN_PATH}/external/include/gpu/*
)

if(NOT EMSCRIPTEN)
    list(APPEND MAIN_INCLUDES ${MAIN_PATH}/include/gpu/*)
else()
    # Exclude webgpu/webgpu.h when using Emscripten
    list(APPEND MAIN_INCLUDES
        ${MAIN_PATH}/include/gpu/gpu.hpp
        ${MAIN_PATH}/include/gpu/utils/logging.hpp
        ${MAIN_PATH}/include/gpu/utils/array_utils.h
        ${MAIN_PATH}/include/gpu/numeric_types/half.hpp
    )
endif()

target_include_directories(${MAIN_LIB} PUBLIC ${MAIN_INCLUDES})

# Definitions
target_compile_definitions(${MAIN_LIB} PUBLIC 
    DART_SHARED_LIB
)

# Set properties
if(NOT EMSCRIPTEN)

else()
    set_target_properties(${MAIN_LIB} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${WEB_OUTPUT_DIR}
    )
    target_compile_options(${MAIN_LIB} PUBLIC
        -O2
        -flto -fno-rtti
    )
    set_target_properties(${MAIN_LIB} PROPERTIES LINK_FLAGS "\
        -O2 \
        -flto \
        -sEXPORTED_FUNCTIONS=_malloc,_free,_memcpy -sEXPORTED_RUNTIME_METHODS=ccall \
        -sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=1gb -sUSE_WEBGPU=1 -sASYNCIFY\
    ")
endif()

# Main JS
if(EMSCRIPTEN)
    file(GLOB JS_SOURCES ${MAIN_PATH}/js/*.js)
    set(JS_OUTPUTS)
    foreach(JS_FILE ${JS_SOURCES})
        get_filename_component(JS_FILENAME ${JS_FILE} NAME)
        set(JS_OUTPUT ${WEB_OUTPUT_DIR}/${JS_FILENAME})
        list(APPEND JS_OUTPUTS ${JS_OUTPUT})
        add_custom_command(
            OUTPUT ${JS_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E copy ${JS_FILE} ${JS_OUTPUT}
            DEPENDS ${JS_FILE}
            COMMENT "Copying ${JS_FILENAME} to web output directory"
        )
    endforeach()
    add_custom_target(${MAIN_JS} ALL
        DEPENDS ${JS_OUTPUTS}
        COMMENT "Copying JS sources to web output directory"
    )
    add_dependencies(${MAIN_LIB} ${MAIN_JS})

    # Configure the modify_js.cmake script
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modify_js.cmake.in
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modify_js.cmake
        @ONLY
    )

    # Add custom command to modify the output JS file
    add_custom_command(
        TARGET ${MAIN_LIB} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Modifying ${MAIN_LIB}.js to use locateFile"
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modify_js.cmake
        WORKING_DIRECTORY ${WEB_OUTPUT_DIR}
        COMMENT "Post-processing ${MAIN_LIB}.js"
    )
endif()

set(TEST_SOURCES
    ${MAIN_PATH}/test/minigpu_test.cpp
)

If(NOT EMSCRIPTEN)
add_executable(minigpu_test ${TEST_SOURCES})
target_sources(${MAIN_LIB} PUBLIC ${MAIN_SOURCES} ${TEST_SOURCES})
else()
target_sources(${MAIN_LIB} PUBLIC ${MAIN_SOURCES})
endif()


target_link_libraries(${MAIN_LIB} PRIVATE gpu)
target_link_libraries(${MAIN_LIB} PRIVATE webgpu)

if(NOT EMSCRIPTEN)
target_link_libraries(minigpu_test PRIVATE gpu)
target_link_libraries(minigpu_test PRIVATE webgpu)
target_link_libraries(minigpu_test PRIVATE ${MAIN_LIB})
endif()


# Additional libs
if(ANDROID)
    target_link_libraries(${MAIN_LIB} OpenSLES log)
endif()