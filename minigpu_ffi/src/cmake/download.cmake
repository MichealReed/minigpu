function(download_files)
    foreach(FILE_NAME ${ARGN})
        set(FILE_PATH "${FILE_NAME}_PATH")
        set(FILE_URL "${FILE_NAME}_URL")

        get_filename_component(DOWNLOAD_DIR "${${FILE_PATH}}" DIRECTORY)
        get_filename_component(FILENAME "${${FILE_PATH}}" NAME)

        if(NOT EXISTS "${DOWNLOAD_DIR}")
            file(MAKE_DIRECTORY "${DOWNLOAD_DIR}")
        endif()

        if(NOT EXISTS "${${FILE_PATH}}")
            file(DOWNLOAD "${${FILE_URL}}" "${${FILE_PATH}}" STATUS DOWNLOAD_STATUS)
            list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
            if(STATUS_CODE EQUAL 0)
                message(STATUS "Downloaded: ${FILENAME}")
            else()
                message(FATAL_ERROR "Failed to download ${FILENAME} (status code: ${STATUS_CODE})")
            endif()
        endif()
    endforeach()
endfunction()

function(download_repository REPO_NAME REPO_URL REPO_PATH)
    # Optional fourth argument: branch name
    set(BRANCH "")
    if(ARGC GREATER 3)
        list(GET ARGV 3 BRANCH)
    endif()

    if(NOT EXISTS "${REPO_PATH}/.git")
        message(STATUS "Downloading ${REPO_NAME} repository...")
        file(MAKE_DIRECTORY "${REPO_PATH}")
        if(NOT "${BRANCH}" STREQUAL "")
            execute_process(
                COMMAND git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO_PATH}"
                RESULT_VARIABLE CLONE_RESULT
            )
        else()
            execute_process(
                COMMAND git clone "${REPO_URL}" "${REPO_PATH}"
                RESULT_VARIABLE CLONE_RESULT
            )
        endif()
        if(NOT CLONE_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to clone ${REPO_NAME} repository")
        endif()
    else()
        message(STATUS "${REPO_NAME} repository already exists")
        if(NOT "${BRANCH}" STREQUAL "")
            execute_process(
                COMMAND git fetch --quiet origin "${BRANCH}"
                WORKING_DIRECTORY "${REPO_PATH}"
                RESULT_VARIABLE FETCH_RESULT
            )
            execute_process(
                COMMAND git checkout "${BRANCH}"
                WORKING_DIRECTORY "${REPO_PATH}"
                RESULT_VARIABLE CHECKOUT_RESULT
            )
            if(NOT CHECKOUT_RESULT EQUAL 0)
                message(WARNING "Failed to checkout branch ${BRANCH} for ${REPO_NAME} repository")
            endif()
        else()
            execute_process(
                COMMAND git fetch --quiet
                WORKING_DIRECTORY "${REPO_PATH}"
                RESULT_VARIABLE FETCH_RESULT
            )
        endif()
        if(NOT FETCH_RESULT EQUAL 0)
            message(WARNING "Failed to fetch updates for ${REPO_NAME} repository")
        endif()
    endif()
endfunction()
