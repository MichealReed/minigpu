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

function(download_files)
    # ... (keep the existing code for the download_files function)
endfunction()

function(download_repository REPO_NAME REPO_URL REPO_PATH)
    set(ZIP_FILE "${CMAKE_CURRENT_BINARY_DIR}/${REPO_NAME}.zip")
    set(EXTRACTED_DIR "${REPO_PATH}/${REPO_NAME}-main")
    set(RENAMED_DIR "${REPO_PATH}/${REPO_NAME}")
    
    if(NOT EXISTS "${RENAMED_DIR}" OR NOT EXISTS "${RENAMED_DIR}/*")
        message(STATUS "Setting up ${REPO_NAME} repository...")
        
        # Download the ZIP file if it doesn't exist in the cache
        if(NOT EXISTS "${ZIP_FILE}")
            message(STATUS "Downloading ${REPO_NAME} repository...")
            file(DOWNLOAD "${REPO_URL}/archive/refs/heads/main.zip" "${ZIP_FILE}" STATUS DOWNLOAD_STATUS)
            list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
            if(NOT STATUS_CODE EQUAL 0)
                message(FATAL_ERROR "Failed to download ${REPO_NAME} repository ZIP file")
            endif()
        endif()
        
        # Extract the ZIP file
        message(STATUS "Extracting ${REPO_NAME} repository...")
        file(MAKE_DIRECTORY "${REPO_PATH}")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar -xf "${ZIP_FILE}"
            WORKING_DIRECTORY "${REPO_PATH}"
            RESULT_VARIABLE EXTRACT_RESULT
        )
        if(NOT EXTRACT_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to extract ${REPO_NAME} repository ZIP file")
        endif()
        
        # Rename the extracted directory to remove the "-main" suffix
        if(EXISTS "${EXTRACTED_DIR}")
            file(RENAME "${EXTRACTED_DIR}" "${RENAMED_DIR}")
        endif()
    endif()
endfunction()

