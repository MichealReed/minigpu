# Specify the filename to search for
set(FILENAME ".projectroot")

# Function to check for file existence up the directory hierarchy
function(find_project_root current_dir filename result_var)
    set(found FALSE)  # Flag to indicate if the file is found
    set(current_check_dir "${current_dir}")  # Start from the given directory
    
    # Windows flutter embeds 8 levels deep
    foreach(i RANGE 0 8)
        set(filepath "${current_check_dir}/${filename}")
        
        if(EXISTS "${filepath}")
            get_filename_component(parent_dir "${current_check_dir}" DIRECTORY)
            set(${result_var} "${parent_dir}" PARENT_SCOPE)
            set(found TRUE)
            break()
        endif()
        
        # Move one level up
        get_filename_component(current_check_dir "${current_check_dir}" DIRECTORY)
    endforeach()
    
    if(NOT found)
        set(${result_var} "" PARENT_SCOPE)  # Set to empty if not found
    endif()
endfunction()