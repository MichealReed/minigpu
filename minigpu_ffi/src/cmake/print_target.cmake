function(print_target target)
    if(NOT TARGET ${target})
        message(STATUS "Target '${target}' does not exist - skipping property print")
        return()
    endif()

    message(STATUS "=== Properties for target: ${target} ===")
    
    # List of properties to print
    set(properties 
        INCLUDE_DIRECTORIES
        SOURCES
        HEADERS
        LINKED_LIBRARIES
    )

    # Iterate through each property
    foreach(prop ${properties})
        get_property(value TARGET ${target} PROPERTY ${prop})
        message(STATUS "${prop}:")
        
        if(value)
            foreach(item ${value})
                message(STATUS "  ${item}")
            endforeach()
        else()
            message(STATUS "  None")
        endif()
    endforeach()
    
    message(STATUS "=====================================")
endfunction()