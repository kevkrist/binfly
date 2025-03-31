#=============================================================================
# Tests
#=============================================================================
cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

include(CTest)

###################################################################################################
# - Fetch Google Test -----------------------------------------------------------------------------

CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG main
)

# Ensure Google Test is correctly included
if(googletest_ADDED)
    include(GoogleTest)
endif()

###################################################################################################

function(ConfigureTest TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME} PRIVATE gtest_main binfly CUDA::cudart)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(${TEST_NAME} PROPERTIES
                                       CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${TEST_NAME} PROPERTIES
                                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    if (MSVC)
        target_compile_options(${TEST_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /MDd>  # MSVC warnings for C++ files only
        )
        target_compile_options(${TEST_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -Xcompiler=/W4>
        )
    else()
        target_compile_options(${TEST_NAME} PRIVATE
            -Wall -Wextra --expt-extended-lambda -Xcompiler=-Wno-subobject-linkage
        )
    endif()
    gtest_discover_tests(${TEST_NAME} EXTRA_ARGS --allow-running-no-tests)
endfunction(ConfigureTest)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - host tests ------------------------------------------------------------------------------------

ConfigureTest(BINARY_SEARCH_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/binary_search_test.cu")
ConfigureTest(WARP_SEARCH_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/warp_search_test.cu")
ConfigureTest(PARTITIONED_SEARCH_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/partitioned_search_test.cu")
ConfigureTest(BLOCK_SEARCH_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/block_search_test.cu")
ConfigureTest(DEVICE_BINFLY_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/device_binfly_test.cu")