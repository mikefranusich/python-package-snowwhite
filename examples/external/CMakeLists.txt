##
## Copyright (c) 2018-2022, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required (VERSION 3.14)

set (CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel")
set ( FILEROOT "testfuncs"  CACHE STRING "C source and target library root name")


if ( NOT DEFINED PY_LIBS_DIR )
    set ( PY_LIBS_DIR ${CMAKE_SOURCE_DIR} )
endif ()


project ( TEST_FUNCS
    VERSION 1.0.1
    DESCRIPTION "test functions"
    LANGUAGES C CXX )

set ( SOURCES ${FILEROOT}.c )


##  Setup flags if needed

if (WIN32)
    set ( LIB_PREF lib )
    set ( CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON )
else ()
    set ( LIB_PREF )
endif ()

set ( PROJECT ${LIB_PREF}${FILEROOT} )

add_library ( ${PROJECT} SHARED ${SOURCES} )
##  target_compile_options     (${PROJECT}  PRIVATE )		## any flags needed?


target_include_directories ( ${PROJECT} PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} )

install (TARGETS ${PROJECT} DESTINATION ${PY_LIBS_DIR})

