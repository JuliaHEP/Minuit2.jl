cmake_minimum_required(VERSION 3.21)
project(Minuit2Wrap)

set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

#---Find JlCxx package-------------------------------------------------------------
find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")
message(STATUS "Found JlCxx at ${JlCxx_location}")
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)
#---Find Minuit2-------------------------------------------------------------------
find_package(Minuit2 REQUIRED)

#  Files are in gen folder
file(REAL_PATH ${CMAKE_SOURCE_DIR}/../gen SOURCE_DIR)
file(GLOB GEN_SOURCES CONFIGURE_DEPENDS  ${SOURCE_DIR}/cpp/Jl*.cxx)
add_library(Minuit2Wrap SHARED ${SOURCE_DIR}/cpp/Minuit2Wrap.cxx ${SOURCE_DIR}/cpp/jlMinuit2.cxx ${GEN_SOURCES})

target_include_directories(Minuit2Wrap PRIVATE ${SOURCE_DIR})
target_link_libraries(Minuit2Wrap JlCxx::cxxwrap_julia JlCxx::cxxwrap_julia_stl Minuit2::Minuit2)

install(TARGETS Minuit2Wrap
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib)
