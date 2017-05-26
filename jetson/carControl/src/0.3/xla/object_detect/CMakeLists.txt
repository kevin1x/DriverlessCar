cmake_minimum_required(VERSION 2.8)
project(object)
find_package(OpenCV REQUIRED)
set(SIMPLE_VERSION 0.46+devel)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

set (source_files
	object_detect.cpp
)

set( test_exec_name object)
add_executable(${test_exec_name} ${source_files})
target_link_libraries( ${test_exec_name} ${OpenCV_LIBS})
