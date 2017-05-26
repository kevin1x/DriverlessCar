cmake_minimum_required(VERSION 2.8)
project(xla)
# find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
set(SIMPLE_VERSION 0.46+devel)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

set (source_files
	main.cpp
	xla.h
	xla.cpp
	lib/line_segment.h
	lib/line_segment.cpp
	lib/api_lane_detection.h
	lib/api_lane_detection.cpp
    lib/IPM.h
    lib/IPM.cpp
    lib/line_detect_topview.cpp
    lib/line_detect_topview.h
    lib/object_detect_topview.cpp
    lib/object_detect_topview.h    
	lib/msac/MSAC.h
	lib/msac/MSAC.cpp
	lib/msac/errorNIETO.h
	lib/msac/errorNIETO.cpp
	lib/msac/lmmin.h
	lib/msac/lmmin.cpp
)

set( test_exec_name xla)
add_executable(${test_exec_name} ${source_files})
target_link_libraries( ${test_exec_name} ${OpenCV_LIBS})
