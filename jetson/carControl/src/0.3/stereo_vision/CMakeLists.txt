
cmake_minimum_required(VERSION 2.8)

project( OpenCV-Kinect )
find_package( OpenCV REQUIRED )
set( OPENNI2_INCLUDE /home/ubuntu/Downloads/2-Linux/OpenNI-Linux-Arm-2.3/Include)
set( OPENNI2_REDIST  /home/ubuntu/Downloads/2-Linux/OpenNI-Linux-Arm-2.3/Redist)
include_directories( ${OPENNI2_INCLUDE} )
link_directories ( ${OPENNI2_REDIST} )

set(libname kinect-cv2 )

set(source_files
api_kinect_cv.cpp
api_kinect_cv.h
)

add_library( ${libname} ${source_files})

target_link_libraries(${test_exec_name2} ${libname} ${OpenCV_LIBS})


set (test_exec_name2 test-kinect-cv2)

add_executable( ${test_exec_name2} test_kinect_cv2.cpp)

target_link_libraries(${test_exec_name2} ${libname})



set (test_exec_name3 test-kinect-cv2-2)

add_executable( ${test_exec_name3} test_kinect_cv2_2.cpp)

target_link_libraries(${test_exec_name3} ${libname})
