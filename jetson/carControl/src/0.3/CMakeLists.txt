cmake_minimum_required(VERSION 2.8)

project( CarControl-0.3 )

find_package( OpenCV REQUIRED )
set( OPENNI2_INCLUDE /home/ubuntu/Downloads/2-Linux/OpenNI-Linux-Arm-2.3/Include)
set( OPENNI2_REDIST  /home/ubuntu/Downloads/2-Linux/OpenNI-Linux-Arm-2.3/Redist)
include_directories( ${OPENNI2_INCLUDE} )

link_directories ( ${OPENNI2_REDIST} )


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
               STRINGS "Debug" "Release")
endif()

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})

include_directories(
      ${OpenCV_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/stereo_vision
      ${CMAKE_CURRENT_SOURCE_DIR}/lane_detection
      ${CMAKE_CURRENT_SOURCE_DIR}/lane_detection/msac
      ${CMAKE_CURRENT_SOURCE_DIR}/peripheral_driver/i2c
      ${CMAKE_CURRENT_SOURCE_DIR}/peripheral_driver/uart
      ${CMAKE_CURRENT_SOURCE_DIR}/xla
)

link_directories( ${LIB_FREENECT_INSTALL_DIR}/lib
                  ${LIBRARY_OUTPUT_PATH}
)

find_package(OpenMP)
if(OPENMP_FOUND)
    message("Find OpenMP: ${OpenMP_C_FLAGS}, ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory(lane_detection)
add_subdirectory(peripheral_driver)
add_subdirectory(stereo_vision)


set( exec_name1 test-steering-coef)
set( source_files1
    main_pid/test-steering-coef.cpp
)
add_executable( ${exec_name1} ${source_files1} )
target_link_libraries( ${exec_name1} i2c-pwm uart)


set( exec_name2 test-autocar)
set( source_files2
    #main_pid/collecting_data.cpp
    main_pid/automaticCar.cpp
    # if you write extra libraries for automaticCar.cpp, declare them here.
    # See CMakeLists.txt in sub-directory for more details. Eg: 
    # main_pid/depth.h
    # main_pid/depth.cpp

	xla/xla.h
	xla/xla.cpp
	xla/lib/line_segment.h
	xla/lib/line_segment.cpp
	xla/lib/api_lane_detection.h
	xla/lib/api_lane_detection.cpp
    xla/lib/IPM.h
    xla/lib/IPM.cpp
    xla/lib/line_detect_topview.cpp
    xla/lib/line_detect_topview.h
    xla/lib/object_detect_topview.cpp
    xla/lib/object_detect_topview.h    
	xla/lib/msac/MSAC.h
	xla/lib/msac/MSAC.cpp
	xla/lib/msac/errorNIETO.h
	xla/lib/msac/errorNIETO.cpp
	xla/lib/msac/lmmin.h
	xla/lib/msac/lmmin.cpp
)
add_executable( ${exec_name2} ${source_files2} )
target_link_libraries( ${exec_name2} vanishing-point i2c-pwm uart kinect-cv2 OpenNI2)

set( exec_name3 check-uart)
set( source_files3
    main_pid/checkUart.cpp
)
add_executable( ${exec_name3} ${source_files3} )
target_link_libraries( ${exec_name3} i2c-pwm uart)

set( exec_name4 run-straight)
set( source_files4
    main_pid/runStraight.cpp
)
add_executable( ${exec_name4} ${source_files4} )
target_link_libraries( ${exec_name4} i2c-pwm uart)
