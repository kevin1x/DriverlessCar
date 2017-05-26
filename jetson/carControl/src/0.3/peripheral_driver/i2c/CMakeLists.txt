cmake_minimum_required(VERSION 2.8)

project( i2c-pwm )

set( lib_name i2c-pwm)

set(source_files
    pca9685.h
    pca9685.cpp
    api_i2c_pwm.cpp
    api_i2c_pwm.h
    ../uart/api_uart.h
    ../uart/api_uart.cpp
    ../uart/rs232.c
    ../uart/rs232.h

)

add_library( ${lib_name} ${source_files})

set( test_exec_name test-steering-throttle)

add_executable( ${test_exec_name} test_steering_throttle.cpp)

target_link_libraries( ${test_exec_name} ${lib_name})
