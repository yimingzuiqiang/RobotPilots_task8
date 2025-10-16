cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(YoloVino)

#--------------链接自己的类---------------


add_executable(calibration ./src/calibration.cpp)


target_include_directories(
    calibration 
    PUBLIC 
    ${CMAKE_SOURCE_DIR}/lib/include
)

target_link_libraries(calibration PUBLIC Camera)

set_target_properties(
    calibration 
    PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/exe
)

