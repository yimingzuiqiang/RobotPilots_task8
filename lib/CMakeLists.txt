cmake_minimum_required(VERSION 3.10)
project(mylib)

add_library(Camera SHARED ./src/Camera.cpp)

target_include_directories(Camera PUBLIC ${CMAKE_SOURCE_DIR}/lib/include/)

#-----------------OpenCV--------------------
find_package(OpenCV 4.6.0 REQUIRED)

target_include_directories(
    Camera 
    PUBLIC 
    ${OpenCV_INCLUDE_DIRS}
)

#链接库
target_link_libraries(
    Camera 
    PUBLIC 
     ${OpenCV_LIBS}
)

#----------------MvCameraControl----------------
set(MV_SOURCE_DIR "/opt/MVS")

target_link_directories(
    Camera 
    PUBLIC 
    ${MV_SOURCE_DIR}/lib/64 
)

target_include_directories(
    Camera 
    PUBLIC
    ${MV_SOURCE_DIR}/include 
)

target_link_libraries(
    Camera 
    PUBLIC 
    MvCameraControl 
)

