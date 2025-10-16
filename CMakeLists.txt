cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(task8)


set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

add_subdirectory(${CMAKE_SOURCE_DIR}/lib)
add_subdirectory(${CMAKE_SOURCE_DIR}/app)
add_subdirectory(${CMAKE_SOURCE_DIR}/vino_task)








