cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(YoloVino)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(yaml-cpp REQUIRED)

file(GLOB_RECURSE YOLOVINO_SRC src/*.cpp )

add_library(${PROJECT_NAME}_LIB  SHARED ${YOLOVINO_SRC})
target_include_directories(${PROJECT_NAME}_LIB PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME}_LIB  PRIVATE ${OpenCV_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}_LIB PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME}_LIB PRIVATE yaml-cpp)
target_link_libraries(${PROJECT_NAME}_LIB PUBLIC openvino::runtime)

target_include_directories(
    ${PROJECT_NAME}_LIB
    PUBLIC 
    ${CMAKE_SOURCE_DIR}/lib/include
)

target_link_libraries(${PROJECT_NAME}_LIB PUBLIC Camera)
