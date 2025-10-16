cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(Task8)

find_package(OpenCV REQUIRED)


add_subdirectory(./src/YoloVino)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE YoloVino_LIB)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

set_target_properties(
    ${PROJECT_NAME} 
    PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/exe
)