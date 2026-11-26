cmake_minimum_required(VERSION 3.16)

project(PCB_DEFECT_DETECTION LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(OpenCV REQUIRED)

# Static library
add_library(pcb_core STATIC
    src/detector.cpp
    src/evaluation.cpp
)

target_include_directories(pcb_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(pcb_core PUBLIC
    ${OpenCV_LIBS}
)

# Batch executable
add_executable(batch src/batch.cpp)
target_link_libraries(batch PRIVATE pcb_core)

# Single executable
add_executable(single src/single.cpp)
target_link_libraries(single PRIVATE pcb_core)