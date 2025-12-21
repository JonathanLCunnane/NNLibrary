cmake_minimum_required(VERSION 3.23)

project(NeuralNetworkLibrary VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(demo)
target_sources(demo
  PRIVATE
    src/demo.cpp
)

#[[
add_library(NeuralNetworkLibrary)
target_sources(NeuralNetworkLibrary
  PRIVATE
    src/foo.cpp
 
  PUBLIC
  FILE_SET neuralNetworkLibraryHeaders
    TYPE HEADERS
    BASE_DIRS
      include
    FILES
      include/foo.h
)
]]
