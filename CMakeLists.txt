cmake_minimum_required(VERSION 3.23)

project(NeuralNetworkLibrary VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
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

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(network_test)
target_sources(network_test
  PRIVATE
    test/network_test.cpp
)
target_link_libraries(
  network_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(network_test)
