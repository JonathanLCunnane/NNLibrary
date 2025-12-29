cmake_minimum_required(VERSION 3.23)

project(NeuralNetworkLibrary VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
set(BUILD_TESTING OFF CACHE BOOL "Disable building tests for Fastor" FORCE)
FetchContent_Declare(
  fastor
  URL https://github.com/romeric/Fastor/archive/refs/tags/V0.6.4.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(fastor)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(tests)
target_sources(tests
  PRIVATE
    test/network_test.cpp
    test/activation_test.cpp
    test/layer_test.cpp
)
target_link_libraries(
  tests
  GTest::gtest_main
  GTest::gmock
  Fastor
)

include(GoogleTest)
gtest_discover_tests(tests)

add_executable(demo)
target_sources(demo
  PRIVATE
    src/demo.cpp
)
target_link_libraries(demo
  PRIVATE
    Fastor
)
