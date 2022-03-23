cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

project(StreamSafetyFirst
        VERSION 00.01.00
        LANGUAGES CXX CUDA)

# ---- Dependencies ----

# ---- Get CPM ----

set(CPM_DOWNLOAD_VERSION 0.34.0)

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage("gh:catchorg/Catch2@2.13.8")

if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
  set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures" FORCE)
endif()
message ("CUDA ARCHS: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Create ranger library ----

# library targets
add_library(ranger INTERFACE)
add_library(ranger::ranger ALIAS ranger)

target_include_directories(ranger INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                         "$<INSTALL_INTERFACE:include>")

# install export targets
install(TARGETS ranger EXPORT ranger-exports)
install(DIRECTORY include/ranger/ DESTINATION include/ranger)

# ---- Built test binary ----

add_executable(ranger_test test_ranger.cu)
target_link_libraries(ranger_test Catch2 ranger)
target_compile_features(ranger_test PRIVATE cxx_std_17)

# ---- Enable testing ----

enable_testing()
add_test(ranger_test ranger_test)
