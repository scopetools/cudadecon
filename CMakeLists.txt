
cmake_minimum_required (VERSION 2.6)

project (cudaDeconv)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(VERSION_MAJOR "0")
set(VERSION_MINOR "0")
set(VERSION_PATCH "1")
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

######################################################################
#
# Set permissions before adding subdirectories
#
######################################################################
set(SCI_GROUP_WRITE GROUP_WRITE)
set(SCI_WORLD_FILE_PERMS WORLD_READ)
set(SCI_WORLD_PROGRAM_PERMS WORLD_READ WORLD_EXECUTE)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()
message(STATUS "Build type is '${CMAKE_BUILD_TYPE}'")

if (NOT APPLE)
  find_package(openmp)
  set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}" )
endif()

if (WIN32)
  # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /EHsc")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /U NDEBUG")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
endif()

message(STATUS "CXX_FLAGS_Release is '${CMAKE_CXX_FLAGS_RELEASE}'")
message(STATUS "CXX_FLAGS is '${CMAKE_CXX_FLAGS}'")
message(STATUS "CXX_FLAGS_DEBUG is '${CMAKE_CXX_FLAGS_DEBUG}'")
message(STATUS "CXX_FLAGS_RELWITHDEBINFO is '${CMAKE_CXX_FLAGS_RELWITHDEBINFO}'")

find_package(CUDA REQUIRED)

if (WIN32)
  set (FFTW3_ROOT_DIR C:/fftw3)
  set (FFTW3F_LIBRARY libfftwf-3.3)
  set (TIFF_LIBRARY c:/libtiff)
  set (TIFF_INCLUDE_DIR c:/libtiff)
endif()

find_package(FFTW3 COMPONENTS single) # threads)
find_package(TIFF REQUIRED)

if (WIN32)
  set (TIFF_LIBRARIES libtiff)
endif()

# message(STATUS "TIFF_LIBRARY is '${TIFF_LIBRARY}'")
# message(STATUS "TIFF_LIBRARIES is '${TIFF_LIBRARIES}'")

if (NOT WIN32)
  find_package(X11)
endif()

if(WIN32)
  find_package( Boost )
else()
  find_package( Boost REQUIRED COMPONENTS program_options filesystem regex system)
endif(WIN32)


set(CUDA_NVCC_FLAGS "-O2;-gencode=arch=compute_20,code=sm_20;-gencode=arch=compute_30,code=sm_30;-gencode=arch=compute_35,code=sm_35")
#--ptxas-options=-v;

add_subdirectory(Buffers)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  "${CMAKE_CURRENT_SOURCE_DIR}/Buffers"
  ${CUDA_INCLUDE_DIRS}
  ${FFTW3_INCLUDE_DIR}
  ${X11_INCLUDE_DIR}
  )


if(WIN32)
  set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};--use-local-env;--cl-version=2013)
  link_directories ( ${Boost_LIBRARY_DIRS} ${FFTW3_INCLUDE_DIR} ${TIFF_LIBRARY} )
  include_directories( ${Boost_INCLUDE_DIRS} ${FFTW3_INCLUDE_DIR}  ${TIFF_INCLUDE_DIR}
    "$ENV{NVCUDASAMPLES_ROOT}/common/inc" )
else()
  include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc )
endif()

# CUDA_ADD_LIBRARY(
#   libcudaDeconv
#   SHARED
#   RL-Biggs-Andrews.cpp
#   RLgpuImpl.cu
#   geometryTransform.cu
#   )

CUDA_ADD_EXECUTABLE(
  cudaDeconv
  linearDecon.cpp
  RL-Biggs-Andrews.cpp
  boostfs.cpp
  RLgpuImpl.cu
  geometryTransform.cu
  )

add_dependencies(
  cudaDeconv
  Buffer
  )

# add_dependencies(
#   libcudaDeconv
#   Buffer
#   )

add_executable(radialft radialft-nonSIM.cpp)
add_executable(otfviewer OTF_TIFF_viewer.cpp)

target_link_libraries(
  cudaDeconv
  Buffer
  ${FFTW3F_LIBRARY}
  ${FFTW3F_THREADS_LIBRARY}
  ${X11_X11_LIB}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_REGEX_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${TIFF_LIBRARIES}
  )
target_link_libraries(
  radialft
  ${FFTW3F_LIBRARY}
  ${FFTW3F_THREADS_LIBRARY}
  ${X11_X11_LIB}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${TIFF_LIBRARIES}
  )
target_link_libraries(
  otfviewer
  ${TIFF_LIBRARIES}
  ${X11_X11_LIB}
  )

CUDA_ADD_CUFFT_TO_TARGET(cudaDeconv)
# CUDA_ADD_CUFFT_TO_TARGET(libcudaDeconv)

