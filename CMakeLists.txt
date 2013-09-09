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

if(WIN32)
set(CMAKE_CXX_FLAGS
  ${CMAKE_CXX_FLAGS}
  " /openmp")
set(CMAKE_CXX_RELEASE_FLAGS
  ${CMAKE_CXX_RELEASE_FLAGS}
  " /DNDEBUG /O2")
else(WIN32)
set(CMAKE_CXX_FLAGS
  ${CMAKE_CXX_FLAGS}
  "-fopenmp")
set(CMAKE_CXX_RELEASE_FLAGS
  ${CMAKE_CXX_RELEASE_FLAGS}
  "-DNDEBUG -O3")
endif(WIN32)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  if (CMAKE_COMPILER_IS_GNUCC)
    execute_process(COMMAND gcc -dumpversion
    OUTPUT_VARIABLE GCC_VER)
    if (GCC_VER VERSION_GREATER 4.6.3)
      set ( CMAKE_C_COMPILER /opt/bin/gcc )
      set ( CMAKE_CXX_COMPILER /opt/bin/g++ )
    endif()
    message (STATUS "gcc used: " ${CMAKE_C_COMPILER} )
  endif()
  set ( CUDA_TOOLKIT_ROOT_DIR /opt/cuda-5.0 )
endif()

if (GCC_VER VERSION_GREATER 4.6.3)
endif()


find_package(CUDA REQUIRED)

if (WIN32)
  set ( TIFF_LIBRARY c:/libtiff )
  set ( TIFF_INCLUDE_DIR c:/libtiff )
endif(WIN32)
find_package(TIFF REQUIRED)

if(WIN32)
  find_package( Boost )
else()
  find_package( Boost REQUIRED COMPONENTS program_options )
endif(WIN32)

set(CUDA_NVCC_FLAGS
  --ptxas-options=-v;-O3;-gencode=arch=compute_20,code=sm_20;-gencode=arch=compute_30,code=sm_30;-gencode=arch=compute_35,code=sm_35)

if (GCC_VER VERSION_GREATER 4.6.3)
  set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--compiler-bindir=/opt/bin)
endif()

add_subdirectory(Buffers)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDA_INCLUDE_DIRS}
  "${CMAKE_CURRENT_SOURCE_DIR}/Buffers"
)

  
if(WIN32)
  set(CUDA_NVCC_FLAGS
     ${CUDA_NVCC_FLAGS};--use-local-env;--cl-version=2010)
  link_directories ( ${Boost_LIBRARY_DIRS} c:/fftw3 ${TIFF_LIBRARY} )
  include_directories( ${Boost_INCLUDE_DIRS} c:/fftw3  ${TIFF_INCLUDE_DIR}
  "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v5.0/common/inc" )
else(WIN32)
  include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc )
endif()

CUDA_ADD_EXECUTABLE(
	cudaDeconv
	linearDecon.cpp
	RL-Biggs-Andrews.cpp
	RLgpuImpl.cu
	geometryTransform.cu
  )

add_dependencies(cudaDeconv
  Buffer
  )

if(WIN32)
target_link_libraries(
  cudaDeconv
  Buffer
  libtiff
  libfftw3f-3
  )
else(WIN32)
target_link_libraries(
  cudaDeconv
  Buffer
  fftw3f
  fftw3f_threads
  X11
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${TIFF_LIBRARIES}
  )
endif(WIN32)

CUDA_ADD_CUFFT_TO_TARGET(cudaDeconv)


