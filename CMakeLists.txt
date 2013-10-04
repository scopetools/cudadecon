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

message(STATUS "Build type is '${CMAKE_BUILD_TYPE}'")

if (WIN32)
  find_package(openmp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -UNDEBUG")
else()
  set(CMAKE_CXX_FLAGS "-Wall -fopenmp")
endif()

# message(STATUS ${CMAKE_CXX_FLAGS})
# message(STATUS ${CMAKE_CXX_FLAGS_RELEASE})
# message(STATUS ${CMAKE_CXX_FLAGS_DEBUG})
# message(STATUS ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})

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

# message(STATUS "${TIFF_LIBRARIES}")

if(WIN32)
  find_package( Boost )
else()
  find_package( Boost REQUIRED COMPONENTS program_options filesystem regex system)
endif(WIN32)

set(CUDA_NVCC_FLAGS "-O2;-gencode=arch=compute_20,code=sm_20;-gencode=arch=compute_30,code=sm_30;-gencode=arch=compute_35,code=sm_35")
#--ptxas-options=-v;

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
     ${CUDA_NVCC_FLAGS};--use-local-env;--cl-version=2012)
  link_directories ( ${Boost_LIBRARY_DIRS} c:/fftw3 ${TIFF_LIBRARY} )
  include_directories( ${Boost_INCLUDE_DIRS} c:/fftw3  ${TIFF_INCLUDE_DIR}
    "$ENV{NVCUDASAMPLES_ROOT}/common/inc" )
else(WIN32)
  include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc )
endif()

# message(STATUS "${CUDA_NVCC_FLAGS}")

CUDA_ADD_EXECUTABLE(
	cudaDeconv
	linearDecon.cpp
	RL-Biggs-Andrews.cpp
	boostfs.cpp
	RLgpuImpl.cu
	geometryTransform.cu
  )

add_dependencies(cudaDeconv
  Buffer
  )

add_executable(radialft radialft-nonSIM.cpp)
add_executable(otfviewer OTF_TIFF_viewer.cpp)

if(WIN32)
target_link_libraries(
  cudaDeconv
  Buffer
  libtiff
  libfftwf-3.3
  )
target_link_libraries(
  radialft
  libtiff
  libfftwf-3.3
  )
target_link_libraries(
  otfviewer
  libtiff
  )
else(WIN32)
target_link_libraries(
  cudaDeconv
  Buffer
  fftw3f
  fftw3f_threads
  X11
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_REGEX_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${TIFF_LIBRARIES}
  )
target_link_libraries(
  radialft
  fftw3f
  fftw3f_threads
  X11
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${TIFF_LIBRARIES}
  )
target_link_libraries(
  otfviewer
  ${TIFF_LIBRARIES}
  X11
  )
endif(WIN32)

CUDA_ADD_CUFFT_TO_TARGET(cudaDeconv)

