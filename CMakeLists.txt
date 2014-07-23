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

# message(STATUS "Build type is '${CMAKE_BUILD_TYPE}'")

if (WIN32)
  find_package(openmp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -UNDEBUG")
else()
  set(CMAKE_CXX_FLAGS "-Wall -fopenmp")
endif()

### on WIN32, /EHsc is already included by default
### message(STATUS "CXX_FLAGS is '${CMAKE_CXX_FLAGS}'")

find_package(CUDA REQUIRED)

if (WIN32)
  set ( TIFF_LIBRARY c:/libtiff )
  set ( TIFF_INCLUDE_DIR c:/libtiff )
endif(WIN32)
find_package(TIFF REQUIRED)

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

CUDA_ADD_LIBRARY(
	libcudaDeconv
    SHARED
	RL-Biggs-Andrews.cpp
	RLgpuImpl.cu
	geometryTransform.cu
  )

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

add_dependencies(
  libcudaDeconv
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
	libcudaDeconv
	Buffer
	libtiff
	libfftwf-3.3
	)
  CUDA_ADD_EXECUTABLE(
	RL_interface_driver
	RL_interface_driver.cpp
	)
  add_dependencies(
	RL_interface_driver
	libcudaDeconv
	)
  target_link_libraries(
	RL_interface_driver
	libcudaDeconv
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
CUDA_ADD_CUFFT_TO_TARGET(libcudaDeconv)

