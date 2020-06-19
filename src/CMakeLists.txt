######################################################################
#
# This will make the following outputs : 
#
# cudaDecon.exe
# otfviewer.exe
# radialft.exe
# libcudaDecon.dll
#
######################################################################


cmake_minimum_required (VERSION 3.10)
# cmake_policy(SET CMP0054 OLD) #Set IF statements to dereference variables like in CMAKE version < 3.1
# cmake_policy(SET CMP0012 NEW) #Set IF statements to use values of numbers and booleans rather than pretend that they could be variables

project (cudaDecon)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(VERSION_MAJOR "0")
set(VERSION_MINOR "4")
set(VERSION_PATCH "1")
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

# set -fPIC for all targets (for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#add_compile_definitions(cimg_display=0)

######################################################################
#
# Set permissions before adding subdirectories
#
######################################################################
set(SCI_GROUP_WRITE GROUP_WRITE)
set(SCI_WORLD_FILE_PERMS WORLD_READ)
set(SCI_WORLD_PROGRAM_PERMS WORLD_READ WORLD_EXECUTE)


# If no CMAKE_BUILD_TYPE is specified on command line, Windows and Unix
# give different treatments. On Windows, CMAKE_BUILD_TYPE is defaulted to
# "Debug"; on Unix, it's some empty string that cannot be legally placed in
# a IF statement to do string compare like "STREQUAL". Hence the following
# 2-step verification. The goal is that if no Release is specified in
# the command line, we want the compiler flag to be almost identical to the
# ones used for Release except for -DNDEBUG

######################################################################
#
# Set compilier
#
######################################################################

if ( NOT CMAKE_BUILD_TYPE )
  set(CMAKE_BUILD_TYPE "Debug")
endif()


if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(CMAKE_BUILD_TYPE "Release")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS}")
  if (WIN32)
	  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
  endif()
endif()


set(platform_dir ${CMAKE_SYSTEM_NAME}-${CMAKE_CXX_COMPILER_ID})

foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)

    foreach(var CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config} CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config} CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config})
        set(${var} "${CMAKE_BINARY_DIR}/${platform_dir}-${config}")
        string(TOLOWER "${${var}}" ${var})  ## why TOLOWER?? --lin
    endforeach()

endforeach()


# message(STATUS "2 Build type is '${CMAKE_BUILD_TYPE}'")

if (APPLE)
  set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++")
else()
  find_package(OpenMP)
  set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}" )
endif()

if (NOT WIN32)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
endif()

if (CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.4)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++14")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11")
  endif()
endif()



######################################################################
#
# Set packages
#
######################################################################

find_package(CUDA REQUIRED)
if (WIN32)
  set (FFTW3_INCLUDE_DIR $ENV{CONDA_PREFIX}/Library/lib)
  set (FFTW3F_LIBRARY fftw3f)
  find_package(FFTW3 COMPONENTS single threads REQUIRED)
else()
  set (FFTW3_INCLUDE_DIR $ENV{CONDA_PREFIX}/include)
  find_library(FFTW3F_LIBRARY fftw3f REQUIRED)
  find_library(FFTW3F_THREADS_LIBRARY fftw3f_threads REQUIRED)
endif()

find_library(TIFF_LIBRARIES tiff 
           HINTS
           $ENV{CONDA_PREFIX}/lib
           REQUIRED)
message(STATUS "Tiff Library: " ${TIFF_LIBRARIES})
message(STATUS "FFTW3 Library: " ${FFTW3F_LIBRARY})

find_package(Boost REQUIRED COMPONENTS program_options filesystem system)


message(STATUS "CUDA version : ${CUDA_VERSION}") 
if(NOT(CUDA_VERSION LESS 10.0))  # if CUDA version is > or = 10.0  
    set(CUDA_NVCC_FLAGS "-O3;\
	-lcudart;\
	-gencode=arch=compute_52,code=sm_52;\ 
    -gencode=arch=compute_61,code=sm_61;\
    -gencode=arch=compute_62,code=sm_62;\
    -gencode=arch=compute_70,code=sm_70;\
    -gencode=arch=compute_72,code=sm_72;\
    -gencode=arch=compute_75,code=sm_75,\
    -gencode=arch=compute_75,code=compute_75") # make available to future hardware see JIT compilation : https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#just-in-time-compilation
  elseif(NOT(CUDA_VERSION LESS 8.0))  # if CUDA version is > or = 8.0
	  set(CUDA_NVCC_FLAGS "-O2;\
		-gencode=arch=compute_30,code=sm_30;\
		-gencode=arch=compute_35,code=sm_35;\
		-gencode=arch=compute_37,code=sm_37;\
		-gencode=arch=compute_50,code=sm_50;\
		-gencode=arch=compute_52,code=sm_52;\
		-gencode=arch=compute_61,code=sm_61") # compute capability 6.1 only avaiable in CUDA 8.0 or greater
	else()
		set(CUDA_NVCC_FLAGS "-O2;\
		-gencode=arch=compute_20,code=sm_20;\
		-gencode=arch=compute_30,code=sm_30;\
		-gencode=arch=compute_35,code=sm_35;\
		-gencode=arch=compute_37,code=sm_37;\
		-gencode=arch=compute_50,code=sm_50;\
		-gencode=arch=compute_52,code=sm_52")
											  # compute capability 6.1 only avaiable in CUDA 8.0 or greater
endif()

# Specifies which compute capabilities are supported by the built .exe http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list 
# Need to add to these if different GPUs are targeted.
#--ptxas-options=-v;

######################################################################
#
# Add subdirectories
#
######################################################################

add_subdirectory(Buffers)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  "${CMAKE_CURRENT_SOURCE_DIR}/Buffers"
  ${CUDA_INCLUDE_DIRS}
  ${FFTW3_INCLUDE_DIR}
  # ${X11_INCLUDE_DIR}
)


if(WIN32)
  set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};--use-local-env;)
  link_directories ( ${Boost_LIBRARY_DIRS} ${FFTW3_INCLUDE_DIR} ${TIFF_LIBRARY} )
  include_directories( ${Boost_INCLUDE_DIRS} ${FFTW3_INCLUDE_DIR}  ${TIFF_INCLUDE_DIR}
    "$ENV{NVCUDASAMPLES_ROOT}/common/inc" )
else()
  include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc )
endif()




##### Make this everything time so that linearDecon.cpp is remade each time.  This will ensure that the build date is correct #####
#add_custom_target(invalidate_files ALL
#                  COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_SOURCE_DIR}/linearDecon.cpp")


######################################################################
#
# Add build outputs and their dependencies
#
######################################################################



#####  libcudaDecon.dll ###########################################

 CUDA_ADD_LIBRARY(
   libcudaDecon
   SHARED
   RL-Biggs-Andrews.cpp
   RLgpuImpl.cu
   geometryTransform.cu
#   camcor.cpp
   camcor.cu
   )

add_dependencies(
   libcudaDecon
   Buffer
   )

if (APPLE)
  set_target_properties(libcudaDecon
                        PROPERTIES
                        OUTPUT_NAME cudaDecon
                        MACOSX_RPATH 1
                        POSITION_INDEPENDENT_CODE ON)
endif()


#####  libLLSpy.dll ###########################################

#  CUDA_ADD_LIBRARY(
#    libLLSpy
#    SHARED
#    geometryTransform.cu
#    camcor.cu
#    llspy.cpp
#    )
# 
# add_dependencies(
#    libLLSpy
#    Buffer
#    )
# 
# if (APPLE)
#   set_target_properties(libLLSpy
#                         PROPERTIES
#                         OUTPUT_NAME LLSpy
#                         MACOSX_RPATH 1)
# endif()


#####  libradialft.dll ###########################################

 add_library(
   libradialft
   SHARED
   radialft_interface.cpp
   )

add_dependencies(
   libradialft
   Buffer
   )

if (WIN32)
  set_property(TARGET libradialft PROPERTY POSITION_INDEPENDENT_CODE ON)
else()
  set_target_properties(libradialft
                        PROPERTIES
                        OUTPUT_NAME radialft
                        MACOSX_RPATH 1
                        POSITION_INDEPENDENT_CODE ON)
endif()




#####  cudaDecon.exe ###########################################

CUDA_ADD_EXECUTABLE(
  cudaDecon
  linearDecon.cpp
  RL-Biggs-Andrews.cpp
  boostfs.cpp
  RLgpuImpl.cu
  geometryTransform.cu
  camcor.cu
  )

add_dependencies(
  cudaDecon
  Buffer
  )


#####  radialft.exe ###########################################
add_executable(radialft radialft-nonSIM.cpp)

#####  otfviewer.exe ###########################################
# add_executable(otfviewer OTF_TIFF_viewer.cpp)


######################################################################
#
# Link libraries
#
######################################################################

target_link_libraries(
  cudaDecon
  Buffer
  ${FFTW3F_LIBRARY}
  ${FFTW3F_THREADS_LIBRARY}
  # ${X11_X11_LIB}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PREDICATE_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${TIFF_LIBRARIES}
  )

if (UNIX AND NOT APPLE)
  target_link_libraries(cudaDecon rt)
endif ()

target_link_libraries(
  radialft
  ${FFTW3F_LIBRARY}
  ${FFTW3F_THREADS_LIBRARY}
  # ${X11_X11_LIB}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${TIFF_LIBRARIES}
)

# target_link_libraries(
#   otfviewer
#   ${TIFF_LIBRARIES}
#   ${X11_X11_LIB}
#   )

# target_link_libraries(
#   libLLSpy
#   Buffer
#   ${X11_X11_LIB}
#   )


target_link_libraries(
  libcudaDecon
  Buffer
  ${FFTW3F_LIBRARY}
  ${FFTW3F_THREADS_LIBRARY}
  # ${X11_X11_LIB}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PREDICATE_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${TIFF_LIBRARIES}
)
CUDA_ADD_CUFFT_TO_TARGET(libcudaDecon)


target_link_libraries(
  libradialft
  ${FFTW3F_LIBRARY}
  ${TIFF_LIBRARIES}
)

CUDA_ADD_CUFFT_TO_TARGET(cudaDecon)

# added for make install to work in conda
set(HEADERS
  CImg.h
  linearDecon.h
)


install(
  TARGETS cudaDecon libcudaDecon radialft libradialft
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(
  FILES ${HEADERS} DESTINATION include
)
