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


cmake_minimum_required (VERSION 3.20)
# cmake_policy(SET CMP0054 OLD) #Set IF statements to dereference variables like in CMAKE version < 3.1
# cmake_policy(SET CMP0012 NEW) #Set IF statements to use values of numbers and booleans rather than pretend that they could be variables

project (cudaDecon VERSION 0.7.0 LANGUAGES CXX CUDA)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# set -fPIC for all targets (for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_definitions(cimg_display=0)

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
        string(TOLOWER "${${var}}" ${var})
    endforeach()

endforeach()


# message(STATUS "2 Build type is '${CMAKE_BUILD_TYPE}'")

find_package(OpenMP)
set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}" )

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

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA version : ${CMAKE_CUDA_COMPILER_VERSION}")
set(CMAKE_CUDA_ARCHITECTURES all-major)

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
)


if(WIN32)
  link_directories ( ${Boost_LIBRARY_DIRS} ${FFTW3_INCLUDE_DIR} ${TIFF_LIBRARY} )
  include_directories( ${Boost_INCLUDE_DIRS} ${FFTW3_INCLUDE_DIR}  ${TIFF_INCLUDE_DIR}
    "$ENV{NVCUDASAMPLES_ROOT}/common/inc" )
else()
  include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc )
endif()

######################################################################
#
# Add build outputs and their dependencies
#
######################################################################

#####  libcudaDecon.dll ###########################################

add_library(
  libcudaDecon
  SHARED
  RL-Biggs-Andrews.cpp
  RLgpuImpl.cu
  geometryTransform.cu
  #  camcor.cu
  )
set_target_properties(libcudaDecon PROPERTIES PREFIX "")


#####  libradialft.dll ###########################################

 add_library(
   libradialft
   SHARED
   radialft_interface.cpp
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

add_executable(
  cudaDecon
  linearDecon.cpp
  RL-Biggs-Andrews.cpp
  boostfs.cpp
  RLgpuImpl.cu
  geometryTransform.cu
  # camcor.cu
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
  PRIVATE
  Buffer
  ${FFTW3F_LIBRARY}
  ${FFTW3F_THREADS_LIBRARY}
  # ${X11_X11_LIB}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PREDICATE_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${TIFF_LIBRARIES}
  CUDA::cufft
  )

if (UNIX)
  target_link_libraries(cudaDecon PRIVATE rt)
endif ()

target_link_libraries(
  radialft
  ${FFTW3F_LIBRARY}
  ${FFTW3F_THREADS_LIBRARY}
  # ${X11_X11_LIB}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${TIFF_LIBRARIES}
)

target_link_libraries(
  libcudaDecon
  PRIVATE
  Buffer
  ${FFTW3F_LIBRARY}
  ${FFTW3F_THREADS_LIBRARY}
  # ${X11_X11_LIB}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PREDICATE_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${TIFF_LIBRARIES}
  CUDA::cufft
)


target_link_libraries(
  libradialft
  ${FFTW3F_LIBRARY}
  ${TIFF_LIBRARIES}
)

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
