include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  ${CUDA_INCLUDE_DIRS}
  "${CMAKE_SOURCE_DIR}/gtest/include"
  )

CUDA_ADD_LIBRARY(
  Buffer
  Buffer.cpp
  bufferExample.cpp
  CPUBuffer.cpp
  GPUBuffer.cpp
  PinnedCPUBuffer.cpp
  )

set(HEADERS
  SIMrecon_TechX/Buffers/Buffer.h
  SIMrecon_TechX/Buffers/CPUBuffer.h
  SIMrecon_TechX/Buffers/GPUBuffer.h
  SIMrecon_TechX/Buffers/PinnedCPUBuffer.h
  )

install(
  TARGETS Buffer
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )

set(TESTS
  test_CPUBuffer
  test_GPUBuffer
  )

set(LIBRARIES
  Buffer
  gtest_main
  gtest
  )

#foreach(t ${TESTS})
#  CUDA_ADD_EXECUTABLE(${t} ${t}.cpp)
#  add_dependencies(${t} ${LIBRARIES})
#  target_link_libraries(${t} ${LIBRARIES})
#  add_test(${t} ${t})
#endforeach(t)

install(
  FILES ${HEADERS} DESTINATION include)

#add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
#      DEPENDS ${TESTS})
