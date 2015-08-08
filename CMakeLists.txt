cmake_minimum_required(VERSION 2.8)

project(fastkpm)
set(FASTKPM_VERSION 0.1.0)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


# -- COMPILE OPTIONS -------------------------------------------

# Installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
set(INSTALL_CMAKE_DIR lib/CMake/fastkpm CACHE PATH "Installation directory for CMake files")

# Performance/debug options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -gdwarf-3")
add_definitions(-D_GLIBCXX_DEBUG=1) # libstdc++
add_definitions(-D_LIBCPP_DEBUG2=1) # libc++

# C++11
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(NOT COMPILER_SUPPORTS_CXX11)
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Aborting.")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# Mac fixes
if (APPLE)
  cmake_policy(SET CMP0042 NEW) # @rpath
  set(LIBS ${LIBS} "-framework Accelerate")
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
endif()


# -- DEPENDENCIES -------------------------------------------

# Thread Building Blocks
find_package(TBB)
if (TBB_FOUND)
  include_directories(SYSTEM ${TBB_INCLUDE_DIRS})
  set(LIBS ${LIBS} ${TBB_LIBRARIES})
  add_definitions(-DWITH_TBB)
endif()

# MPI
find_package(MPI)
if (MPI_FOUND)
  include_directories(${MPI_INCLUDE_PATH})
  set(LIBS ${LIBS} ${MPI_LIBRARIES})
  set(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_LINK_FLAGS}")
  add_definitions(-DWITH_MPI)
endif()

# CUDA
find_package(CUDA 6.5)
if (CUDA_FOUND)
  message(STATUS "Found CUDA " ${CUDA_VERSION})
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  set(LIBS ${LIBS} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})
  add_definitions(-DWITH_CUDA)
  if (APPLE)
    # With recent versions of Xcode, must explicitly use clang
    set(CUDA_HOST_COMPILER /usr/bin/clang)
  endif()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2;
    -gencode=arch=compute_20,code=sm_20;
    -gencode=arch=compute_30,code=sm_30;
    -gencode=arch=compute_35,code=sm_35;
    -gencode=arch=compute_50,code=sm_50)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

# Armadillo
find_package(Armadillo 4.400.2 REQUIRED CONFIG
             PATHS "/usr/local" "$ENV{ARMA_ROOT}" NO_SYSTEM_ENVIRONMENT_PATH)
include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})
set(LIBS ${LIBS} ${ARMADILLO_LIBRARIES})

# Boost
find_package(Boost 1.55.0 REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})


# -- BUILD -------------------------------------------

add_subdirectory(src)


# -- INSTALLATION -------------------------------------------

# Generate fastkpmTargets.cmake suitable for use by downstreams
export(TARGETS fastkpm FILE "${PROJECT_BINARY_DIR}/fastkpmTargets.cmake")

# Register the build-tree with a global CMake-registry
export(PACKAGE fastkpm)

# Create the fastkpmConfig.cmake and fastkpmConfigVersion files
# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src")
configure_file(fastkpmConfig.cmake.in "${PROJECT_BINARY_DIR}/fastkpmConfig.cmake" @ONLY)
# ... for the install tree
set(CONF_INCLUDE_DIRS "${INSTALL_INCLUDE_DIR}")
configure_file(fastkpmConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/fastkpmConfig.cmake" @ONLY)
# ... for both
configure_file(fastkpmConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/fastkpmConfigVersion.cmake" @ONLY)

# Install fastkpmConfig.cmake and fastkpmConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/fastkpmConfig.cmake"
  "${PROJECT_BINARY_DIR}/fastkpmConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
 
# Install fastkpmTargets.cmake
install(EXPORT fastkpmTargets DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)


# -- REPORT -------------------------------------------

message("")
message("Configuration results")
message("---------------------")
message("C++ compiler        : ${CMAKE_CXX_COMPILER}")
message("C++ compiler flags  : ${CMAKE_CXX_FLAGS}")

get_property(INC_DIRS DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("LIBS:     ${LIBS}")
message("INCLUDES: ${INC_DIRS}")
message("WITH_MPI: ${MPI_FOUND}")
message("")
