cmake_minimum_required(VERSION 3.18)

project(fastkpm)
set(FASTKPM_VERSION 0.1.0)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


# -- COMPILE OPTIONS -------------------------------------------

# Installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
set(INSTALL_CMAKE_DIR lib/CMake/fastkpm CACHE PATH "Installation directory for CMake files")

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# Performance/debug options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -gdwarf-3")
add_definitions(-D_GLIBCXX_DEBUG=1) # libstdc++
add_definitions(-D_LIBCPP_DEBUG2=1) # libc++

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

# Fast Fourier Transform
find_package(FFTW)
if (FFTW_FOUND)
  include_directories(SYSTEM ${FFTW_INCLUDE_DIRS})
  set(LIBS ${LIBS} ${FFTW_LIBRARIES})
  add_definitions(-DWITH_FFTW)
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
find_package(CUDA 11.3)
if (CUDA_FOUND)
  message(STATUS "Found CUDA " ${CUDA_VERSION})
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  set(LIBS ${LIBS} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})
  add_definitions(-DWITH_CUDA)
  if (APPLE)
    # With recent versions of Xcode, must explicitly use clang
    set(CUDA_HOST_COMPILER /usr/bin/clang)
  endif()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2;-Xcompiler=-fPIC;
    -gencode=arch=compute_52,code=sm_52;
    -gencode=arch=compute_60,code=sm_60;
    -gencode=arch=compute_61,code=sm_61;
    -gencode=arch=compute_70,code=sm_70;
    -gencode=arch=compute_70,code=compute_70)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

# Armadillo
find_package(Armadillo REQUIRED)
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
