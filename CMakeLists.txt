cmake_minimum_required(VERSION 2.8)

project(FastKPM)
set(FASTKPM_VERSION 0.1.0)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


# -- OPTIONS -------------------------------------------

# MPI SUPPORT
set(WITH_MPI no CACHE BOOL "Build with MPI support")

# Installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
set(INSTALL_CMAKE_DIR lib/CMake/FastKPM CACHE PATH "Installation directory for CMake files")

# Performance/debug options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
add_definitions(-D_GLIBCXX_DEBUG=1) # libstdc++
add_definitions(-D_LIBCPP_DEBUG2=1) # libc++


# -- C++11 -------------------------------------------

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

# Armadillo
find_package(Armadillo 4.400.2 REQUIRED CONFIG
             PATHS "/usr/local" "$ENV{ARMA_ROOT}" NO_SYSTEM_ENVIRONMENT_PATH)
include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})
set(LIBS ${LIBS} ${ARMADILLO_LIBRARIES})

# Thread Building Blocks
find_package(TBB)
if (TBB_FOUND)
    include_directories(SYSTEM ${TBB_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${TBB_LIBRARIES})
    add_definitions(-DWITH_TBB)
endif()

# CUDA
find_package(CUDA 6.0)
if (CUDA_FOUND)
    message(STATUS "Found CUDA " ${CUDA_VERSION})
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})
    add_definitions(-DWITH_CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

# Boost
find_package(Boost 1.55.0 REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})


if (WITH_MPI)
    # MPI
    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
    set(LIBS ${LIBS} ${MPI_LIBRARIES})
    set(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_LINK_FLAGS}")
    # BOOST MPI
    set(Boost_USE_MULTITHREADED ON)
    find_package(Boost REQUIRED COMPONENTS MPI)
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${Boost_MPI_LIBRARY})
endif()


# -- BUILD -------------------------------------------

# include_directories(src)
add_subdirectory(src)


# -- INSTALLATION -------------------------------------------

# Generate FastKPMTargets.cmake suitable for use by downstreams 
export(TARGETS fastkpm FILE "${PROJECT_BINARY_DIR}/FastKPMTargets.cmake")

# Register the build-tree with a global CMake-registry
export(PACKAGE FastKPM)

# Create the FastKPMConfig.cmake and FastKPMConfigVersion files
# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src")
configure_file(FastKPMConfig.cmake.in "${PROJECT_BINARY_DIR}/FastKPMConfig.cmake" @ONLY)
# ... for the install tree
set(CONF_INCLUDE_DIRS "${INSTALL_INCLUDE_DIR}")
configure_file(FastKPMConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FastKPMConfig.cmake" @ONLY)
# ... for both
configure_file(FastKPMConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/FastKPMConfigVersion.cmake" @ONLY)

# Install FastKPMConfig.cmake and FastKPMConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FastKPMConfig.cmake"
  "${PROJECT_BINARY_DIR}/FastKPMConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
 
# Install FastKPMTargets.cmake
install(EXPORT FastKPMTargets DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)


# -- REPORT -------------------------------------------

message("")
message("Configuration results")
message("---------------------")
message("C++ compiler        : ${CMAKE_CXX_COMPILER}")
message("C++ compiler flags  : ${CMAKE_CXX_FLAGS}")

get_property(INC_DIRS DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("LIBS:     ${LIBS}")
message("INCLUDES: ${INC_DIRS}")
message("WITH_MPI: ${WITH_MPI}")
message("")
