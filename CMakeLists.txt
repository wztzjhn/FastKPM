cmake_minimum_required(VERSION 3.18)

project(fastkpm VERSION 0.1.0)
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
option(WITH_CUDA "Enable CUDA support" ON)
if(WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit 11.3)
        if(CUDAToolkit_FOUND)
            message(STATUS "Found CUDA " ${CUDAToolkit_VERSION})
            add_definitions(-DWITH_CUDA)
        else()
            message(STATUS "CUDA Toolkit not found, building without CUDA support") 
            set(WITH_CUDA OFF)
        endif()
    else()
        message(STATUS "No CUDA compiler found, building without CUDA support")
        set(WITH_CUDA OFF)
    endif()
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

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Generate and install export targets
install(EXPORT fastkpmTargets
    FILE fastkpmTargets.cmake
    NAMESPACE fastkpm::
    DESTINATION lib/cmake/fastkpm
)

# Generate the version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/fastkpmConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Configure the config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/fastkpmConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/fastkpmConfig.cmake"
    INSTALL_DESTINATION lib/cmake/fastkpm
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

# Install the config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/fastkpmConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/fastkpmConfigVersion.cmake"
    DESTINATION lib/cmake/fastkpm
)

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
