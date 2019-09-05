cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)

project(PyPBN VERSION 0.0.1 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

include(CMIPFEMBVBINBuildOptions)

if(ENABLE_COMPILER_WARNINGS)
    include(add_cxx_flag_if_supported)
    include(CompilerWarnings)
endif()

find_package(CoinUtils REQUIRED)
find_package(Clp REQUIRED)
find_package(Eigen3 3.1 REQUIRED)
find_package(Ipopt REQUIRED)

add_subdirectory(pybind11)

pybind11_add_module(pypbn_ext
    src/clpsimplex_affiliations_solver.cpp
    src/fembv_bin_linear.cpp
    src/femh1_bin_linear_hmc.cpp
    src/femh1_bin_linear_mh.cpp
    src/linear_constraint.cpp
    src/local_linear_model.cpp
    src/local_linear_model_ipopt_solver.cpp
    src/pypbn.cpp)

target_link_libraries(pypbn_ext PUBLIC
    Clp::Clp
    CoinUtils::CoinUtils
    Ipopt::Ipopt)

target_include_directories(pypbn_ext PUBLIC
    ${EIGEN3_INCLUDE_DIR})

if(ENABLE_COMPILER_WARNINGS)
    target_compile_options(pypbn_ext PRIVATE
        ${CXX_WARNING_FLAGS})
endif()

