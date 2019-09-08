cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)

project(PyPBN VERSION 0.0.1 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

include(PyPBNBuildOptions)

if(ENABLE_COMPILER_WARNINGS)
    include(add_cxx_flag_if_supported)
    include(CompilerWarnings)
endif()

find_package(CoinUtils REQUIRED)
find_package(Clp REQUIRED)
find_package(Eigen3 3.1 REQUIRED)
find_package(Ipopt REQUIRED)

add_subdirectory(pybind11)

set(PyPBN_SOURCE_DIR src)

set(PyPBN_SOURCES
  ${PyPBN_SOURCE_DIR}/clpsimplex_affiliations_solver.cpp
  ${PyPBN_SOURCE_DIR}/fembv_bin_linear.cpp
  ${PyPBN_SOURCE_DIR}/femh1_bin_linear_distributions.cpp
  ${PyPBN_SOURCE_DIR}/femh1_bin_linear_hmc.cpp
  ${PyPBN_SOURCE_DIR}/femh1_bin_linear_mh.cpp
  ${PyPBN_SOURCE_DIR}/densities.cpp
  ${PyPBN_SOURCE_DIR}/linear_constraint.cpp
  ${PyPBN_SOURCE_DIR}/local_linear_model.cpp
  ${PyPBN_SOURCE_DIR}/local_linear_model_ipopt_solver.cpp
  )

pybind11_add_module(pypbn_ext
  ${PyPBN_SOURCES}
  ${PyPBN_SOURCE_DIR}/pypbn.cpp)

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

if(ENABLE_TESTING)
  enable_testing()

  set(PyPBN_TEST_SOURCES
    test/catch_main.cpp
    test/test_femh1_bin_linear_distributions.cpp)

  add_executable(test_pypbn_ext
    ${PyPBN_SOURCES}
    ${PyPBN_TEST_SOURCES})

  target_link_libraries(test_pypbn_ext PUBLIC
    Clp::Clp
    CoinUtils::CoinUtils
    Ipopt::Ipopt)

  target_include_directories(test_pypbn_ext
    PUBLIC ${EIGEN3_INCLUDE_DIR}
    PRIVATE ${PyPBN_SOURCE_DIR} test/catch)

  if(ENABLE_COMPILER_WARNINGS)
    target_compile_options(test_pypbn_ext
      PRIVATE
      ${CXX_WARNING_FLAGS})
  endif()

  set_target_properties(test_pypbn_ext PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS OFF)

  include(Catch)
  catch_discover_tests(test_pypbn_ext)

  add_executable(PyPBN::test_pypbn_ext ALIAS test_pypbn_ext)

  add_custom_target(check COMMAND "ctest")
  add_dependencies(check test_pypbn_ext)
endif()
