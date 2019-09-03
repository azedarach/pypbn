#.rst:
# FindCoinUtils
# ----------
#
# Find CoinUtils include dirs and libraries.
#
# This module sets the following variables:
#
# ::
#
#   CoinUtils_FOUND - set to true if the library is found
#   CoinUtils_INCLUDE_DIR - directory containg CoinUtils headers
#   CoinUtils_INCLUDE_DIRS - list of required include directories
#   CoinUtils_LIBRARY - the CoinUtils library
#   CoinUtils_LIBRARIES - list of libraries to be linked
#   CoinUtils_VERSION - version of CoinUtils library
#
# and defines the following imported targets:
#
# ::
#
#   CoinUtils::CoinUtils

function(_coinutils_get_version _coinutils_version_h _found_major _found_minor _found_patch)
  file(READ "${_coinutils_version_h}" _coinutils_version_header)

  string(REGEX MATCH "define[ \t]+COINUTILS_VERSION_MAJOR[ \t]+([0-9]+)" _coinutils_major_version_match "${_coinutils_version_header}")
  set(${_found_major} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  string(REGEX MATCH "define[ \t]+COINUTILS_VERSION_MINOR[ \t]+([0-9]+)" _coinutils_minor_version_match "${_coinutils_version_header}")
  set(${_found_minor} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  string(REGEX MATCH "define[ \t]+COINUTILS_VERSION_RELEASE[ \t]+([0-9]+)" _coinutils_patch_version_match "${_coinutils_version_header}")
  set(${_found_patch} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction(_coinutils_get_version)


if(NOT CoinUtils_INCLUDE_DIR)
  find_path(CoinUtils_INCLUDE_DIR
    NAMES CoinPragma.hpp
    PATH_SUFFIXES
    coinutils/coin
    coin)
endif()

if(CoinUtils_INCLUDE_DIR)
  _coinutils_get_version(
    ${CoinUtils_INCLUDE_DIR}/CoinUtilsConfig.h
    CoinUtils_VERSION_MAJOR
    CoinUtils_VERSION_MINOR
    CoinUtils_VERSION_PATCHLEVEL)
  set(CoinUtils_VERSION ${CoinUtils_VERSION_MAJOR}.${CoinUtils_VERSION_MINOR}.${CoinUtils_VERSION_PATCHLEVEL})
endif()

if(NOT CoinUtils_LIBRARY)
  find_library(CoinUtils_LIBRARY
      NAMES CoinUtils)
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(CoinUtils
    FOUND_VAR CoinUtils_FOUND
    REQUIRED_VARS
    CoinUtils_INCLUDE_DIR
    CoinUtils_LIBRARY
    VERSION_VAR
    CoinUtils_VERSION)

if(CoinUtils_FOUND)
  set(CoinUtils_INCLUDE_DIRS ${CoinUtils_INCLUDE_DIR})
  set(CoinUtils_LIBRARIES ${CoinUtils_LIBRARY})
endif()

if(CoinUtils_FOUND AND NOT TARGET CoinUtils::CoinUtils)
    add_library(CoinUtils::CoinUtils UNKNOWN IMPORTED)
    set_target_properties(CoinUtils::CoinUtils PROPERTIES
        IMPORTED_LOCATION "${CoinUtils_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${CoinUtils_INCLUDE_DIRS}")
endif()

mark_as_advanced(
    CoinUtils_INCLUDE_DIRS
    CoinUtils_LIBRARIES
)
