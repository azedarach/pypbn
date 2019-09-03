#.rst:
# FindClp
# ----------
#
# Find Clp include dirs and libraries.
#
# This module sets the following variables:
#
# ::
#
#   Clp_FOUND - set to true if the library is found
#   Clp_INCLUDE_DIR - directory containing Clp headers
#   Clp_INCLUDE_DIRS - list of required include directories
#   Clp_LIBRARY - the Clp library
#   Clp_LIBRARIES - list of libraries to be linked
#   Clp_VERSION - version of Clp library
#
# and defines the following imported targets:
#
# ::
#
#   Clp::Clp

function(_clp_get_version _clp_version_h _found_major _found_minor _found_patch)
  file(READ "${_clp_version_h}" _clp_version_header)

  string(REGEX MATCH "define[ \t]+CLP_VERSION_MAJOR[ \t]+([0-9]+)" _clp_major_version_match "${_clp_version_header}")
  set(${_found_major} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  string(REGEX MATCH "define[ \t]+CLP_VERSION_MINOR[ \t]+([0-9]+)" _clp_minor_version_match "${_clp_version_header}")
  set(${_found_minor} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  string(REGEX MATCH "define[ \t]+CLP_VERSION_RELEASE[ \t]+([0-9]+)" _clp_patch_version_match "${_clp_version_header}")
  set(${_found_patch} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction(_clp_get_version)

if(NOT Clp_INCLUDE_DIR)
  find_path(Clp_INCLUDE_DIR
      NAMES ClpSimplex.hpp
      PATH_SUFFIXES
      clp/coin
      coin)
endif()

if(Clp_INCLUDE_DIR)
    _clp_get_version(
        ${Clp_INCLUDE_DIR}/ClpConfig.h
        Clp_VERSION_MAJOR
        Clp_VERSION_MINOR
        Clp_VERSION_PATCHLEVEL)
    set(Clp_VERSION ${Clp_VERSION_MAJOR}.${Clp_VERSION_MINOR}.${Clp_VERSION_PATCHLEVEL})
endif()

if(NOT Clp_LIBRARY)
  find_library(Clp_LIBRARY
      NAMES Clp)
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Clp
    FOUND_VAR Clp_FOUND
    REQUIRED_VARS
    Clp_INCLUDE_DIR
    Clp_LIBRARY
    VERSION_VAR
    Clp_VERSION)

if(Clp_FOUND)
  set(Clp_INCLUDE_DIRS ${Clp_INCLUDE_DIR})
  set(Clp_LIBRARIES ${Clp_LIBRARY})
endif()

if(Clp_FOUND AND NOT TARGET Clp::Clp)
    add_library(Clp::Clp UNKNOWN IMPORTED)
    set_target_properties(Clp::Clp PROPERTIES
        IMPORTED_LOCATION "${Clp_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${Clp_INCLUDE_DIRS}")
endif()

mark_as_advanced(
    Clp_INCLUDE_DIRS
    Clp_LIBRARIES
)
