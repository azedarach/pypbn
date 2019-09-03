#.rst:
# FindIpopt
# ----------
#
# Find Ipopt include dirs and libraries.
#
# This module sets the following variables:
#
# ::
#
#   Ipopt_FOUND - set to true if the library is found
#   Ipopt_INCLUDE_DIR - directory containing Ipopt headers
#   Ipopt_INCLUDE_DIRS - list of required include directories
#   Ipopt_LIBRARY - the Ipopt library
#   Ipopt_LIBRARIES - list of libraries to be linked
#   Ipopt_VERSION - version of Ipopt library
#
# and defines the following imported targets:
#
# ::
#
#   Ipopt::Ipopt

function(_ipopt_get_version _ipopt_version_h _found_major _found_minor _found_patch)
  file(READ "${_ipopt_version_h}" _ipopt_version_header)

  string(REGEX MATCH "define[ \t]+IPOPT_VERSION_MAJOR[ \t]+([0-9]+)" _ipopt_major_version_match "${_ipopt_version_header}")
  set(${_found_major} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  string(REGEX MATCH "define[ \t]+IPOPT_VERSION_MINOR[ \t]+([0-9]+)" _ipopt_minor_version_match "${_ipopt_version_header}")
  set(${_found_minor} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  string(REGEX MATCH "define[ \t]+IPOPT_VERSION_RELEASE[ \t]+([0-9]+)" _ipopt_patch_version_match "${_ipopt_version_header}")
  set(${_found_patch} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction(_ipopt_get_version)

if(NOT Ipopt_INCLUDE_DIR)
  find_path(Ipopt_INCLUDE_DIR
      NAMES IpIpoptApplication.hpp
      PATH_SUFFIXES
      ipopt/coin
      coin)
endif()

if(Ipopt_INCLUDE_DIR)
  find_file(Ipopt_VERSION_FILE
    NAMES
    config_ipopt_default.h
    IpoptConfig.h
    PATHS
    ${Ipopt_INCLUDE_DIR}
    )

    if(NOT Ipopt_VERSION_FILE STREQUAL "Ipopt_VERSION_FILE-NOTFOUND")
      _ipopt_get_version(
          ${Ipopt_VERSION_FILE}
          Ipopt_VERSION_MAJOR
          Ipopt_VERSION_MINOR
          Ipopt_VERSION_PATCHLEVEL)
      set(Ipopt_VERSION ${Ipopt_VERSION_MAJOR}.${Ipopt_VERSION_MINOR}.${Ipopt_VERSION_PATCHLEVEL})
    endif()
endif()

if(NOT Ipopt_LIBRARY)
  find_library(Ipopt_LIBRARY
      NAMES ipopt)
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Ipopt
    FOUND_VAR Ipopt_FOUND
    REQUIRED_VARS
    Ipopt_INCLUDE_DIR
    Ipopt_LIBRARY
    VERSION_VAR
    Ipopt_VERSION)

if(Ipopt_FOUND)
  set(Ipopt_INCLUDE_DIRS ${Ipopt_INCLUDE_DIR})
  set(Ipopt_LIBRARIES ${Ipopt_LIBRARY})
endif()

if(Ipopt_FOUND AND NOT TARGET Ipopt::Ipopt)
    add_library(Ipopt::Ipopt UNKNOWN IMPORTED)
    set_target_properties(Ipopt::Ipopt PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "HAVE_CSTDDEF"
        IMPORTED_LOCATION "${Ipopt_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Ipopt_INCLUDE_DIRS}")
endif()

mark_as_advanced(
    Ipopt_INCLUDE_DIRS
    Ipopt_LIBRARIES
)
