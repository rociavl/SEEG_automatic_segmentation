cmake_minimum_required(VERSION 3.16.3...3.19.7)

project(SEEG)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.example.com/SEEG")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Rocio Avalos (UPC.)")
set(EXTENSION_DESCRIPTION "This extension provides tools for SEEG masking.")
set(EXTENSION_ICONURL "https://www.example.com/SEEG.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/SEEG_screenshot.png")
set(EXTENSION_DEPENDS "NA") # Dependent extensions

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SEEG_masking)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})