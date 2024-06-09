cmake_minimum_required(VERSION 3.0)
project(PerCharTokenizer)

set(CMAKE_C_STANDARD 11)

# Define the shared library
add_library(PerCharTokenizer SHARED PerCharTokenizer.c)

# Set the output directory for the shared library
set_target_properties(PerCharTokenizer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)