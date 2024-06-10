cmake_minimum_required(VERSION 3.5)
project(DNATokenizer)

set(CMAKE_C_STANDARD 11)

# Define the shared library
add_library(DNATokenizer SHARED ../tokenizer/tokenizer.c)

# Set the output directory for the shared library
set_target_properties(DNATokenizer PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ../tokenizer/lib
  RUNTIME_OUTPUT_DIRECTORY ../tokenizer/lib
)