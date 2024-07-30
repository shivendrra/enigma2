cmake_minimum_required(VERSION 3.12)
project(kmer_c)

set(pybind11_DIR "C:/Users/shivh/AppData/Roaming/Python/Python311/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

set(CMAKE_C_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.XX.XXXXX/bin/Hostx64/x64/cl.exe")
set(CMAKE_CXX_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.XX.XXXXX/bin/Hostx64/x64/cl.exe")

add_library(kmer_c MODULE tokenizer/csrc/kmer.cpp tokenizer/csrc/bindings.cpp)
target_link_libraries(kmer_c PRIVATE pybind11::module)

set_target_properties(kmer_c PROPERTIES PREFIX "" SUFFIX ".pyd")