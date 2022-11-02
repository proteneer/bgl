cmake_minimum_required(VERSION 2.8.12)
project(bgl_wrapper)


set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3")

set(CMAKE_OSX_ARCHITECTURES "x86_64")
set(PYBIND_SRC_DIR pybind11)
set(BOOST_DIR /Users/yzhao/Code/bgl/boost_1_80_0/)

include_directories(${BOOST_DIR})

if(NOT EXISTS ${PYBIND_SRC_DIR})
  execute_process(COMMAND git clone --branch v2.6 https://github.com/pybind/pybind11.git ${PYBIND_SRC_DIR})
endif()

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/${PYBIND_SRC_DIR})

pybind11_add_module(bgl_wrapper bgl_wrapper.cpp)