cmake_minimum_required(VERSION 3.20)
project(main_server CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(httplib INTERFACE)

target_include_directories(httplib INTERFACE ${CMAKE_SOURCE_DIR}/../third_party)

find_package(OpenCV REQUIRED)

add_executable(main_server server.cpp)

#target_compile_definitions(main_server PRIVATE CPPHTTPLIB_FROM_DATA_SUPPORT)

target_link_libraries(main_server PRIVATE ${OpenCV_LIBS} httplib)