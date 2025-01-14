cmake_minimum_required(VERSION 3.24)
project(mydlib)

if (NOT CMAKE_BUILD_TYPE)
	message("Setting CMAKE_BUILD_TYPE to Debug")
	set(CMAKE_BUILD_TYPE "Debug")
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)
if(IPO_SUPPORTED)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wextra -ggdb")

find_package(dlib REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(SQLITE3 REQUIRED sqlite3)

add_executable(mydlib embed.cpp embed.h dlib.cpp sqlconn.cpp sqlconn.h sqlite.h)
target_link_libraries(mydlib dlib::dlib ${SQLITE3_LIBRARIES})
