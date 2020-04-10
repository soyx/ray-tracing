CMAKE_MINIMUM_REQUIRED( VERSION 3.5)

project(ray_tracing)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP)
if(OPENMP_FOUND AND CMAKE_SYSTEM_NAME MATCHES "Linux")  
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
        include
)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

add_executable(${PROJECT_NAME} ${SRC_LIST} src/main.cpp include/light/sphereLightSource.h include/light/lightSource.h src/light/sphereLightSource.cpp src/light/quadLishtSource.cpp)

