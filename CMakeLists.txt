CMAKE_MINIMUM_REQUIRED( VERSION 3.5)

project(ray_tracing)

set(CMAKE_CXX_STANDARD 11)

include_directories(
        include
)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

add_executable(${PROJECT_NAME} ${SRC_LIST} src/main.cpp include/shape.h  include/light/sphereLightSource.h include/light/lightSource.h src/light/sphereLightSource.cpp src/light/quadLishtSource.cpp)

