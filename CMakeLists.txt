CMAKE_MINIMUM_REQUIRED( VERSION 3.5)

project(ray_tracing)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)

include_directories(
        include
        ${CUDA_INCLUDE_DIRS}
)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

#add_executable(${PROJECT_NAME} ${SRC_LIST})
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

