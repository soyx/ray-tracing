CMAKE_MINIMUM_REQUIRED( VERSION 3.9)

project(ray_tracing)

set(CMAKE_CXX_STANDARD 11)

#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/lib)
find_package(OpenCV REQUIRED)

include_directories(
        include
#        ${OPENCV_DIRS}
)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

add_executable(${PROJECT_NAME} ${SRC_LIST} src/main.cpp include/shape.h  include/light/sphereLightSource.h include/light/lightSource.h src/light/sphereLightSource.cpp)

#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})