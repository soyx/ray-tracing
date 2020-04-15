CMAKE_MINIMUM_REQUIRED( VERSION 3.5)

project(ray_tracing)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 11)

include_directories(
        include
        /usr/include
        /usr/local/include
)
set(ASSIMP_LIB /usr/local/lib/libassimp.dylib)
# add_library(util
#         src/core/transform.cpp
#         src/core/matrix.cpp
# )

# add_library(shape
#         src/shape/sphere.cpp
# )
# target_link_libraries(shape
#         util
# )

# add_library(camera
#         src/core/camera.cpp
# )
# target_link_libraries(camera
#         util
# )

# add_library(scene
#         src/core/scene.cpp
# )
# target_link_libraries(scene
#         util
#         shape
#         camera
# )

add_library(render_library
        src/core/transform.cpp
        src/core/matrix.cpp
        src/shape/sphere.cpp
        src/shape/triangle.cpp
        src/shape/cube.cpp
        src/core/camera.cpp
        src/core/scene.cpp
        src/core/model.cpp
)

target_link_libraries(render_library
        ${ASSIMP_LIB}
)


add_executable(test_sphere
        src/test/test_sphere.cpp        
)

target_link_libraries(test_sphere
        render_library
)

add_executable(test_mesh
        src/test/test_mesh.cpp        
)

target_link_libraries(test_mesh
        render_library
)

add_executable(test_assimp
        src/test/test_assimp.cpp
)

target_link_libraries(test_assimp
        render_library
)