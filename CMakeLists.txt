cmake_minimum_required(VERSION 3.10)
project(vk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(external/glfw)
add_subdirectory(external/glslang)
add_subdirectory(external/SPIRV-Cross)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(
    ${PROJECT_NAME} PUBLIC 
    glfw
    glslang
    glslang-default-resource-limits
    SPIRV
    spirv-cross-glsl
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    "${PROJECT_SOURCE_DIR}/external/vulkan/include"
    "${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include"
    "${PROJECT_SOURCE_DIR}/external/glfw/include"
    "${PROJECT_SOURCE_DIR}/external/glm"
    "${PROJECT_SOURCE_DIR}/external/tinyobjloader"
    "${PROJECT_SOURCE_DIR}/external/tinygltf"
    "${PROJECT_SOURCE_DIR}/external/glslang"
    "${PROJECT_SOURCE_DIR}/external/SPIRV-Cross"
)
