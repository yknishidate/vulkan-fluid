cmake_minimum_required(VERSION 3.10)
project(vk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(external/glfw)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(
    ${PROJECT_NAME} PUBLIC 
    glfw
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    "${PROJECT_SOURCE_DIR}/external/vulkan/include"
    "${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include"
    "${PROJECT_SOURCE_DIR}/external/glfw/include"
    "${PROJECT_SOURCE_DIR}/external/glm"
)

target_compile_definitions(
    ${PROJECT_NAME} PUBLIC 
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
)
