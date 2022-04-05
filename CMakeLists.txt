cmake_minimum_required(VERSION 3.10)
project(vulkan_compute LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(external/glfw)
add_subdirectory(external/glslang)

add_executable(${PROJECT_NAME} 
    src/main.cpp 
    src/image.hpp
    src/kernel.hpp
)

target_link_libraries(
    ${PROJECT_NAME} PUBLIC 
    glfw
    glslang
    glslang-default-resource-limits
    SPIRV
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    "${PROJECT_SOURCE_DIR}/external/vulkan/include"
    "${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include"
    "${PROJECT_SOURCE_DIR}/external/glfw/include"
    "${PROJECT_SOURCE_DIR}/external/glm"
    "${PROJECT_SOURCE_DIR}/external/glslang"
)

file(COPY "${CMAKE_SOURCE_DIR}/shader" DESTINATION ${CMAKE_BINARY_DIR})
