#include <iostream>
#include <vulkan/vulkan_raii.hpp>
#include <vk_mem_alloc.h>
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>
#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/ResourceLimits.h>

int main()
{
    const uint32_t WIDTH = 1280;
    const uint32_t HEIGHT = 720;

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

    vk::raii::Context context;

    vk::ApplicationInfo appInfo;
    appInfo.apiVersion = VK_API_VERSION_1_3;

    vk::InstanceCreateInfo instanceCreateInfo;
    instanceCreateInfo.pApplicationInfo = &appInfo;

    vk::raii::Instance instance{ context, instanceCreateInfo };

    glslang::InitializeProcess();
    glslang::FinalizeProcess();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}
