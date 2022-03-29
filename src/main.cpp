#include <string>
#include <iostream>
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/ResourceLimits.h>
#include "buffer.hpp"
#include "image.hpp"
#include "kernel.hpp"

const std::string externalForceShader = R"(
#version 460
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0, rgba8) uniform image2D velocityImage;
layout(binding = 1) uniform UniformBufferObject {
    vec2 mousePosition;
    vec2 mouseMove;
} ubo;
void main()
{
    float mouseSize = 50.0;
    float dist = length(gl_GlobalInvocationID.xy - ubo.mousePosition);
    if(dist < mouseSize){
        imageStore(velocityImage, ivec2(gl_GlobalInvocationID.xy), vec4(1));
    }
}
)";

const std::string renderShader = R"(
#version 460
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0, rgba8) uniform image2D renderImage;
layout(binding = 1, rgba8) uniform image2D velocityImage;
void main()
{
    vec4 velocity = imageLoad(velocityImage, ivec2(gl_GlobalInvocationID.xy));
    imageStore(renderImage, ivec2(gl_GlobalInvocationID.xy), velocity);
}
)";

VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageTypes,
    VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
    void* pUserData)
{
    std::cerr << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

struct UniformBufferObject
{
    float mousePosition[2];
    float mouseMove[2];
};

int main()
{
    try {
        // Create window
        const int width = 1280;
        const int height = 720;
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        GLFWwindow* window = glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr);

        // Create context
        vk::raii::Context context;

        // Gather extensions
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        // Gather layers
        std::vector<const char*> layers{ "VK_LAYER_KHRONOS_validation" };

        // Create instance
        vk::ApplicationInfo appInfo;
        appInfo.apiVersion = VK_API_VERSION_1_2;
        vk::InstanceCreateInfo instanceCreateInfo;
        instanceCreateInfo.setPApplicationInfo(&appInfo);
        instanceCreateInfo.setPEnabledExtensionNames(extensions);
        instanceCreateInfo.setPEnabledLayerNames(layers);
        vk::raii::Instance instance{ context, instanceCreateInfo };

        // Create debug messenger
        vk::DebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo;
        debugMessengerCreateInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        debugMessengerCreateInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        debugMessengerCreateInfo.setPfnUserCallback(&debugUtilsMessengerCallback);
        vk::raii::DebugUtilsMessengerEXT debugUtilsMessenger{ instance, debugMessengerCreateInfo };

        // Pick first physical device
        vk::raii::PhysicalDevice physicalDevice = vk::raii::PhysicalDevices{ instance }.front();

        // Create surface
        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(VkInstance(*instance), window, nullptr, &_surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        vk::raii::SurfaceKHR surface{ instance, _surface };

        // Find queue families
        uint32_t computeFamily;
        uint32_t presentFamily;
        std::vector queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
        for (uint32_t index = 0; index < queueFamilyProperties.size(); index++) {
            if (queueFamilyProperties[index].queueFlags & vk::QueueFlagBits::eCompute) {
                computeFamily = index;
            }
            if (physicalDevice.getSurfaceSupportKHR(index, *surface)) {
                presentFamily = index;
            }
        }

        // Create device
        const std::vector requiredExtensions{
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };

        float queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo computeQueueCreateInfo;
        computeQueueCreateInfo.setQueueFamilyIndex(computeFamily);
        computeQueueCreateInfo.setQueueCount(1);
        computeQueueCreateInfo.setPQueuePriorities(&queuePriority);

        vk::DeviceQueueCreateInfo presentQueueCreateInfo;
        presentQueueCreateInfo.setQueueFamilyIndex(presentFamily);
        presentQueueCreateInfo.setQueueCount(1);
        presentQueueCreateInfo.setPQueuePriorities(&queuePriority);

        std::vector deviceQueueCreateInfos{
            computeQueueCreateInfo,
            presentQueueCreateInfo
        };

        vk::DeviceCreateInfo deviceCreateInfo;
        deviceCreateInfo.setQueueCreateInfos(computeQueueCreateInfo);
        deviceCreateInfo.setPEnabledLayerNames(layers);
        deviceCreateInfo.setPEnabledExtensionNames(requiredExtensions);
        vk::raii::Device device{ physicalDevice, deviceCreateInfo };

        // Get queue
        vk::raii::Queue computeQueue{ device, computeFamily, 0 };
        vk::raii::Queue presentQueue{ device, presentFamily, 0 };

        // Create command pool
        vk::CommandPoolCreateInfo commandPoolCreateInfo;
        commandPoolCreateInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        commandPoolCreateInfo.setQueueFamilyIndex(computeFamily);
        vk::raii::CommandPool commandPool{ device, commandPoolCreateInfo };

        // Create command buffer
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo{ *commandPool, vk::CommandBufferLevel::ePrimary, 1 };
        vk::raii::CommandBuffers commandBuffers{ device, commandBufferAllocateInfo };
        vk::raii::CommandBuffer commandBuffer = std::move(commandBuffers.front());

        // Create swapchain
        vk::SwapchainCreateInfoKHR swapchainCreateInfo;
        swapchainCreateInfo.setSurface(*surface);
        swapchainCreateInfo.setMinImageCount(3);
        swapchainCreateInfo.setImageFormat(vk::Format::eB8G8R8A8Unorm);
        swapchainCreateInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
        swapchainCreateInfo.setImageExtent({ width, height });
        swapchainCreateInfo.setImageArrayLayers(1);
        swapchainCreateInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst);
        swapchainCreateInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
        swapchainCreateInfo.setPresentMode(vk::PresentModeKHR::eFifo);
        swapchainCreateInfo.setClipped(true);
        vk::raii::SwapchainKHR swapchain{ device, swapchainCreateInfo };
        std::vector swapChainImages = swapchain.getImages();

        // Create resources
        Image renderImage{ device, physicalDevice, commandBuffer, computeQueue, width, height };
        Image velocityImage{ device, physicalDevice, commandBuffer, computeQueue, width, height };
        Buffer uniformBuffer{ device, physicalDevice, commandBuffer, computeQueue, sizeof(UniformBufferObject) };

        UniformBufferObject ubo;
        ubo.mousePosition[0] = 0.0f;
        ubo.mousePosition[1] = 0.0f;
        ubo.mouseMove[0] = 0.0f;
        ubo.mouseMove[1] = 0.0f;
        uniformBuffer.copy(&ubo);

        // Create descriptor pool
        std::vector poolSizes{
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 10},
            vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 10}
        };
        vk::DescriptorPoolCreateInfo descPoolCreateInfo;
        descPoolCreateInfo.setPoolSizes(poolSizes);
        descPoolCreateInfo.setMaxSets(10);
        descPoolCreateInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        vk::raii::DescriptorPool descPool{ device, descPoolCreateInfo };

        // Create compute pipeline
        std::vector<vk::DescriptorSetLayoutBinding> renderKernelBindings{
            {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}
        };
        std::vector<vk::DescriptorSetLayoutBinding> externalForceKernelBindings{
            {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        };

        // Create kernels
        ComputeKernel renderKernel{ device, renderShader, renderKernelBindings, descPool };
        ComputeKernel externalForceKernel{ device, externalForceShader, externalForceKernelBindings, descPool };
        renderKernel.updateDescriptorSet(device, 0, 1, renderImage);
        renderKernel.updateDescriptorSet(device, 1, 1, velocityImage);
        externalForceKernel.updateDescriptorSet(device, 0, 1, velocityImage);
        externalForceKernel.updateDescriptorSet(device, 1, 1, uniformBuffer);

        // Main loop
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // Get mouse input
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            ubo.mouseMove[0] = xpos - ubo.mousePosition[0];
            ubo.mouseMove[1] = ypos - ubo.mousePosition[1];
            ubo.mousePosition[0] = xpos;
            ubo.mousePosition[1] = ypos;
            uniformBuffer.copy(&ubo);

            // Acquire next image
            vk::raii::Semaphore semaphore{ device, vk::SemaphoreCreateInfo {} };
            auto [result, imageIndex] = swapchain.acquireNextImage(UINT64_MAX, *semaphore);
            auto swapChainImage = swapChainImages[imageIndex];

            // Dispatch compute shader
            commandBuffer.reset();
            commandBuffer.begin(vk::CommandBufferBeginInfo{});
            externalForceKernel.run(commandBuffer, width, height);
            renderKernel.run(commandBuffer, width, height);

            // Copy render image
            setImageLayout(commandBuffer, *renderImage.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal);
            setImageLayout(commandBuffer, swapChainImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

            vk::ImageCopy copyRegion;
            copyRegion.setSrcSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
            copyRegion.setDstSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
            copyRegion.setExtent({ width, height, 1 });
            commandBuffer.copyImage(*renderImage.image, vk::ImageLayout::eTransferSrcOptimal,
                                    swapChainImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);

            setImageLayout(commandBuffer, *renderImage.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
            setImageLayout(commandBuffer, swapChainImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);

            commandBuffer.end();

            // Submit command buffer
            vk::SubmitInfo submitInfo;
            submitInfo.setCommandBuffers(*commandBuffer);
            computeQueue.submit(submitInfo);
            computeQueue.waitIdle();

            // Present image
            vk::PresentInfoKHR presentInfo;
            presentInfo.setSwapchains(*swapchain);
            presentInfo.setImageIndices(imageIndex);
            presentQueue.presentKHR(presentInfo);
            presentQueue.waitIdle();
        }
        glfwDestroyWindow(window);
        glfwTerminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}
