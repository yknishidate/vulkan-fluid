#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <string>
#include <chrono>
#include <iostream>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include "buffer.hpp"
#include "image.hpp"
#include "kernel.hpp"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

const std::string externalForceShader = R"(
#version 460
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0, rgba32f) uniform image2D velocityImage;
layout(binding = 1) uniform UniformBufferObject {
    vec2 mousePosition;
    vec2 mouseMove;
} ubo;
void main()
{
    float mouseSize = 120.0;
    float dist = length(gl_GlobalInvocationID.xy - ubo.mousePosition);
    if(dist < mouseSize && length(ubo.mouseMove) > 4.0){
        vec2 force = ubo.mouseMove * 0.05;
        imageStore(velocityImage, ivec2(gl_GlobalInvocationID.xy), vec4(force, 0, 1));
    }
}
)";

const std::string advectShader = R"(
#version 460
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0) uniform sampler2D inVelocitySampler;
layout(binding = 1, rgba32f) uniform image2D outVelocityImage;

vec2 toUV(vec2 value)
{
    return (value + vec2(0.5)) / (vec2(gl_NumWorkGroups) * vec2(gl_WorkGroupSize));
}

void main()
{
    vec2 uv = toUV(gl_GlobalInvocationID.xy);
    vec2 velocity = texture(inVelocitySampler, uv).xy;
    float dt = 20.0;
    vec2 offset = -velocity * dt;
    velocity = texture(inVelocitySampler, toUV(gl_GlobalInvocationID.xy + offset)).xy;
    imageStore(outVelocityImage, ivec2(gl_GlobalInvocationID.xy), vec4(velocity, 0, 1));
}
)";

const std::string divergenceShader = R"(
#version 460
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0) uniform sampler2D velocitySampler;
layout(binding = 1, rgba32f) uniform image2D divergenceImage;

vec2 toUV(vec2 value)
{
    return (value + vec2(0.5)) / (vec2(gl_NumWorkGroups) * vec2(gl_WorkGroupSize));
}

void main()
{
    float vel_x0 = texture(velocitySampler, toUV(gl_GlobalInvocationID.xy - vec2(1, 0))).x;
    float vel_x1 = texture(velocitySampler, toUV(gl_GlobalInvocationID.xy + vec2(1, 0))).x;
    float vel_y0 = texture(velocitySampler, toUV(gl_GlobalInvocationID.xy - vec2(0, 1))).y;
    float vel_y1 = texture(velocitySampler, toUV(gl_GlobalInvocationID.xy + vec2(0, 1))).y;
    float dx = vel_x1 - vel_x0;
    float dy = vel_y1 - vel_y0;
    float divergence = (dx + dy) / 2.0;
    imageStore(divergenceImage, ivec2(gl_GlobalInvocationID.xy), vec4(divergence));
}
)";

const std::string pressureShader = R"(
#version 460
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0) uniform sampler2D inPressureSampler;
layout(binding = 1) uniform sampler2D divergenceSampler;
layout(binding = 2, rgba32f) uniform image2D outPressureImage;

vec2 toUV(vec2 value)
{
    return (value + vec2(0.5)) / (vec2(gl_NumWorkGroups) * vec2(gl_WorkGroupSize));
}

void main()
{
    vec2 uv = toUV(gl_GlobalInvocationID.xy);
    float pres_x0 = texture(inPressureSampler, toUV(gl_GlobalInvocationID.xy - vec2(1, 0))).x;
    float pres_x1 = texture(inPressureSampler, toUV(gl_GlobalInvocationID.xy + vec2(1, 0))).x;
    float pres_y0 = texture(inPressureSampler, toUV(gl_GlobalInvocationID.xy - vec2(0, 1))).x;
    float pres_y1 = texture(inPressureSampler, toUV(gl_GlobalInvocationID.xy + vec2(0, 1))).x;
    float div = texture(divergenceSampler, uv).x;
    float relaxed = (pres_x0 + pres_x1 + pres_y0 + pres_y1 - div) / 4.0;
    imageStore(outPressureImage, ivec2(gl_GlobalInvocationID.xy), vec4(relaxed));
}
)";

const std::string renderShader = R"(
#version 460
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0, rgba8) uniform image2D renderImage;
layout(binding = 1, rgba32f) uniform image2D inVelocityImage;
layout(binding = 2, rgba32f) uniform image2D divergenceImage;
layout(binding = 3, rgba32f) uniform image2D pressureImage;
layout(binding = 4, rgba32f) uniform image2D outVelocityImage;

vec2 toUV(vec2 value)
{
    return (value + vec2(0.5)) / (vec2(gl_NumWorkGroups) * vec2(gl_WorkGroupSize));
}

void main()
{
    float pres_x0 = imageLoad(pressureImage, ivec2(gl_GlobalInvocationID.xy - vec2(1, 0))).x;
    float pres_x1 = imageLoad(pressureImage, ivec2(gl_GlobalInvocationID.xy + vec2(1, 0))).x;
    float pres_y0 = imageLoad(pressureImage, ivec2(gl_GlobalInvocationID.xy - vec2(0, 1))).y;
    float pres_y1 = imageLoad(pressureImage, ivec2(gl_GlobalInvocationID.xy + vec2(0, 1))).y;
    float dx = (pres_x1 - pres_x0) / 2.0;
    float dy = (pres_y1 - pres_y0) / 2.0;
    vec2 gradient = vec2(dx, dy);

    vec2 velocity = imageLoad(inVelocityImage, ivec2(gl_GlobalInvocationID.xy)).xy;
    velocity = (velocity - gradient) * 0.995;
    imageStore(outVelocityImage, ivec2(gl_GlobalInvocationID.xy), vec4(velocity, 0, 1));
    vec3 color = vec3(abs(velocity.x) * 3, 0, abs(velocity.y) * 3);
    imageStore(renderImage, ivec2(gl_GlobalInvocationID.xy), vec4(color, 1));
}
)";

VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageTypes,
    VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
    void* pUserData)
{
    std::cerr << pCallbackData->pMessage << std::endl;
    assert(false);
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

        // Gather extensions
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        // Gather layers
        std::vector<const char*> layers{ "VK_LAYER_KHRONOS_validation" };

        // Create instance
        vk::DynamicLoader dl;
        auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        vk::ApplicationInfo appInfo;
        appInfo.apiVersion = VK_API_VERSION_1_2;
        vk::InstanceCreateInfo instanceCreateInfo;
        instanceCreateInfo.setPApplicationInfo(&appInfo);
        instanceCreateInfo.setPEnabledExtensionNames(extensions);
        instanceCreateInfo.setPEnabledLayerNames(layers);
        vk::UniqueInstance instance = vk::createInstanceUnique(instanceCreateInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

        // Create debug messenger
        vk::DebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo;
        debugMessengerCreateInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        debugMessengerCreateInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        debugMessengerCreateInfo.setPfnUserCallback(&debugUtilsMessengerCallback);
        vk::UniqueDebugUtilsMessengerEXT debugUtilsMessenger = instance->createDebugUtilsMessengerEXTUnique(debugMessengerCreateInfo);

        // Pick first physical device
        vk::PhysicalDevice physicalDevice = instance->enumeratePhysicalDevices().front();

        // Create surface
        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(VkInstance(*instance), window, nullptr, &_surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        vk::UniqueSurfaceKHR surface{ _surface, {*instance} };

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

        vk::DeviceCreateInfo deviceCreateInfo;
        deviceCreateInfo.setQueueCreateInfos(computeQueueCreateInfo);
        deviceCreateInfo.setPEnabledLayerNames(layers);
        deviceCreateInfo.setPEnabledExtensionNames(requiredExtensions);
        vk::UniqueDevice device = physicalDevice.createDeviceUnique(deviceCreateInfo);

        // Get queue
        vk::Queue computeQueue = device->getQueue(computeFamily, 0);
        vk::Queue presentQueue = device->getQueue(presentFamily, 0);

        // Create command pool
        vk::CommandPoolCreateInfo commandPoolCreateInfo;
        commandPoolCreateInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        commandPoolCreateInfo.setQueueFamilyIndex(computeFamily);
        vk::UniqueCommandPool commandPool = device->createCommandPoolUnique(commandPoolCreateInfo);

        // Create command buffer
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo;
        commandBufferAllocateInfo.setCommandPool(*commandPool);
        commandBufferAllocateInfo.setLevel(vk::CommandBufferLevel::ePrimary);
        commandBufferAllocateInfo.setCommandBufferCount(1);
        std::vector commandBuffers = device->allocateCommandBuffersUnique(commandBufferAllocateInfo);
        vk::UniqueCommandBuffer commandBuffer = std::move(commandBuffers.front());

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
        vk::UniqueSwapchainKHR swapchain = device->createSwapchainKHRUnique(swapchainCreateInfo);
        std::vector swapChainImages = device->getSwapchainImagesKHR(*swapchain);

        // Create resources
        Image renderImage{ *device, physicalDevice, *commandBuffer, computeQueue, width, height, vk::Format::eB8G8R8A8Unorm };
        Image velocityImage0{ *device, physicalDevice, *commandBuffer, computeQueue, width, height };
        Image velocityImage1{ *device, physicalDevice, *commandBuffer, computeQueue, width, height };
        Image divergenceImage{ *device, physicalDevice, *commandBuffer, computeQueue, width, height };
        Image pressureImage0{ *device, physicalDevice, *commandBuffer, computeQueue, width, height };
        Image pressureImage1{ *device, physicalDevice, *commandBuffer, computeQueue, width, height };
        Buffer uniformBuffer{ *device, physicalDevice, *commandBuffer, computeQueue, sizeof(UniformBufferObject) };

        UniformBufferObject ubo;
        ubo.mousePosition[0] = 0.0f;
        ubo.mousePosition[1] = 0.0f;
        ubo.mouseMove[0] = 0.0f;
        ubo.mouseMove[1] = 0.0f;
        uniformBuffer.copy(&ubo);

        // Create descriptor pool
        std::vector poolSizes{
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 20},
            vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 20},
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 20},
        };
        vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo;
        descriptorPoolCreateInfo.setPoolSizes(poolSizes);
        descriptorPoolCreateInfo.setMaxSets(10);
        descriptorPoolCreateInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        vk::UniqueDescriptorPool descPool = device->createDescriptorPoolUnique(descriptorPoolCreateInfo);

        // Create bindings
        std::vector<vk::DescriptorSetLayoutBinding> externalForceKernelBindings{
            {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        };
        std::vector<vk::DescriptorSetLayoutBinding> advectKernelBindings{
            {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        };
        std::vector<vk::DescriptorSetLayoutBinding> divergenceKernelBindings{
            {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        };
        std::vector<vk::DescriptorSetLayoutBinding> pressureKernelBindings{
            {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute},
            {2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        };
        std::vector<vk::DescriptorSetLayoutBinding> renderKernelBindings{
            {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
            {2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
            {3, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
            {4, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        };

        // Create kernels
        ComputeKernel externalForceKernel{ *device, externalForceShader, externalForceKernelBindings, *descPool };
        ComputeKernel advectKernel{ *device, advectShader, advectKernelBindings, *descPool };
        ComputeKernel divergenceKernel{ *device, divergenceShader, divergenceKernelBindings, *descPool };
        ComputeKernel pressureKernel{ *device, pressureShader, pressureKernelBindings, *descPool };
        ComputeKernel renderKernel{ *device, renderShader, renderKernelBindings, *descPool };
        externalForceKernel.updateDescriptorSet(0, 1, velocityImage0);
        externalForceKernel.updateDescriptorSet(1, 1, uniformBuffer);
        advectKernel.updateDescriptorSet(0, 1, velocityImage0, vk::DescriptorType::eCombinedImageSampler);
        advectKernel.updateDescriptorSet(1, 1, velocityImage1);
        divergenceKernel.updateDescriptorSet(0, 1, velocityImage1, vk::DescriptorType::eCombinedImageSampler);
        divergenceKernel.updateDescriptorSet(1, 1, divergenceImage);
        pressureKernel.updateDescriptorSet(0, 1, pressureImage0, vk::DescriptorType::eCombinedImageSampler);
        pressureKernel.updateDescriptorSet(1, 1, divergenceImage, vk::DescriptorType::eCombinedImageSampler);
        pressureKernel.updateDescriptorSet(2, 1, pressureImage1);
        renderKernel.updateDescriptorSet(0, 1, renderImage);
        renderKernel.updateDescriptorSet(1, 1, velocityImage1);
        renderKernel.updateDescriptorSet(2, 1, divergenceImage);
        renderKernel.updateDescriptorSet(3, 1, pressureImage1);
        renderKernel.updateDescriptorSet(4, 1, velocityImage0);

        // Main loop
        uint32_t frame = 0;
        auto startTime = std::chrono::high_resolution_clock::now();
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
            vk::UniqueSemaphore semaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo{});
            uint32_t imageIndex = device->acquireNextImageKHR(*swapchain, UINT64_MAX, *semaphore);
            auto swapChainImage = swapChainImages[imageIndex];

            // Dispatch compute shader
            commandBuffer->begin(vk::CommandBufferBeginInfo{});
            externalForceKernel.run(*commandBuffer, width, height);
            advectKernel.run(*commandBuffer, width, height);
            divergenceKernel.run(*commandBuffer, width, height);

            vk::ImageCopy copyRegion;
            copyRegion.setSrcSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
            copyRegion.setDstSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
            copyRegion.setExtent({ width, height, 1 });

            const int iteration = 16;
            for (int i = 0; i < iteration; i++) {
                pressureKernel.run(*commandBuffer, width, height);

                // Copy render image
                //// pressureImage1 -> pressureImage0
                setImageLayout(*commandBuffer, *pressureImage1.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal);
                setImageLayout(*commandBuffer, *pressureImage0.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferDstOptimal);
                commandBuffer->copyImage(*pressureImage1.image, vk::ImageLayout::eTransferSrcOptimal,
                                         *pressureImage0.image, vk::ImageLayout::eTransferDstOptimal, copyRegion);
                setImageLayout(*commandBuffer, *pressureImage1.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
                setImageLayout(*commandBuffer, *pressureImage0.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral);
            }

            renderKernel.run(*commandBuffer, width, height);

            // Copy render image
            //// render -> swapchain
            setImageLayout(*commandBuffer, *renderImage.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal);
            setImageLayout(*commandBuffer, swapChainImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
            commandBuffer->copyImage(*renderImage.image, vk::ImageLayout::eTransferSrcOptimal,
                                     swapChainImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);
            setImageLayout(*commandBuffer, *renderImage.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
            setImageLayout(*commandBuffer, swapChainImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);

            commandBuffer->end();

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

            frame++;
            if (frame % 100 == 0) {
                auto endTime = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
                std::cout << 100000.0f / elapsed << " fps" << std::endl;
                startTime = std::chrono::high_resolution_clock::now();
            }
        }
        glfwDestroyWindow(window);
        glfwTerminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}
