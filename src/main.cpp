#include <string>
#include <iostream>
#include <chrono>
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
layout(binding = 0, rgba32f) uniform image2D velocityImage;
layout(binding = 1) uniform UniformBufferObject {
    vec2 mousePosition;
    vec2 mouseMove;
} ubo;
void main()
{
    float mouseSize = 120.0;
    float dist = length(gl_GlobalInvocationID.xy - ubo.mousePosition);
    if(dist < mouseSize && length(ubo.mouseMove) > 10.0){
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
    velocity = (velocity - gradient) * 0.99;
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
        Image renderImage{ device, physicalDevice, commandBuffer, computeQueue, width, height, vk::Format::eB8G8R8A8Unorm };
        Image velocityImage0{ device, physicalDevice, commandBuffer, computeQueue, width, height };
        Image velocityImage1{ device, physicalDevice, commandBuffer, computeQueue, width, height };
        Image divergenceImage{ device, physicalDevice, commandBuffer, computeQueue, width, height };
        Image pressureImage0{ device, physicalDevice, commandBuffer, computeQueue, width, height };
        Image pressureImage1{ device, physicalDevice, commandBuffer, computeQueue, width, height };
        Buffer uniformBuffer{ device, physicalDevice, commandBuffer, computeQueue, sizeof(UniformBufferObject) };

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
        vk::DescriptorPoolCreateInfo descPoolCreateInfo;
        descPoolCreateInfo.setPoolSizes(poolSizes);
        descPoolCreateInfo.setMaxSets(10);
        descPoolCreateInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        vk::raii::DescriptorPool descPool{ device, descPoolCreateInfo };

        // Create compute pipeline
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
        ComputeKernel externalForceKernel{ device, externalForceShader, externalForceKernelBindings, descPool };
        ComputeKernel advectKernel{ device, advectShader, advectKernelBindings, descPool };
        ComputeKernel divergenceKernel{ device, divergenceShader, divergenceKernelBindings, descPool };
        ComputeKernel pressureKernel{ device, pressureShader, pressureKernelBindings, descPool };
        ComputeKernel renderKernel{ device, renderShader, renderKernelBindings, descPool };
        externalForceKernel.updateDescriptorSet(device, 0, 1, velocityImage0);
        externalForceKernel.updateDescriptorSet(device, 1, 1, uniformBuffer);
        advectKernel.updateDescriptorSet(device, 0, 1, velocityImage0, vk::DescriptorType::eCombinedImageSampler);
        advectKernel.updateDescriptorSet(device, 1, 1, velocityImage1);
        divergenceKernel.updateDescriptorSet(device, 0, 1, velocityImage1, vk::DescriptorType::eCombinedImageSampler);
        divergenceKernel.updateDescriptorSet(device, 1, 1, divergenceImage);
        pressureKernel.updateDescriptorSet(device, 0, 1, pressureImage0, vk::DescriptorType::eCombinedImageSampler);
        pressureKernel.updateDescriptorSet(device, 1, 1, divergenceImage, vk::DescriptorType::eCombinedImageSampler);
        pressureKernel.updateDescriptorSet(device, 2, 1, pressureImage1);
        renderKernel.updateDescriptorSet(device, 0, 1, renderImage);
        renderKernel.updateDescriptorSet(device, 1, 1, velocityImage1);
        renderKernel.updateDescriptorSet(device, 2, 1, divergenceImage);
        renderKernel.updateDescriptorSet(device, 3, 1, pressureImage1);
        renderKernel.updateDescriptorSet(device, 4, 1, velocityImage0);

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
            vk::raii::Semaphore semaphore{ device, vk::SemaphoreCreateInfo {} };
            auto [result, imageIndex] = swapchain.acquireNextImage(UINT64_MAX, *semaphore);
            auto swapChainImage = swapChainImages[imageIndex];

            // Dispatch compute shader
            commandBuffer.reset();
            commandBuffer.begin(vk::CommandBufferBeginInfo{});
            externalForceKernel.run(commandBuffer, width, height);
            advectKernel.run(commandBuffer, width, height);
            divergenceKernel.run(commandBuffer, width, height);

            vk::ImageCopy copyRegion;
            copyRegion.setSrcSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
            copyRegion.setDstSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
            copyRegion.setExtent({ width, height, 1 });

            const int iteration = 16;
            for (int i = 0; i < iteration; i++) {
                pressureKernel.run(commandBuffer, width, height);

                setImageLayout(commandBuffer, *pressureImage1.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal);
                setImageLayout(commandBuffer, *pressureImage0.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferDstOptimal);
                commandBuffer.copyImage(*pressureImage1.image, vk::ImageLayout::eTransferSrcOptimal,
                                        *pressureImage0.image, vk::ImageLayout::eTransferDstOptimal, copyRegion);
                setImageLayout(commandBuffer, *pressureImage1.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
                setImageLayout(commandBuffer, *pressureImage0.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral);
            }

            renderKernel.run(commandBuffer, width, height);

            // Copy render image
            //// render -> swapchain
            //// velocity1 -> velocity0
            setImageLayout(commandBuffer, *renderImage.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal);
            setImageLayout(commandBuffer, swapChainImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
            setImageLayout(commandBuffer, *velocityImage1.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal);
            setImageLayout(commandBuffer, *velocityImage0.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

            commandBuffer.copyImage(*renderImage.image, vk::ImageLayout::eTransferSrcOptimal,
                                    swapChainImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);

            setImageLayout(commandBuffer, *renderImage.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
            setImageLayout(commandBuffer, swapChainImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);
            setImageLayout(commandBuffer, *velocityImage0.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral);
            setImageLayout(commandBuffer, *velocityImage1.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);

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
