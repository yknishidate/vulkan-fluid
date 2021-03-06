#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <string>
#include <iostream>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include "image.hpp"
#include "kernel.hpp"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

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

struct PushConstants
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
        std::vector<const char*> layers{ "VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_monitor" };

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

        // Find queue family
        uint32_t queueFamily{ UINT32_MAX };
        std::vector queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
        for (uint32_t index = 0; index < queueFamilyProperties.size(); index++) {
            auto supportCompute = queueFamilyProperties[index].queueFlags & vk::QueueFlagBits::eCompute;
            auto supportPresent = physicalDevice.getSurfaceSupportKHR(index, *surface);
            if (supportCompute && supportPresent) {
                queueFamily = index;
            }
        }
        if (queueFamily == UINT32_MAX) {
            throw std::runtime_error("Failed to find queue family.");
        }

        // Create device
        const std::vector requiredExtensions{ VK_KHR_SWAPCHAIN_EXTENSION_NAME };

        float queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo;
        queueCreateInfo.setQueueFamilyIndex(queueFamily);
        queueCreateInfo.setQueueCount(1);
        queueCreateInfo.setPQueuePriorities(&queuePriority);

        vk::DeviceCreateInfo deviceCreateInfo;
        deviceCreateInfo.setQueueCreateInfos(queueCreateInfo);
        deviceCreateInfo.setPEnabledExtensionNames(requiredExtensions);
        vk::UniqueDevice device = physicalDevice.createDeviceUnique(deviceCreateInfo);

        // Get queue
        vk::Queue queue = device->getQueue(queueFamily, 0);

        // Create command pool
        vk::CommandPoolCreateInfo commandPoolCreateInfo;
        commandPoolCreateInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        commandPoolCreateInfo.setQueueFamilyIndex(queueFamily);
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
        std::vector swapchainImages = device->getSwapchainImagesKHR(*swapchain);

        // Create resources
        Image renderImage{ *device, physicalDevice, *commandBuffer, queue, width, height, vk::Format::eB8G8R8A8Unorm };
        Image velocityImage0{ *device, physicalDevice, *commandBuffer, queue, width, height };
        Image velocityImage1{ *device, physicalDevice, *commandBuffer, queue, width, height };
        Image divergenceImage{ *device, physicalDevice, *commandBuffer, queue, width, height };
        Image pressureImage0{ *device, physicalDevice, *commandBuffer, queue, width, height };
        Image pressureImage1{ *device, physicalDevice, *commandBuffer, queue, width, height };

        // Create descriptor pool
        std::vector<vk::DescriptorPoolSize> poolSizes{
            {vk::DescriptorType::eStorageImage, 20},
            {vk::DescriptorType::eCombinedImageSampler, 20},
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
        ComputeKernel externalForceKernel{ *device, "shader/externalForce.comp", externalForceKernelBindings, *descPool, sizeof(PushConstants) };
        ComputeKernel advectKernel{ *device, "shader/advect.comp", advectKernelBindings, *descPool };
        ComputeKernel divergenceKernel{ *device, "shader/divergence.comp", divergenceKernelBindings, *descPool };
        ComputeKernel pressureKernel{ *device, "shader/pressure.comp", pressureKernelBindings, *descPool };
        ComputeKernel renderKernel{ *device, "shader/render.comp", renderKernelBindings, *descPool };
        externalForceKernel.updateDescriptorSet(0, 1, velocityImage0);
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
        PushConstants pushConstants{ {0.0f, 0.0f}, {0.0f, 0.0f} };
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // Get mouse input
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            pushConstants.mouseMove[0] = static_cast<float>(xpos) - pushConstants.mousePosition[0];
            pushConstants.mouseMove[1] = static_cast<float>(ypos) - pushConstants.mousePosition[1];
            pushConstants.mousePosition[0] = static_cast<float>(xpos);
            pushConstants.mousePosition[1] = static_cast<float>(ypos);

            // Acquire next image
            vk::UniqueSemaphore semaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo{});
            uint32_t imageIndex = device->acquireNextImageKHR(*swapchain, UINT64_MAX, *semaphore).value;
            vk::Image swapchainImage = swapchainImages[imageIndex];

            // Dispatch compute shader
            commandBuffer->begin(vk::CommandBufferBeginInfo{});
            externalForceKernel.run(*commandBuffer, width, height, &pushConstants);
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
            setImageLayout(*commandBuffer, swapchainImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
            commandBuffer->copyImage(*renderImage.image, vk::ImageLayout::eTransferSrcOptimal,
                                     swapchainImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);
            setImageLayout(*commandBuffer, *renderImage.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
            setImageLayout(*commandBuffer, swapchainImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);

            commandBuffer->end();

            // Submit command buffer
            vk::SubmitInfo submitInfo;
            submitInfo.setCommandBuffers(*commandBuffer);
            queue.submit(submitInfo);
            queue.waitIdle();

            // Present image
            vk::PresentInfoKHR presentInfo;
            presentInfo.setSwapchains(*swapchain);
            presentInfo.setImageIndices(imageIndex);
            if (queue.presentKHR(presentInfo) != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to present.");
            }
        }
        glfwDestroyWindow(window);
        glfwTerminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}
