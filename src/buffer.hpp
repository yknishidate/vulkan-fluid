#pragma once
#include <vulkan/vulkan_raii.hpp>

struct Buffer
{
    Buffer(const vk::raii::Device& device,
           const vk::raii::PhysicalDevice& physicalDevice,
           const vk::raii::CommandBuffer& commandBuffer,
           const vk::raii::Queue& queue,
           vk::DeviceSize size)
        : size{ size }
        , buffer{ device, makeBufferCreateInfo(size) }
        , memory{ device, makeMemoryAllocationInfo(device, physicalDevice) }
    {
        buffer.bindMemory(*memory, 0);
    }

    vk::BufferCreateInfo makeBufferCreateInfo(vk::DeviceSize size)
    {
        vk::BufferCreateInfo createInfo;
        createInfo.setSize(size);
        createInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
        return createInfo;
    }

    vk::MemoryAllocateInfo makeMemoryAllocationInfo(const vk::raii::Device& device,
                                                    const vk::raii::PhysicalDevice& physicalDevice)
    {
        vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        vk::MemoryRequirements requirements = buffer.getMemoryRequirements();
        uint32_t memoryTypeIndex;
        vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
        for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
            if (requirements.memoryTypeBits & (1 << index)) {
                memoryTypeIndex = index;
            }
        }

        vk::MemoryAllocateInfo memoryAllocateInfo;
        memoryAllocateInfo.setAllocationSize(requirements.size);
        memoryAllocateInfo.setMemoryTypeIndex(memoryTypeIndex);
        return memoryAllocateInfo;
    }

    void copy(void* data)
    {
        if (!mapped) {
            mapped = memory.mapMemory(0, size);
        }
        memcpy(mapped, data, size);
    }

    vk::DeviceSize size;
    vk::raii::Buffer buffer;
    vk::raii::DeviceMemory memory;
    void* mapped = nullptr;
};
