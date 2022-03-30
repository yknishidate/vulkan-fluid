#pragma once
#include "common.hpp"

struct Buffer
{
    Buffer(vk::Device device,
           vk::PhysicalDevice physicalDevice,
           vk::DeviceSize size)
        : device{ device }
        , size{ size }
    {
        createBuffer();
        allocateMemory(physicalDevice);
        bindMemory();
    }

    void createBuffer()
    {
        vk::BufferCreateInfo createInfo;
        createInfo.setSize(size);
        createInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
        buffer = device.createBufferUnique(createInfo);
    }

    void allocateMemory(vk::PhysicalDevice physicalDevice)
    {
        vk::MemoryRequirements requirements = device.getBufferMemoryRequirements(*buffer);
        uint32_t memoryTypeIndex = findMemoryTypeIndex(physicalDevice, requirements,
                                                       vk::MemoryPropertyFlagBits::eHostVisible |
                                                       vk::MemoryPropertyFlagBits::eHostCoherent);
        vk::MemoryAllocateInfo memoryAllocateInfo;
        memoryAllocateInfo.setAllocationSize(requirements.size);
        memoryAllocateInfo.setMemoryTypeIndex(memoryTypeIndex);
        memory = device.allocateMemoryUnique(memoryAllocateInfo);
    }

    void bindMemory()
    {
        device.bindBufferMemory(*buffer, *memory, 0);
    }

    void copy(void* data)
    {
        if (!mapped) {
            mapped = device.mapMemory(*memory, 0, size);
        }
        memcpy(mapped, data, size);
    }

    vk::Device device;
    vk::DeviceSize size;
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    void* mapped = nullptr;
};
