#pragma once
#include <vulkan/vulkan.hpp>

inline uint32_t findMemoryTypeIndex(vk::PhysicalDevice physicalDevice,
                                    vk::MemoryRequirements requirements,
                                    vk::MemoryPropertyFlags memoryProp)
{
    uint32_t memoryTypeIndex{ UINT32_MAX };
    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
    for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
        auto propertyFlags = memoryProperties.memoryTypes[index].propertyFlags;
        bool match = (propertyFlags & memoryProp) == memoryProp;
        if (requirements.memoryTypeBits & (1 << index) && match) {
            memoryTypeIndex = index;
        }
    }
    if (memoryTypeIndex == UINT32_MAX) {
        throw std::runtime_error("Failed to find memory type index.");
    }
    return memoryTypeIndex;
}
