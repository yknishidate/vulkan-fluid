#include <iostream>
#include <vulkan/vulkan_raii.hpp>
#include <vk_mem_alloc.h>
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <tiny_obj_loader.h>
#include <tiny_gltf.h>
#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/ResourceLimits.h>
#include <spirv_glsl.hpp>

// sample code of Vulkan-Headers
void createInstance()
{
    vk::raii::Context context;

    vk::ApplicationInfo appInfo;
    appInfo.apiVersion = VK_API_VERSION_1_3;

    vk::InstanceCreateInfo instanceCreateInfo;
    instanceCreateInfo.pApplicationInfo = &appInfo;

    vk::raii::Instance instance{ context, instanceCreateInfo };
}

// sample code of GLFW
void runWindow()
{
    const uint32_t WIDTH = 1280;
    const uint32_t HEIGHT = 720;

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

// sample code of glslang
std::vector<unsigned int> compileToSPV(EShLanguage stage,
                                       const std::string& glslShader)
{
    glslang::InitializeProcess();

    const char* shaderStrings[1];
    shaderStrings[0] = glslShader.data();

    glslang::TShader shader(stage);
    shader.setStrings(shaderStrings, 1);

    EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);

    if (!shader.parse(&glslang::DefaultTBuiltInResource, 100, false, messages)) {
        throw std::runtime_error(shader.getInfoLog());
    }

    glslang::TProgram program;
    program.addShader(&shader);

    if (!program.link(messages)) {
        throw std::runtime_error(shader.getInfoLog());
    }

    std::vector<unsigned int> spvShader;
    glslang::GlslangToSpv(*program.getIntermediate(stage), spvShader);
    glslang::FinalizeProcess();

    return spvShader;
}


// sample code of SPIRV-Cross
void reflectSPV(const std::vector<unsigned int>& spvShader)
{
    spirv_cross::CompilerGLSL glsl{ spvShader };
    spirv_cross::ShaderResources resources = glsl.get_shader_resources();

    // output "set" and "binding"
    for (auto& resource : resources.storage_images) {
        unsigned set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
        unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
        std::cout << resource.name.c_str() << " set=" << set << " binding=" << binding << std::endl;
    }
    for (auto& resource : resources.uniform_buffers) {
        unsigned set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
        unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
        std::cout << resource.name.c_str() << " set=" << set << " binding=" << binding << std::endl;
    }
}

const auto glslShader = R"(
    #version 460
    layout(local_size_x = 1, local_size_y = 1) in;
    layout(binding = 0, set = 0, rgba8) uniform image2D inputImage;
    layout(binding = 1, set = 0, rgba8) uniform image2D outputImage;
    layout(binding = 2, set = 0) uniform UniformData{ int frame; } uniformData;

    void main()
    {
    }
)";

int main()
{
    createInstance();
    std::vector<unsigned int> spvShader = compileToSPV(EShLangCompute, glslShader);
    reflectSPV(spvShader);
    runWindow();
}
