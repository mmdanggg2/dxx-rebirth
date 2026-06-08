/*
 * This file is part of the DXX-Rebirth project <https://www.dxx-rebirth.com/>.
 * It is copyright by its individual contributors, as recorded in the
 * project's Git history.  See COPYING.txt at the top level for license
 * terms and a link to the Git history.
 */

#include "dxxsconf.h"
#if DXX_USE_VULKAN

#include "vulkan_init.h"
#include "vulkan_textures.h"
#include "window.h"
#include "error.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <set>
#include <span>
#include <vector>

namespace dcx {

/* SDL window handle — defined here for common Vulkan build (used by event.cpp) */
SDL_Window *g_pRebirthSDLMainWindow;

/* Vulkan global state */
VkInstance vk_instance;
VkPhysicalDevice vk_physical_device;
VkDevice vk_device;
VkQueue vk_graphics_queue;
VkSurfaceKHR vk_surface;
VkSwapchainKHR vk_swapchain;

uint32_t vk_graphics_queue_family;
uint32_t vk_surface_family;
VkExtent2D vk_surface_extent;
VkFormat vk_swapchain_format;
VkFormat vk_depth_format;

VkPipeline vk_render_pipeline;
VkRenderPass vk_render_pass;
VkDescriptorSetLayout vk_descriptor_set_layout;
VkPipelineLayout vk_pipeline_layout;

VkShaderModule vk_vertex_shader;
VkShaderModule vk_fragment_shader;

std::vector<VkFramebuffer> vk_framebuffers;
std::vector<VkCommandBuffer> vk_command_buffers;
std::vector<VkSemaphore> vk_image_available_semaphores;
std::vector<VkSemaphore> vk_render_finished_semaphores;
std::vector<VkFence> vk_in_flight_fences;

uint32_t vk_current_frame = 0;

/* Validation layers */
#ifdef NDEBUG
static constexpr bool enableValidationLayers = false;
#else
static constexpr bool enableValidationLayers = true;
#endif

static const char *const validationLayers[] = {
	"VK_LAYER_KHRONOS_validation",
};

static const std::vector<const char *> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

static VkDebugUtilsMessengerEXT vk_debug_messenger = VK_NULL_HANDLE;

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
	void * /*pUserData*/)
{
	if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
		con_printf(CON_URGENT, "Vulkan: message type %d: %s", messageType, pCallbackData->pMessage);
	}
	return VK_FALSE;
}

[[maybe_unused]] static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
	const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
	const VkAllocationCallbacks *pAllocator,
	VkDebugUtilsMessengerEXT *pDebugMessenger)
{
	auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
		vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
	if (func != nullptr)
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	return VK_ERROR_EXTENSION_NOT_PRESENT;
}

static void DestroyDebugUtilsMessengerEXT(VkInstance instance,
	VkDebugUtilsMessengerEXT debugMessenger,
	const VkAllocationCallbacks *pAllocator)
{
	auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
		vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
	if (func != nullptr)
		func(instance, debugMessenger, pAllocator);
}

static void pickPhysicalDevice()
{
	uint32_t deviceCount = 0;
	VkResult result = vkEnumeratePhysicalDevices(vk_instance, &deviceCount, nullptr);
	if (!( result == VK_SUCCESS )) Error("Vulkan: vkEnumeratePhysicalDevices failed");;
	if (!( deviceCount > 0 )) Error("Vulkan: No physical devices found");;

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(vk_instance, &deviceCount, devices.data());

	for (const auto &device : devices) {
		VkPhysicalDeviceProperties props{};
		vkGetPhysicalDeviceProperties(device, &props);

		/* Prefer discrete or integrated GPUs */
		if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
			props.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
			continue;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
		if (queueFamilyCount == 0)
			continue;

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		bool foundGraphics = false;
		bool foundPresent = false;
		uint32_t graphicsFamily = 0;
		uint32_t presentFamily = 0;

		for (uint32_t i = 0; i < queueFamilyCount; i++) {
			if (!foundGraphics && (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
				graphicsFamily = i;
				foundGraphics = true;
			}
			VkBool32 presentSupport = VK_FALSE;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, vk_surface, &presentSupport);
			if (presentSupport) {
				presentFamily = i;
				foundPresent = true;
			}
		}

		if (foundGraphics && foundPresent) {
			vk_physical_device = device;
			vk_graphics_queue_family = graphicsFamily;
			vk_surface_family = presentFamily;
			return;
		}
	}

	Error("Vulkan: Could not find a suitable physical device");
}

static void createDevice()
{
	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo queueCreateInfo{};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex = vk_graphics_queue_family;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = &queuePriority;

	VkPhysicalDeviceFeatures deviceFeatures{};
	deviceFeatures.samplerAnisotropy = VK_TRUE;

	VkDeviceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.queueCreateInfoCount = 1;
	createInfo.pQueueCreateInfos = &queueCreateInfo;
	createInfo.pEnabledFeatures = &deviceFeatures;
	createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
	createInfo.ppEnabledExtensionNames = deviceExtensions.data();

	if (enableValidationLayers) {
		createInfo.enabledLayerCount = static_cast<uint32_t>(std::size(validationLayers));
		createInfo.ppEnabledLayerNames = validationLayers;
	} else {
		createInfo.enabledLayerCount = 0;
	}

	VkResult result = vkCreateDevice(vk_physical_device, &createInfo, nullptr, &vk_device);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create device");;

	vkGetDeviceQueue(vk_device, vk_graphics_queue_family, 0, &vk_graphics_queue);
}

[[maybe_unused]] static VkShaderModule compileShaderModule(const std::vector<char> &code)
{
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

	VkShaderModule shaderModule;
	VkResult result = vkCreateShaderModule(vk_device, &createInfo, nullptr, &shaderModule);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create shader module");;
	return shaderModule;
}

/* Fullscreen quad for textured rendering: two triangles covering clip space */
static const std::array<float, 24> fullscreenQuad = {
	-1.0f, -1.0f,  0.0f, 0.0f,
	 1.0f, -1.0f,  1.0f, 0.0f,
	 1.0f,  1.0f,  1.0f, 1.0f,
	-1.0f, -1.0f,  0.0f, 0.0f,
	 1.0f,  1.0f,  1.0f, 1.0f,
	-1.0f,  1.0f,  0.0f, 1.0f,
};

static VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
static VkDeviceMemory vk_vertex_buffer_memory = VK_NULL_HANDLE;

void initVertexBuffers()
{
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = fullscreenQuad.size() * sizeof(float);
	bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkResult result = vkCreateBuffer(vk_device, &bufferInfo, nullptr, &vk_vertex_buffer);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create vertex buffer");;

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(vk_device, vk_vertex_buffer, &memRequirements);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;

	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &memProperties);
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((memRequirements.memoryTypeBits & (1 << i)) &&
			(memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
			(memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
			allocInfo.memoryTypeIndex = i;
			break;
		}
	}

	result = vkAllocateMemory(vk_device, &allocInfo, nullptr, &vk_vertex_buffer_memory);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to allocate vertex buffer memory");;

	void *data;
	result = vkMapMemory(vk_device, vk_vertex_buffer_memory, 0, bufferInfo.size, 0, &data);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to map vertex buffer memory");;
	std::memcpy(data, fullscreenQuad.data(), bufferInfo.size);
	vkUnmapMemory(vk_device, vk_vertex_buffer_memory);
}

VkImage vk_depth_image = VK_NULL_HANDLE;
VkDeviceMemory vk_depth_image_memory = VK_NULL_HANDLE;
VkImageView vk_depth_image_view = VK_NULL_HANDLE;

void initDepthResources()
{
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = vk_surface_extent.width;
	imageInfo.extent.height = vk_surface_extent.height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.format = vk_depth_format;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VkResult result = vkCreateImage(vk_device, &imageInfo, nullptr, &vk_depth_image);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create depth image");;

	VkMemoryRequirements memReqs;
	vkGetImageMemoryRequirements(vk_device, vk_depth_image, &memReqs);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReqs.size;

	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &memProperties);
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((memReqs.memoryTypeBits & (1 << i)) &&
			(memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
			allocInfo.memoryTypeIndex = i;
			break;
		}
	}

	result = vkAllocateMemory(vk_device, &allocInfo, nullptr, &vk_depth_image_memory);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to allocate depth image memory");;

	result = vkBindImageMemory(vk_device, vk_depth_image, vk_depth_image_memory, 0);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to bind depth image memory");;

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = vk_depth_image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = vk_depth_format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.layerCount = 1;

	result = vkCreateImageView(vk_device, &viewInfo, nullptr, &vk_depth_image_view);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create depth image view");;
}

void vks_init_instance()
{
	VkApplicationInfo appInfo{};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "DXX-Rebirth";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "DXX-Rebirth Engine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;
	createInfo.enabledExtensionCount = 0;
	createInfo.enabledLayerCount = 0;

	if (enableValidationLayers) {
		/* Query available layers */
		uint32_t layerCount = 0;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		bool layersSupported = true;
		for (const char *layerName : validationLayers) {
			bool found = false;
			for (const auto &lp : availableLayers) {
				if (std::strcmp(lp.layerName, layerName) == 0) {
					found = true;
					break;
				}
			}
			if (!found) {
				layersSupported = false;
				break;
			}
		}

		if (layersSupported) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(std::size(validationLayers));
			createInfo.ppEnabledLayerNames = validationLayers;

			VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
			debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			debugCreateInfo.pfnUserCallback = debugCallback;
			createInfo.pNext = &debugCreateInfo;
		} else {
			con_printf(CON_URGENT, "Vulkan: Validation layers not available, continuing without them");
		}
	}

	VkResult result = vkCreateInstance(&createInfo, nullptr, &vk_instance);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create instance");;
}

void vks_init_physical_device()
{
	vk_surface = VK_NULL_HANDLE;
	pickPhysicalDevice();
}

void vks_init_device()
{
	createDevice();
}

void vks_init_surface(void * /*window_handle*/)
{
	/* Surface creation is delegated to similar/arch/vulkan/vk_init.cpp
	 * which has access to the SDL window surface.
	 */
}

void vks_init_swapchain(uint32_t width, uint32_t height)
{
	vk_surface_extent = {width, height};

	/* Find supported swapchain format */
	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(vk_physical_device, vk_surface, &formatCount, nullptr);
	std::vector<VkSurfaceFormatKHR> formats(formatCount);
	vkGetPhysicalDeviceSurfaceFormatsKHR(vk_physical_device, vk_surface, &formatCount, formats.data());

	if (formatCount == 1 && formats[0].format == VK_FORMAT_UNDEFINED)
		vk_swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
	else {
		bool found = false;
		for (const auto &f : formats) {
			if (f.format == VK_FORMAT_B8G8R8A8_UNORM) {
				vk_swapchain_format = f.format;
				found = true;
				break;
			}
		}
		if (!found)
			vk_swapchain_format = formats[0].format;
	}

	/* Find present mode */
	uint32_t modeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(vk_physical_device, vk_surface, &modeCount, nullptr);
	std::vector<VkPresentModeKHR> presentModes(modeCount);
	vkGetPhysicalDeviceSurfacePresentModesKHR(vk_physical_device, vk_surface, &modeCount, presentModes.data());

	VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
	for (const auto &pm : presentModes) {
		if (pm == VK_PRESENT_MODE_MAILBOX_KHR) {
			presentMode = pm;
			break;
		}
	}

	VkSwapchainCreateInfoKHR createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = vk_surface;
	createInfo.minImageCount = 3;
	createInfo.imageFormat = vk_swapchain_format;
	createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	createInfo.imageExtent = vk_surface_extent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	createInfo.presentMode = presentMode;
	createInfo.clipped = VK_TRUE;
	createInfo.oldSwapchain = VK_NULL_HANDLE;

	VkResult result = vkCreateSwapchainKHR(vk_device, &createInfo, nullptr, &vk_swapchain);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create swapchain");;
}

void vks_init_render_pass()
{
	VkAttachmentDescription colorAttachment{};
	colorAttachment.format = vk_swapchain_format;
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentDescription depthAttachment{};
	depthAttachment.format = vk_depth_format;
	depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorAttachmentRef{};
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthAttachmentRef{};
	depthAttachmentRef.attachment = 1;
	depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;
	subpass.pDepthStencilAttachment = &depthAttachmentRef;

	VkSubpassDependency dependency{};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};

	VkRenderPassCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	createInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
	createInfo.pAttachments = attachments.data();
	createInfo.subpassCount = 1;
	createInfo.pSubpasses = &subpass;
	createInfo.dependencyCount = 1;
	createInfo.pDependencies = &dependency;

	VkResult result = vkCreateRenderPass(vk_device, &createInfo, nullptr, &vk_render_pass);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create render pass");;
}

void vks_init_pipeline()
{
	/* Vertex input */
	VkVertexInputBindingDescription bindingDesc{};
	bindingDesc.binding = 0;
	bindingDesc.stride = sizeof(float) * 4;
	bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attrDescs[2]{};
	attrDescs[0].location = 0;
	attrDescs[0].binding = 0;
	attrDescs[0].format = VK_FORMAT_R32G32_SFLOAT;
	attrDescs[0].offset = 0;
	attrDescs[1].location = 1;
	attrDescs[1].binding = 0;
	attrDescs[1].format = VK_FORMAT_R32G32_SFLOAT;
	attrDescs[1].offset = sizeof(float) * 2;

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
	vertexInputInfo.vertexAttributeDescriptionCount = 2;
	vertexInputInfo.pVertexAttributeDescriptions = attrDescs;

	/* Input assembly */
	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	/* Viewport and scissor */
	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = static_cast<float>(vk_surface_extent.width);
	viewport.height = static_cast<float>(vk_surface_extent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor{};
	scissor.offset = {0, 0};
	scissor.extent = vk_surface_extent;

	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	/* Rasterizer */
	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;

	/* Multisample */
	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	/* Color blend */
	VkPipelineColorBlendAttachmentState blendAttachment{};
	blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	blendAttachment.blendEnable = VK_FALSE;

	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &blendAttachment;

	/* Depth and stencil */
	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.stencilTestEnable = VK_FALSE;

	/* Dynamic state */
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = 0;
	dynamicState.pDynamicStates = nullptr;

	/* Pipeline layout */
	VkPipelineLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.setLayoutCount = 1;
	layoutInfo.pSetLayouts = &vk_descriptor_set_layout;
	layoutInfo.pushConstantRangeCount = 0;

	VkResult result = vkCreatePipelineLayout(vk_device, &layoutInfo, nullptr, &vk_pipeline_layout);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create pipeline layout");;

	/* Pipeline — stages will be filled in once shaders are compiled */
	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 0;
	pipelineInfo.pStages = nullptr;
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = vk_pipeline_layout;
	pipelineInfo.renderPass = vk_render_pass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

	result = vkCreateGraphicsPipelines(vk_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vk_render_pipeline);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create graphics pipeline");;
}

void vks_init_framebuffers(uint32_t width, uint32_t height)
{
	vk_framebuffers.resize(3);

	VkImageView attachments[2] = {VK_NULL_HANDLE, vk_depth_image_view};

	for (uint32_t i = 0; i < 3; i++) {
		VkFramebufferCreateInfo fbInfo{};
		fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbInfo.renderPass = vk_render_pass;
		fbInfo.attachmentCount = 2;
		fbInfo.pAttachments = attachments;
		fbInfo.width = width;
		fbInfo.height = height;
		fbInfo.layers = 1;

		VkResult result = vkCreateFramebuffer(vk_device, &fbInfo, nullptr, &vk_framebuffers[i]);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create framebuffer");;
	}
}

void vks_init_command_buffers()
{
	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = vk_graphics_queue_family;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	VkCommandPool commandPool;
	VkResult result = vkCreateCommandPool(vk_device, &poolInfo, nullptr, &commandPool);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create command pool");;

	vk_command_buffers.resize(3);

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = commandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = static_cast<uint32_t>(vk_command_buffers.size());

	result = vkAllocateCommandBuffers(vk_device, &allocInfo, vk_command_buffers.data());
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to allocate command buffers");;
}

void vks_init_sync_objects()
{
	VkSemaphoreCreateInfo semaphoreInfo{};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	vk_image_available_semaphores.resize(3);
	vk_render_finished_semaphores.resize(3);
	vk_in_flight_fences.resize(3);

	for (size_t i = 0; i < 3; i++) {
		VkResult result = vkCreateSemaphore(vk_device, &semaphoreInfo, nullptr, &vk_image_available_semaphores[i]);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create semaphore");;
		result = vkCreateSemaphore(vk_device, &semaphoreInfo, nullptr, &vk_render_finished_semaphores[i]);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create semaphore");;
		result = vkCreateFence(vk_device, &fenceInfo, nullptr, &vk_in_flight_fences[i]);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create fence");;
	}
}

void vks_wait_for_flight_fence(uint32_t frame)
{
	vkWaitForFences(vk_device, 1, &vk_in_flight_fences[frame], VK_TRUE, UINT64_MAX);
	vkResetFences(vk_device, 1, &vk_in_flight_fences[frame]);
}

int vks_acquire_next_image()
{
	VkResult result = vkAcquireNextImageKHR(vk_device, vk_swapchain, UINT64_MAX,
		vk_image_available_semaphores[vk_current_frame], VK_NULL_HANDLE, &vk_current_frame);

	if (result == VK_ERROR_OUT_OF_DATE_KHR)
		return -1;
	if (result == VK_SUBOPTIMAL_KHR)
		return 1;
	if (result != VK_SUCCESS)
		return -1;
	return 0;
}

void vks_shutdown()
{
	vkDeviceWaitIdle(vk_device);

	vks_shutdown_textures();

	for (auto &fence : vk_in_flight_fences)
		vkDestroyFence(vk_device, fence, nullptr);
	for (auto &sem : vk_render_finished_semaphores)
		vkDestroySemaphore(vk_device, sem, nullptr);
	for (auto &sem : vk_image_available_semaphores)
		vkDestroySemaphore(vk_device, sem, nullptr);
	for (auto &fb : vk_framebuffers)
		vkDestroyFramebuffer(vk_device, fb, nullptr);

	if (vk_depth_image_view)
		vkDestroyImageView(vk_device, vk_depth_image_view, nullptr);
	if (vk_depth_image_memory)
		vkFreeMemory(vk_device, vk_depth_image_memory, nullptr);
	if (vk_depth_image)
		vkDestroyImage(vk_device, vk_depth_image, nullptr);

	if (vk_vertex_buffer_memory)
		vkFreeMemory(vk_device, vk_vertex_buffer_memory, nullptr);
	if (vk_vertex_buffer)
		vkDestroyBuffer(vk_device, vk_vertex_buffer, nullptr);

	if (vk_render_pipeline)
		vkDestroyPipeline(vk_device, vk_render_pipeline, nullptr);
	if (vk_pipeline_layout)
		vkDestroyPipelineLayout(vk_device, vk_pipeline_layout, nullptr);
	if (vk_render_pass)
		vkDestroyRenderPass(vk_device, vk_render_pass, nullptr);
	if (vk_swapchain)
		vkDestroySwapchainKHR(vk_device, vk_swapchain, nullptr);

	vkDestroyDevice(vk_device, nullptr);
	vkDestroySurfaceKHR(vk_instance, vk_surface, nullptr);
	vkDestroyInstance(vk_instance, nullptr);

	if (vk_debug_messenger)
		DestroyDebugUtilsMessengerEXT(vk_instance, vk_debug_messenger, nullptr);
}

} /* namespace dcx */

namespace dcx {

void vks_start_frame(grs_canvas &)
{
	/* Stub: Vulkan frame would be started here */
}

void vks_end_frame()
{
	/* Stub: Vulkan frame would be ended here */
}

} /* namespace dcx */

#endif /* DXX_USE_VULKAN */
