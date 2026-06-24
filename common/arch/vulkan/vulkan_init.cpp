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
#include "vulkan/shaders/generated.h"
#include "window.h"
#include "error.h"

#include <SDL.h>
#include <SDL_vulkan.h>
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
std::vector<VkImageView> vk_swapchain_image_views;
std::vector<VkImage> vk_swapchain_images;

uint32_t vk_graphics_queue_family;
uint32_t vk_surface_family;
VkExtent2D vk_surface_extent;
VkFormat vk_swapchain_format;
VkFormat vk_depth_format;

VkPipeline vk_2d_pipeline;
VkRenderPass vk_render_pass;
VkDescriptorSetLayout vk_descriptor_set_layout;
VkPipelineLayout vk_2d_pipeline_layout;
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_white_descriptor_set = VK_NULL_HANDLE;

VkShaderModule vk_vertex_shader;
VkShaderModule vk_fragment_shader;

std::vector<VkFramebuffer> vk_framebuffers;
std::vector<VkCommandBuffer> vk_command_buffers;
std::vector<VkCommandPool> vk_command_pools;
std::vector<VkSemaphore> vk_image_available_semaphores;
std::vector<VkSemaphore> vk_present_semaphores;
std::vector<VkFence> vk_in_flight_fences;

uint32_t vk_current_frame = 0;
uint32_t vk_image_index = 0;

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

	/* Device layers deprecated since Vulkan 1.0 — only enable on VkInstance. */
	createInfo.enabledLayerCount = 0;

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

/* Per-frame vertex buffers: one per frame-in-flight, host-visible and
 * persistently mapped. 2D draws append vertices to the current frame's
 * buffer; vks_begin_frame resets that buffer's cursor once its fence has
 * retired, so the GPU is done with the region we overwrite. */
struct vks_frame_vb
{
	VkBuffer buffer{VK_NULL_HANDLE};
	VkDeviceMemory memory{VK_NULL_HANDLE};
	void *mapped{};
	VkDeviceSize capacity{};
	VkDeviceSize offset{};
};
static std::vector<vks_frame_vb> vk_frame_vbs;
static constexpr VkDeviceSize VKS_VB_CAPACITY = 1 << 20; /* 1 MiB (~32k vertices) */

/* Find a host-visible, host-coherent memory type for a buffer. */
static uint32_t find_host_visible_memory_type(VkMemoryRequirements memRequirements)
{
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &memProperties);
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((memRequirements.memoryTypeBits & (1 << i)) &&
			(memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
			(memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
			return i;
		}
	}
	Error("Vulkan: no host-visible coherent memory type for vertex buffer");
}

void vks_init_vertex_buffers()
{
	const size_t n = vk_command_buffers.size();
	vk_frame_vbs.assign(n, vks_frame_vb{});
	for (auto &vb : vk_frame_vbs) {
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = VKS_VB_CAPACITY;
		bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkResult result = vkCreateBuffer(vk_device, &bufferInfo, nullptr, &vb.buffer);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create vertex buffer");

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(vk_device, vb.buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = find_host_visible_memory_type(memRequirements);

		result = vkAllocateMemory(vk_device, &allocInfo, nullptr, &vb.memory);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to allocate vertex buffer memory");

		result = vkBindBufferMemory(vk_device, vb.buffer, vb.memory, 0);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to bind vertex buffer memory");

		result = vkMapMemory(vk_device, vb.memory, 0, VKS_VB_CAPACITY, 0, &vb.mapped);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to map vertex buffer memory");

		vb.capacity = VKS_VB_CAPACITY;
		vb.offset = 0;
	}
}

void vks_destroy_vertex_buffers()
{
	for (auto &vb : vk_frame_vbs) {
		if (vb.mapped)
			vkUnmapMemory(vk_device, vb.memory);
		if (vb.buffer)
			vkDestroyBuffer(vk_device, vb.buffer, nullptr);
		if (vb.memory)
			vkFreeMemory(vk_device, vb.memory, nullptr);
	}
	vk_frame_vbs.clear();
}

vks_vertex_alloc vks_alloc_vertices(uint32_t count)
{
	auto &vb = vk_frame_vbs[vk_current_frame];
	if (vb.offset + count * sizeof(vks_vertex) > vb.capacity)
		Error("Vulkan: vertex buffer overflow");
	vks_vertex_alloc a;
	a.buffer = vb.buffer;
	a.offset = vb.offset;
	a.vertices = reinterpret_cast<vks_vertex *>(static_cast<char *>(vb.mapped) + vb.offset);
	vb.offset += count * sizeof(vks_vertex);
	return a;
}

VkCommandBuffer vks_get_command_buffer()
{
	return vk_command_buffers[vk_current_frame];
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

void vks_init_instance(SDL_Window *sdl_window)
{
	VkApplicationInfo appInfo{};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "DXX-Rebirth";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "DXX-Rebirth Engine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	/* Query required instance extensions from SDL — these are platform-specific
	 * (e.g. VK_KHR_wayland_surface, VK_KHR_xlib_surface, VK_KHR_win32_surface)
	 * and are needed by SDL_Vulkan_CreateSurface later. */
	uint32_t extensionCount = 0;
	SDL_Vulkan_GetInstanceExtensions(sdl_window, &extensionCount, nullptr);
	std::vector<const char *> requiredExtensions(extensionCount);
	SDL_Vulkan_GetInstanceExtensions(sdl_window, &extensionCount, requiredExtensions.data());

	bool layersSupported = false;

	if (enableValidationLayers) {
		/* The debug messenger requires its own extension. */
		requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		/* Query available layers */
		uint32_t layerCount = 0;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		layersSupported = true;
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
	}

	VkInstanceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;
	createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
	createInfo.ppEnabledExtensionNames = requiredExtensions.data();
	createInfo.enabledLayerCount = 0;

	if (enableValidationLayers && layersSupported) {
		createInfo.enabledLayerCount = static_cast<uint32_t>(std::size(validationLayers));
		createInfo.ppEnabledLayerNames = validationLayers;
	} else if (enableValidationLayers && !layersSupported) {
		con_printf(CON_URGENT, "Vulkan: Validation layers not available, continuing without them");
	}

	VkResult result = vkCreateInstance(&createInfo, nullptr, &vk_instance);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create instance");;

	if (enableValidationLayers && layersSupported) {
		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		debugCreateInfo.pfnUserCallback = debugCallback;
		if (CreateDebugUtilsMessengerEXT(vk_instance, &debugCreateInfo, nullptr, &vk_debug_messenger) != VK_SUCCESS)
			con_printf(CON_URGENT, "Vulkan: Failed to create debug messenger");
	}
}

void vks_init_physical_device()
{
	pickPhysicalDevice();
}

void vks_init_device()
{
	createDevice();
}

void vks_init_surface(SDL_Window *window_handle)
{
	VkSurfaceKHR surface{};
	SDL_bool result = SDL_Vulkan_CreateSurface(window_handle, vk_instance, &surface);
	if (result != SDL_TRUE)
		Error("Vulkan: Failed to create surface from SDL window: %s", SDL_GetError());
	vk_surface = surface;
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

void vks_init_swapchain_image_views()
{
	uint32_t imageCount;
	vkGetSwapchainImagesKHR(vk_device, vk_swapchain, &imageCount, nullptr);
	vk_swapchain_images.resize(imageCount);
	vkGetSwapchainImagesKHR(vk_device, vk_swapchain, &imageCount, vk_swapchain_images.data());

	vk_swapchain_image_views.resize(imageCount);
	for (uint32_t i = 0; i < imageCount; i++) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = vk_swapchain_images[i];
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = vk_swapchain_format;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.layerCount = 1;

		VkResult result = vkCreateImageView(vk_device, &viewInfo, nullptr, &vk_swapchain_image_views[i]);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create swapchain image view");
	}
}

void vks_destroy_swapchain_image_views()
{
	for (auto &view : vk_swapchain_image_views)
		vkDestroyImageView(vk_device, view, nullptr);
	vk_swapchain_image_views.clear();
}

void vks_record_initial_barriers()
{
	/* Fetch swapchain images */
	uint32_t imageCount;
	vkGetSwapchainImagesKHR(vk_device, vk_swapchain, &imageCount, nullptr);
	std::vector<VkImage> swapchainImages(imageCount);
	vkGetSwapchainImagesKHR(vk_device, vk_swapchain, &imageCount, swapchainImages.data());

	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.layerCount = 1;

	VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = vk_command_pools[0];
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	for (uint32_t i = 0; i < imageCount; i++) {
		barrier.image = swapchainImages[i];

		VkCommandBuffer tmpCmd;
		vkAllocateCommandBuffers(vk_device, &allocInfo, &tmpCmd);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(tmpCmd, &beginInfo);

		vkCmdPipelineBarrier(tmpCmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		vkEndCommandBuffer(tmpCmd);

		vkFreeCommandBuffers(vk_device, vk_command_pools[0], 1, &tmpCmd);
	}
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

	std::array<VkAttachmentDescription, 2> attachments = {{colorAttachment, depthAttachment}};

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
	/* Vertex input: pos(2) + uv(2) + color(4) = 32 bytes. */
	VkVertexInputBindingDescription bindingDesc{};
	bindingDesc.binding = 0;
	bindingDesc.stride = sizeof(vks_vertex);
	bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attrDescs[3]{};
	attrDescs[0].location = 0;
	attrDescs[0].binding = 0;
	attrDescs[0].format = VK_FORMAT_R32G32_SFLOAT;
	attrDescs[0].offset = 0; /* pos */
	attrDescs[1].location = 1;
	attrDescs[1].binding = 0;
	attrDescs[1].format = VK_FORMAT_R32G32_SFLOAT;
	attrDescs[1].offset = offsetof(vks_vertex, u);
	attrDescs[2].location = 2;
	attrDescs[2].binding = 0;
	attrDescs[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
	attrDescs[2].offset = offsetof(vks_vertex, r);

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
	vertexInputInfo.vertexAttributeDescriptionCount = 3;
	vertexInputInfo.pVertexAttributeDescriptions = attrDescs;

	/* Input assembly: triangle lists (quads/lines/pixels all emit triangles). */
	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	/* Viewport and scissor are set dynamically per draw (full screen). */
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;

	/* Rasterizer: no culling (2D), 1px lines. */
	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_NONE;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;

	/* Multisample */
	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	/* Color blend: standard alpha. Opaque primitives draw alpha 1 (no change);
	 * color-key transparency relies on the shader discarding alpha-0 texels. */
	VkPipelineColorBlendAttachmentState blendAttachment{};
	blendAttachment.blendEnable = VK_TRUE;
	blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
	blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &blendAttachment;

	/* Depth disabled for 2D (the attachment still exists for future 3D use). */
	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_FALSE;
	depthStencil.depthWriteEnable = VK_FALSE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.stencilTestEnable = VK_FALSE;

	/* Dynamic state: viewport + scissor set per frame. */
	constexpr std::array dynamicStates{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates = dynamicStates.data();

	/* Pipeline layout: texture descriptor set + 16-byte vertex push constant
	 * (pixel->NDC scale/offset). */
	VkPushConstantRange pushRange{};
	pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	pushRange.offset = 0;
	pushRange.size = sizeof(float) * 4;

	VkPipelineLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.setLayoutCount = 1;
	layoutInfo.pSetLayouts = &vk_descriptor_set_layout;
	layoutInfo.pushConstantRangeCount = 1;
	layoutInfo.pPushConstantRanges = &pushRange;

	VkResult result = vkCreatePipelineLayout(vk_device, &layoutInfo, nullptr, &vk_2d_pipeline_layout);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create pipeline layout");

	/* Vertex shader SPIR-V (generated at build time from GLSL) */
	VkShaderModuleCreateInfo vertexInfo{};
	vertexInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	vertexInfo.codeSize = vulkan::vertex_spv_size();
	vertexInfo.pCode = vulkan::vertex_spv_code();
	if (vkCreateShaderModule(vk_device, &vertexInfo, nullptr, &vk_vertex_shader) != VK_SUCCESS)
		Error("Vulkan: Failed to create vertex shader module");

	/* Fragment shader SPIR-V (generated at build time from GLSL) */
	VkShaderModuleCreateInfo fragmentInfo{};
	fragmentInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	fragmentInfo.codeSize = vulkan::fragment_spv_size();
	fragmentInfo.pCode = vulkan::fragment_spv_code();
	if (vkCreateShaderModule(vk_device, &fragmentInfo, nullptr, &vk_fragment_shader) != VK_SUCCESS)
		Error("Vulkan: Failed to create fragment shader module");

	std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vk_vertex_shader;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = vk_fragment_shader;
	stages[1].pName = "main";

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
	pipelineInfo.pStages = stages.data();
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = vk_2d_pipeline_layout;
	pipelineInfo.renderPass = vk_render_pass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

	result = vkCreateGraphicsPipelines(vk_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vk_2d_pipeline);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create graphics pipeline");
}

void vks_init_framebuffers(uint32_t width, uint32_t height)
{
	/* One framebuffer per swapchain image — each must reference its own color
	 * view so a recorded frame renders into the image it will present. */
	const auto imageCount = vk_swapchain_image_views.size();
	vk_framebuffers.resize(imageCount);

	for (uint32_t i = 0; i < imageCount; i++) {
		VkImageView attachments[2] = {vk_swapchain_image_views[i], vk_depth_image_view};

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
	vk_command_pools.resize(3);

	for (uint32_t i = 0; i < 3; i++) {
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = vk_graphics_queue_family;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

		VkResult result = vkCreateCommandPool(vk_device, &poolInfo, nullptr, &vk_command_pools[i]);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create command pool");;
	}

	vk_command_buffers.resize(3);

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = vk_command_pools[0];
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = static_cast<uint32_t>(vk_command_buffers.size());

	VkResult result = vkAllocateCommandBuffers(vk_device, &allocInfo, vk_command_buffers.data());
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
	vk_in_flight_fences.resize(3);

	for (size_t i = 0; i < 3; i++) {
		VkResult result = vkCreateSemaphore(vk_device, &semaphoreInfo, nullptr, &vk_image_available_semaphores[i]);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create semaphore");;
		result = vkCreateFence(vk_device, &fenceInfo, nullptr, &vk_in_flight_fences[i]);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create fence");;
	}

	/* Present-wait semaphores are per swapchain image, not per frame-in-flight:
	 * an image is only re-acquired once its previous present has retired, which
	 * is exactly when its present semaphore is guaranteed free. Indexing them by
	 * frame trips VUID-vkQueueSubmit-pSignalSemaphores-00067. */
	vk_present_semaphores.resize(vk_swapchain_images.size());
	for (auto &sem : vk_present_semaphores) {
		VkResult result = vkCreateSemaphore(vk_device, &semaphoreInfo, nullptr, &sem);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create present semaphore");;
	}
}

void vks_init_descriptor_pool()
{
	/* One combined-image-sampler descriptor per texture slot. Created once in
	 * gr_init and persists across resize/swapchain rebuilds, so existing
	 * textures keep valid descriptor sets when the screen mode changes. */
	VkDescriptorPoolSize poolSize{};
	poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSize.descriptorCount = VKS_MAX_TEXTURES;

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	poolInfo.maxSets = VKS_MAX_TEXTURES;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;

	VkResult result = vkCreateDescriptorPool(vk_device, &poolInfo, nullptr, &vk_descriptor_pool);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create descriptor pool");;
}

void vks_destroy_descriptor_pool()
{
	if (vk_descriptor_pool)
	{
		vkDestroyDescriptorPool(vk_device, vk_descriptor_pool, nullptr);
		vk_descriptor_pool = VK_NULL_HANDLE;
	}
}

void vks_shutdown()
{
	vkDeviceWaitIdle(vk_device);

	vks_shutdown_textures();
	vks_destroy_descriptor_pool();

	for (auto &fence : vk_in_flight_fences)
		vkDestroyFence(vk_device, fence, nullptr);
	for (auto &sem : vk_present_semaphores)
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

	vks_destroy_white_texture();
	vks_destroy_vertex_buffers();

	if (vk_fragment_shader)
		vkDestroyShaderModule(vk_device, vk_fragment_shader, nullptr);
	if (vk_vertex_shader)
		vkDestroyShaderModule(vk_device, vk_vertex_shader, nullptr);
	if (vk_2d_pipeline)
		vkDestroyPipeline(vk_device, vk_2d_pipeline, nullptr);
	if (vk_2d_pipeline_layout)
		vkDestroyPipelineLayout(vk_device, vk_2d_pipeline_layout, nullptr);
	if (vk_render_pass)
		vkDestroyRenderPass(vk_device, vk_render_pass, nullptr);
	if (vk_swapchain)
		vkDestroySwapchainKHR(vk_device, vk_swapchain, nullptr);

	vks_destroy_swapchain_image_views();

	if (vk_descriptor_set_layout)
		vkDestroyDescriptorSetLayout(vk_device, vk_descriptor_set_layout, nullptr);

	for (auto &pool : vk_command_pools)
		vkDestroyCommandPool(vk_device, pool, nullptr);

	vkDestroyDevice(vk_device, nullptr);
	vkDestroySurfaceKHR(vk_instance, vk_surface, nullptr);
	
	if (vk_debug_messenger)
		DestroyDebugUtilsMessengerEXT(vk_instance, vk_debug_messenger, nullptr);

	vkDestroyInstance(vk_instance, nullptr);

}

} /* namespace dcx */

namespace dcx {

/* Maximum frames buffered on the GPU. Must be strictly less than the swapchain
 * image count (3) so that, by the time a frame index recycles and its
 * render-finished semaphore is re-signaled, the swapchain has already consumed
 * that semaphore in the matching present. Reusing it with no slack trips
 * VUID-vkQueueSubmit-pSignalSemaphores-00067. Each frame owns its fence, two
 * semaphores, and a command buffer, all indexed by vk_current_frame. */
constexpr uint32_t VK_MAX_FRAMES_IN_FLIGHT = 2;

/* True while the current frame's command buffer is recording and its render
 * pass is open. Draw calls append to it; vks_present_frame() closes it. */
static bool vk_frame_recording = false;

/* Begin a new frame: recycle the per-frame command buffer, acquire the next
 * swapchain image (into vk_image_index, distinct from the frame index), and
 * open a render pass that clears color and depth. Returns false if the
 * swapchain is out of date, so the caller can skip submit/present and let
 * the resize path (gr_set_mode) rebuild it. */
static bool vks_begin_frame()
{
	vkWaitForFences(vk_device, 1, &vk_in_flight_fences[vk_current_frame], VK_TRUE, UINT64_MAX);
	vkResetFences(vk_device, 1, &vk_in_flight_fences[vk_current_frame]);

	/* This frame's vertex buffer is now retired (fence waited); reuse it. */
	if (vk_current_frame < vk_frame_vbs.size())
		vk_frame_vbs[vk_current_frame].offset = 0;

	VkResult result = vkAcquireNextImageKHR(vk_device, vk_swapchain, UINT64_MAX,
		vk_image_available_semaphores[vk_current_frame], VK_NULL_HANDLE, &vk_image_index);

	if (result == VK_ERROR_OUT_OF_DATE_KHR)
		return false;
	if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		Error("Vulkan: Failed to acquire next swapchain image");

	VkCommandBuffer cmd = vk_command_buffers[vk_current_frame];
	vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
		Error("Vulkan: Failed to begin command buffer");

	VkClearValue clearValues[2]{};
	clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
	clearValues[1].depthStencil = {1.0f, 0};

	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass = vk_render_pass;
	renderPassInfo.framebuffer = vk_framebuffers[vk_image_index];
	renderPassInfo.renderArea.offset = {0, 0};
	renderPassInfo.renderArea.extent = vk_surface_extent;
	renderPassInfo.clearValueCount = 2;
	renderPassInfo.pClearValues = clearValues;

	vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	vk_frame_recording = true;
	return true;
}

/* Called from g3_start_frame at the start of each 3D frame. */
void vks_start_frame(grs_canvas &)
{
	vks_begin_frame();
}

/* Ensure a frame is recording. Used by 2D draw paths that may run outside a
 * 3D frame (menus, loading screens) so their draws — and the per-frame clear
 * — still happen. */
void vks_ensure_frame()
{
	if (!vk_frame_recording)
		vks_begin_frame();
}

void vks_end_frame()
{
	/* Intentionally a no-op: the render pass stays open so 2D overlays drawn
	 * after the 3D scene (HUD, menus) record into the same command buffer. It
	 * is closed and submitted by vks_present_frame(). */
}

/* Close the render pass, submit the frame's command buffer, and present. This
 * is the entire body of gr_flip(). */
void vks_present_frame()
{
	if (!vk_frame_recording && !vks_begin_frame())
		return;

	VkCommandBuffer cmd = vk_command_buffers[vk_current_frame];

	vkCmdEndRenderPass(cmd);
	if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
		Error("Vulkan: Failed to end command buffer");
	vk_frame_recording = false;

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &vk_image_available_semaphores[vk_current_frame];
	submitInfo.pWaitDstStageMask = &waitStage;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmd;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &vk_present_semaphores[vk_image_index];

	VkResult result = vkQueueSubmit(vk_graphics_queue, 1, &submitInfo, vk_in_flight_fences[vk_current_frame]);
	if (result != VK_SUCCESS)
		Error("Vulkan: Failed to submit render commands");

	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &vk_present_semaphores[vk_image_index];
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &vk_swapchain;
	presentInfo.pImageIndices = &vk_image_index;

	result = vkQueuePresentKHR(vk_graphics_queue, &presentInfo);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
		return;
	if (result != VK_SUCCESS)
		con_printf(CON_URGENT, "Vulkan: vkQueuePresentKHR failed");

	vk_current_frame = (vk_current_frame + 1) % VK_MAX_FRAMES_IN_FLIGHT;
}

} /* namespace dcx */

#endif /* DXX_USE_VULKAN */
