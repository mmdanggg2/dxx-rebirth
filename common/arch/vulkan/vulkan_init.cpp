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
#include "vulkan_sync.h"
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
VkPipeline vk_3d_pipeline;
VkPipeline vk_3d_pipeline_additive_a;
VkPipeline vk_3d_pipeline_additive_c;
VkPipeline vk_3d_line_pipeline;
VkRenderPass vk_render_pass;
VkDescriptorSetLayout vk_descriptor_set_layout;
VkPipelineLayout vk_2d_pipeline_layout;
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_white_descriptor_set = VK_NULL_HANDLE;

VkShaderModule vk_vertex_shader;
VkShaderModule vk_3d_vertex_shader;
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
/* Maximum frames buffered on the GPU. Kept strictly below the swapchain image
 * count so an image is always free to acquire. The present (render-finished)
 * semaphore is owned per swapchain image, not per frame: a present binds it to
 * that image and the spec (VUID-vkQueueSubmit-pSignalSemaphores-00067) forbids
 * re-signaling until that image is re-acquired — see vks_init_sync_objects(). */
static constexpr uint32_t VK_MAX_FRAMES_IN_FLIGHT = 2;

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

vks_vertex_alloc vks_alloc_bytes(uint32_t bytes)
{
	auto &vb = vk_frame_vbs[vk_current_frame];
	if (vb.offset + bytes > vb.capacity)
		Error("Vulkan: vertex buffer overflow");
	vks_vertex_alloc a;
	a.buffer = vb.buffer;
	a.offset = vb.offset;
	a.data = static_cast<char *>(vb.mapped) + vb.offset;
	vb.offset += bytes;
	return a;
}

vks_vertex_alloc vks_alloc_vertices(uint32_t count)
{
	return vks_alloc_bytes(count * sizeof(vks_vertex));
}

VkCommandBuffer vks_get_command_buffer()
{
	return vk_command_buffers[vk_current_frame];
}

/* One depth image per swapchain image, so concurrent frames (rendering to
 * different swapchain images) never alias a single depth attachment. */
std::vector<VkImage> vk_depth_images;
std::vector<VkDeviceMemory> vk_depth_image_memories;
std::vector<VkImageView> vk_depth_image_views;

void initDepthResources()
{
	destroyDepthResources();
	const auto count = vk_swapchain_images.size();
	vk_depth_images.assign(count, VK_NULL_HANDLE);
	vk_depth_image_memories.assign(count, VK_NULL_HANDLE);
	vk_depth_image_views.assign(count, VK_NULL_HANDLE);

	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &memProperties);

	for (uint32_t i = 0; i < count; i++) {
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

		if (vkCreateImage(vk_device, &imageInfo, nullptr, &vk_depth_images[i]) != VK_SUCCESS)
			Error("Vulkan: Failed to create depth image");

		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(vk_device, vk_depth_images[i], &memReqs);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memReqs.size;
		for (uint32_t m = 0; m < memProperties.memoryTypeCount; m++) {
			if ((memReqs.memoryTypeBits & (1u << m)) &&
				(memProperties.memoryTypes[m].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
				allocInfo.memoryTypeIndex = m;
				break;
			}
		}

		if (vkAllocateMemory(vk_device, &allocInfo, nullptr, &vk_depth_image_memories[i]) != VK_SUCCESS)
			Error("Vulkan: Failed to allocate depth image memory");
		if (vkBindImageMemory(vk_device, vk_depth_images[i], vk_depth_image_memories[i], 0) != VK_SUCCESS)
			Error("Vulkan: Failed to bind depth image memory");

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = vk_depth_images[i];
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = vk_depth_format;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(vk_device, &viewInfo, nullptr, &vk_depth_image_views[i]) != VK_SUCCESS)
			Error("Vulkan: Failed to create depth image view");
	}
}

void destroyDepthResources()
{
	for (auto &v : vk_depth_image_views)
		if (v) vkDestroyImageView(vk_device, v, nullptr);
	for (auto &m : vk_depth_image_memories)
		if (m) vkFreeMemory(vk_device, m, nullptr);
	for (auto &img : vk_depth_images)
		if (img) vkDestroyImage(vk_device, img, nullptr);
	vk_depth_image_views.clear();
	vk_depth_image_memories.clear();
	vk_depth_images.clear();
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
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

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
	/* --- pipeline layout (shared by 2D and 3D) --- */
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
	if (vkCreatePipelineLayout(vk_device, &layoutInfo, nullptr, &vk_2d_pipeline_layout) != VK_SUCCESS)
		Error("Vulkan: Failed to create pipeline layout");

	/* --- shader modules (generated from GLSL at build time) --- */
	VkShaderModuleCreateInfo vinfo{};
	vinfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	vinfo.codeSize = vulkan::vertex_spv_size();
	vinfo.pCode = vulkan::vertex_spv_code();
	if (vkCreateShaderModule(vk_device, &vinfo, nullptr, &vk_vertex_shader) != VK_SUCCESS)
		Error("Vulkan: Failed to create vertex shader module");

	VkShaderModuleCreateInfo v3info{};
	v3info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	v3info.codeSize = vulkan::vertex3d_spv_size();
	v3info.pCode = vulkan::vertex3d_spv_code();
	if (vkCreateShaderModule(vk_device, &v3info, nullptr, &vk_3d_vertex_shader) != VK_SUCCESS)
		Error("Vulkan: Failed to create 3D vertex shader module");

	VkShaderModuleCreateInfo finfo{};
	finfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	finfo.codeSize = vulkan::fragment_spv_size();
	finfo.pCode = vulkan::fragment_spv_code();
	if (vkCreateShaderModule(vk_device, &finfo, nullptr, &vk_fragment_shader) != VK_SUCCESS)
		Error("Vulkan: Failed to create fragment shader module");

	/* --- state shared by both pipelines --- */
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;

	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

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

	constexpr std::array dynamicStates{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates = dynamicStates.data();

	VkPipelineShaderStageCreateInfo fragStage{};
	fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragStage.module = vk_fragment_shader;
	fragStage.pName = "main";

	auto make_rasterizer = []() {
		VkPipelineRasterizationStateCreateInfo r{};
		r.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		r.depthClampEnable = VK_FALSE;
		r.rasterizerDiscardEnable = VK_FALSE;
		r.polygonMode = VK_POLYGON_MODE_FILL;
		r.lineWidth = 1.0f;
		r.cullMode = VK_CULL_MODE_NONE;
		r.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		r.depthBiasEnable = VK_FALSE;
		return r;
	};

	/* --- 2D pipeline: pos(2)+uv(2)+color(4), triangle list, depth off --- */
	{
		VkVertexInputBindingDescription bindingDesc{};
		bindingDesc.binding = 0;
		bindingDesc.stride = sizeof(vks_vertex);
		bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription attrs[3]{};
		attrs[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
		attrs[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(vks_vertex, u)};
		attrs[2] = {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(vks_vertex, r)};

		VkPipelineVertexInputStateCreateInfo vertexInput{};
		vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInput.vertexBindingDescriptionCount = 1;
		vertexInput.pVertexBindingDescriptions = &bindingDesc;
		vertexInput.vertexAttributeDescriptionCount = 3;
		vertexInput.pVertexAttributeDescriptions = attrs;

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		VkPipelineRasterizationStateCreateInfo rasterizer = make_rasterizer();

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_FALSE;
		depthStencil.depthWriteEnable = VK_FALSE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vk_vertex_shader;
		stages[0].pName = "main";
		stages[1] = fragStage;

		VkGraphicsPipelineCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		info.stageCount = 2;
		info.pStages = stages;
		info.pVertexInputState = &vertexInput;
		info.pInputAssemblyState = &inputAssembly;
		info.pViewportState = &viewportState;
		info.pRasterizationState = &rasterizer;
		info.pMultisampleState = &multisampling;
		info.pColorBlendState = &colorBlending;
		info.pDepthStencilState = &depthStencil;
		info.pDynamicState = &dynamicState;
		info.layout = vk_2d_pipeline_layout;
		info.renderPass = vk_render_pass;
		info.subpass = 0;

		if (vkCreateGraphicsPipelines(vk_device, VK_NULL_HANDLE, 1, &info, nullptr, &vk_2d_pipeline) != VK_SUCCESS)
			Error("Vulkan: Failed to create 2D graphics pipeline");
	}

	/* --- 3D pipeline: pos(3)+uv(2)+color(3), triangle fan, depth test+write --- */
	{
		VkVertexInputBindingDescription bindingDesc{};
		bindingDesc.binding = 0;
		bindingDesc.stride = sizeof(vks_vertex3d);
		bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription attrs[3]{};
		attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
		attrs[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(vks_vertex3d, u)};
		attrs[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(vks_vertex3d, r)};

		VkPipelineVertexInputStateCreateInfo vertexInput{};
		vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInput.vertexBindingDescriptionCount = 1;
		vertexInput.pVertexBindingDescriptions = &bindingDesc;
		vertexInput.vertexAttributeDescriptionCount = 3;
		vertexInput.pVertexAttributeDescriptions = attrs;
		/* (inputAssembly is built per-pipeline inside make_3d so the line
		 * pipeline can use LINE_LIST while polygon pipelines use TRIANGLE_FAN.) */

		VkPipelineRasterizationStateCreateInfo rasterizer = make_rasterizer();

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vk_3d_vertex_shader;
		stages[0].pName = "main";
		stages[1] = fragStage;

		/* Build a 3D pipeline with the given colour blend factors. Depth config
		 * is identical for all blend modes: additive layers still depth-test
		 * against the mine (so glows hide behind walls) but ADD rather than
		 * replace, so overlapping additive layers -- a weapon's bright inner
		 * core drawn before its outer shell -- combine instead of the shell
		 * occluding the core. Mirrors ogl_set_blending: additive_a is
		 * (SRC_ALPHA, ONE), additive_c is (ONE, ONE). */
		auto make_3d = [&](VkBlendFactor srcColor, VkBlendFactor dstColor, VkPrimitiveTopology topology, VkPipeline &out) {
			VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
			inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
			inputAssembly.topology = topology;
			VkPipelineColorBlendAttachmentState b{};
			b.blendEnable = VK_TRUE;
			b.srcColorBlendFactor = srcColor;
			b.dstColorBlendFactor = dstColor;
			b.colorBlendOp = VK_BLEND_OP_ADD;
			b.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
			b.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
			b.alphaBlendOp = VK_BLEND_OP_ADD;
			b.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
				VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
			VkPipelineColorBlendStateCreateInfo cb{};
			cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
			cb.logicOpEnable = VK_FALSE;
			cb.attachmentCount = 1;
			cb.pAttachments = &b;
			VkGraphicsPipelineCreateInfo info{};
			info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
			info.stageCount = 2;
			info.pStages = stages;
			info.pVertexInputState = &vertexInput;
			info.pInputAssemblyState = &inputAssembly;
			info.pViewportState = &viewportState;
			info.pRasterizationState = &rasterizer;
			info.pMultisampleState = &multisampling;
			info.pColorBlendState = &cb;
			info.pDepthStencilState = &depthStencil;
			info.pDynamicState = &dynamicState;
			info.layout = vk_2d_pipeline_layout;
			info.renderPass = vk_render_pass;
			info.subpass = 0;
			if (vkCreateGraphicsPipelines(vk_device, VK_NULL_HANDLE, 1, &info, nullptr, &out) != VK_SUCCESS)
				Error("Vulkan: Failed to create 3D graphics pipeline");
		};
		make_3d(VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, vk_3d_pipeline);
		make_3d(VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, vk_3d_pipeline_additive_a);
		make_3d(VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, vk_3d_pipeline_additive_c);
		make_3d(VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, vk_3d_line_pipeline);
	}
}

void vks_init_framebuffers(uint32_t width, uint32_t height)
{
	/* One framebuffer per swapchain image — each must reference its own color
	 * view so a recorded frame renders into the image it will present. */
	const auto imageCount = vk_swapchain_image_views.size();
	vk_framebuffers.resize(imageCount);

	for (uint32_t i = 0; i < imageCount; i++) {
		VkImageView attachments[2] = {vk_swapchain_image_views[i], vk_depth_image_views[i]};

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
	/* Destroy any previously created pools and their buffers (e.g. after a
	 * resize).  The pools vector holds the old handles; clear() would leave
	 * the vector empty and the next loop would overwrite indices 0..N-1
	 * without destroying them. */
	if (!vk_command_pools.empty()) {
		/* Free the command buffers BEFORE destroying their pool: destroying a
		 * pool implicitly frees its buffers, so doing it afterward would touch
		 * already-destroyed handles. All buffers are allocated from pool[0]
		 * (see vkAllocateCommandBuffers below). */
		if (!vk_command_buffers.empty())
			vkFreeCommandBuffers(vk_device, vk_command_pools[0],
			                    static_cast<uint32_t>(vk_command_buffers.size()),
			                    vk_command_buffers.data());
		for (auto &pool : vk_command_pools)
			vkDestroyCommandPool(vk_device, pool, nullptr);
		vk_command_buffers.clear();
		vk_command_pools.clear();
	}

	vk_command_pools.resize(VK_MAX_FRAMES_IN_FLIGHT);

	for (uint32_t i = 0; i < VK_MAX_FRAMES_IN_FLIGHT; i++) {
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = vk_graphics_queue_family;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

		VkResult result = vkCreateCommandPool(vk_device, &poolInfo, nullptr, &vk_command_pools[i]);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create command pool");;
	}

	vk_command_buffers.resize(VK_MAX_FRAMES_IN_FLIGHT);

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

	/* Destroy any previously created semaphores and fences (e.g. after a
	 * resize). */
	for (auto &sem : vk_image_available_semaphores)
		if (sem)
			vkDestroySemaphore(vk_device, sem, nullptr);
	for (auto &sem : vk_present_semaphores)
		if (sem)
			vkDestroySemaphore(vk_device, sem, nullptr);
	for (auto &fence : vk_in_flight_fences)
		if (fence)
			vkDestroyFence(vk_device, fence, nullptr);

	/* image-available semaphores and in-flight fences are owned per
	 * frame-in-flight slot (indexed by vk_current_frame). The present
	 * semaphore is owned per swapchain image: vkQueuePresentKHR binds it
	 * to that image, and the spec forbids re-signaling it until the image
	 * is re-acquired, so it must be indexed by vk_image_index — one per
	 * swapchain image, not per frame in flight. */
	vk_image_available_semaphores.resize(VK_MAX_FRAMES_IN_FLIGHT);
	vk_in_flight_fences.resize(VK_MAX_FRAMES_IN_FLIGHT);
	const size_t imageCount = vk_swapchain_image_views.size();
	vk_present_semaphores.resize(imageCount);

	for (size_t i = 0; i < VK_MAX_FRAMES_IN_FLIGHT; i++) {
		VkResult result = vkCreateSemaphore(vk_device, &semaphoreInfo, nullptr, &vk_image_available_semaphores[i]);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create semaphore");;
		result = vkCreateFence(vk_device, &fenceInfo, nullptr, &vk_in_flight_fences[i]);
		if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create fence");;
	}
	for (size_t i = 0; i < imageCount; i++) {
		VkResult result = vkCreateSemaphore(vk_device, &semaphoreInfo, nullptr, &vk_present_semaphores[i]);
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

	destroyDepthResources();

	vks_destroy_white_texture();
	vks_destroy_vertex_buffers();

	if (vk_fragment_shader)
		vkDestroyShaderModule(vk_device, vk_fragment_shader, nullptr);
	if (vk_vertex_shader)
		vkDestroyShaderModule(vk_device, vk_vertex_shader, nullptr);
	if (vk_3d_vertex_shader)
		vkDestroyShaderModule(vk_device, vk_3d_vertex_shader, nullptr);
	if (vk_2d_pipeline)
		vkDestroyPipeline(vk_device, vk_2d_pipeline, nullptr);
	if (vk_3d_pipeline)
		vkDestroyPipeline(vk_device, vk_3d_pipeline, nullptr);
	if (vk_3d_pipeline_additive_a)
		vkDestroyPipeline(vk_device, vk_3d_pipeline_additive_a, nullptr);
	if (vk_3d_pipeline_additive_c)
		vkDestroyPipeline(vk_device, vk_3d_pipeline_additive_c, nullptr);
	if (vk_3d_line_pipeline)
		vkDestroyPipeline(vk_device, vk_3d_line_pipeline, nullptr);
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
	vk_device = VK_NULL_HANDLE;
	vkDestroySurfaceKHR(vk_instance, vk_surface, nullptr);
	
	if (vk_debug_messenger)
		DestroyDebugUtilsMessengerEXT(vk_instance, vk_debug_messenger, nullptr);

	vkDestroyInstance(vk_instance, nullptr);

}

} /* namespace dcx */

namespace dcx {

/* True while the current frame's command buffer is recording and its render
 * pass is open. Draw calls append to it; vks_present_frame() closes it. */
static bool vk_frame_recording = false;
bool vks_is_frame_recording()
{
	return vk_frame_recording;
}

/* Current 3D blend mode. Set by gr_settransblend (via vks_set_blend) and read
 * by vks_prepare_3d to pick the matching 3D pipeline. */
static gr_blend vk_current_blend = gr_blend::normal;
void vks_set_blend(gr_blend b) { vk_current_blend = b; }
gr_blend vks_get_blend() { return vk_current_blend; }

/* Set when acquire/present reports the swapchain out of date (or suboptimal);
 * vks_begin_frame rebuilds it before the next acquire. */
static bool vk_need_recreate = false;

/* Query the current surface extent from the physical device. On Wayland the
 * surface reports an undefined extent (0xFFFFFFFF), so fall back to the last
 * known size. */
static VkExtent2D vks_query_surface_extent()
{
	VkSurfaceCapabilitiesKHR caps{};
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk_physical_device, vk_surface, &caps);
	if (caps.currentExtent.width != UINT32_MAX && caps.currentExtent.height != UINT32_MAX)
		return caps.currentExtent;
	return vk_surface_extent;
}

void vks_recreate_swapchain(uint32_t w, uint32_t h)
{
	if (vk_swapchain) {
		vulkan_sync_helper.deinit();
		vkDeviceWaitIdle(vk_device);

		for (auto &fb : vk_framebuffers)
			vkDestroyFramebuffer(vk_device, fb, nullptr);
		if (vk_2d_pipeline)
			vkDestroyPipeline(vk_device, vk_2d_pipeline, nullptr);
		if (vk_3d_pipeline)
			vkDestroyPipeline(vk_device, vk_3d_pipeline, nullptr);
		if (vk_3d_pipeline_additive_a)
			vkDestroyPipeline(vk_device, vk_3d_pipeline_additive_a, nullptr);
		if (vk_3d_pipeline_additive_c)
			vkDestroyPipeline(vk_device, vk_3d_pipeline_additive_c, nullptr);
		if (vk_3d_line_pipeline)
			vkDestroyPipeline(vk_device, vk_3d_line_pipeline, nullptr);
		vkDestroySwapchainKHR(vk_device, vk_swapchain, nullptr);
		vk_swapchain = VK_NULL_HANDLE;
		for (auto &view : vk_swapchain_image_views)
			vkDestroyImageView(vk_device, view, nullptr);
		vk_swapchain_image_views.clear();
		destroyDepthResources();
		if (vk_render_pass)
			vkDestroyRenderPass(vk_device, vk_render_pass, nullptr);
	}

	vk_surface_extent = {w, h};
	vks_init_swapchain(w, h);
	vks_init_swapchain_image_views();
	vks_init_render_pass();
	initDepthResources();
	vks_init_command_buffers();
	vks_init_pipeline();
	vks_init_framebuffers(w, h);
	vks_init_sync_objects();
	vks_destroy_vertex_buffers();
	vks_init_vertex_buffers();
	if (vk_white_descriptor_set == VK_NULL_HANDLE)
		vks_init_white_texture();

	vulkan_sync_helper.init();
	last_width = w;
	last_height = h;
}

/* Begin a new frame: recycle the per-frame command buffer, acquire the next
 * swapchain image (into vk_image_index, distinct from the frame index), and
 * open a render pass that clears color and depth. Returns false if the
 * swapchain is out of date, so the caller can skip submit/present and let
 * the resize path (gr_set_mode) rebuild it. */
static bool vks_begin_frame()
{
	/* A frame may already be recording. A 2D draw (via vks_ensure_frame) can
	 * precede g3_start_frame within one displayed frame, and a window
	 * transition can leave the previous frame's pass still open. In both cases
	 * the 2D and 3D draws must composite into the single open render pass and
	 * be presented together by gr_flip — the single-buffer-per-frame model the
	 * software and OpenGL backends use. Starting a fresh pass here would
	 * orphan and present the in-progress frame, flickering every frame. */
	if (vk_frame_recording)
		return true;

	/* If the swapchain was reported out of date (by a prior acquire or
	 * present), rebuild it before attempting to acquire again. */
	if (vk_need_recreate) {
		const auto ext = vks_query_surface_extent();
		vks_recreate_swapchain(ext.width, ext.height);
		vk_need_recreate = false;
		vk_frame_recording = false;
	}

	/* Wait for this frame slot's previous submission to retire. The fence is
	 * reset only after a successful acquire below, so a failed acquire leaves
	 * it signaled and this wait returns immediately next time. */
	vkWaitForFences(vk_device, 1, &vk_in_flight_fences[vk_current_frame], VK_TRUE, UINT64_MAX);
	/* Destroy textures freed during earlier frames now that every command
	 * buffer that could reference them has completed (fence above + idle wait). */
	vks_flush_pending_texture_frees(vk_current_frame);

	/* This frame's vertex buffer is now retired (fence waited); reuse it. */
	if (vk_current_frame < vk_frame_vbs.size())
		vk_frame_vbs[vk_current_frame].offset = 0;

	VkResult result = vkAcquireNextImageKHR(vk_device, vk_swapchain, UINT64_MAX,
		vk_image_available_semaphores[vk_current_frame], VK_NULL_HANDLE, &vk_image_index);

	if (result == VK_ERROR_OUT_OF_DATE_KHR) {
		vk_need_recreate = true;
		return false;
	}
	if (result == VK_SUBOPTIMAL_KHR)
		vk_need_recreate = true;
	if (result != VK_SUCCESS) {
		con_printf(CON_URGENT, "Vulkan: vkAcquireNextImageKHR returned %d", static_cast<int>(result));
		if (result == VK_ERROR_SURFACE_LOST_KHR || result == VK_ERROR_DEVICE_LOST ||
			result == VK_ERROR_TOO_MANY_OBJECTS || result == VK_ERROR_OUT_OF_HOST_MEMORY ||
			result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
			vk_need_recreate = true;
			return false;
		}
		Error("Vulkan: Failed to acquire next swapchain image (result=%d)", static_cast<int>(result));
	}

	/* Now that we hold a valid image, reset the fence for this frame's
	 * upcoming submit. On a failed acquire above this fence stays signaled,
	 * preventing the next begin_frame from hanging. */
	vkResetFences(vk_device, 1, &vk_in_flight_fences[vk_current_frame]);

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

/* Ensure a frame is recording (begins one if none is open — begin_frame is
 * idempotent). Returns false if the swapchain is out of date, so 2D callers
 * can skip drawing. Used by draw paths that may run outside a 3D frame. */
bool vks_ensure_frame()
{
	return vks_begin_frame();
}

/* Close the render pass, submit the frame's command buffer, and present. This
 * is the entire body of gr_flip(). Skips presenting when nothing was drawn;
 * on swapchain loss flags recreate for the next begin_frame. */
void vks_present_frame()
{
	if (!vk_frame_recording)
	{
		return;
	}

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
		con_printf(CON_URGENT, "Vulkan: vkQueueSubmit failed: %d", static_cast<int>(result));

	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &vk_present_semaphores[vk_image_index];
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &vk_swapchain;
	presentInfo.pImageIndices = &vk_image_index;

	result = vkQueuePresentKHR(vk_graphics_queue, &presentInfo);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
		vk_need_recreate = true;
		return;
	}
	if (result != VK_SUCCESS)
		con_printf(CON_URGENT, "Vulkan: vkQueuePresentKHR failed");

	vk_current_frame = (vk_current_frame + 1) % VK_MAX_FRAMES_IN_FLIGHT;
}

void vks_end_frame()
{
	/* Intentionally a no-op: the render pass stays open so 2D overlays drawn
	 * after the 3D scene (HUD, menus) record into the same command buffer. It
	 * is closed and submitted by vks_present_frame(). */
}


} /* namespace dcx */

#endif /* DXX_USE_VULKAN */
