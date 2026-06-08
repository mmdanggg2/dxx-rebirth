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
#include "gr.h"
#include "game.h"
#include "config.h"
#include "console.h"
#include "args.h"
#include "gamefont.h"
#include "error.h"
#include "inferno.h"
#include "vers_id.h"
#include "internal.h"
#include "physfsx.h"
#include "u_mem.h"
#include "strutil.h"
#include "dxxerror.h"
#include "timer.h"
#include "maths.h"
#include "multi.h"
#include "args.h"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <cstring>
#include <algorithm>
#include <array>
#include <vector>
#include <set>

/* Window position tracking — defined here since ogl/gr.cpp only compiles for OpenGL builds */
int g_iRebirthWindowX = 0, g_iRebirthWindowY = 0;

namespace dcx {

/* Vulkan global state — populated by vk_init.cpp */
SDL_Window *g_pRebirthVulkanWindow = nullptr;

/* Sync helper */
extern vks_sync vulkan_sync_helper;

/* Fullscreen tracking */
static int gr_installed = 0;
static int vulkan_initialized = 0;

/* Depth format selection */
static void find_depth_format()
{
	VkFormat candidateFormats[] = {
		VK_FORMAT_D32_SFLOAT_S8_UINT,
		VK_FORMAT_D32_SFLOAT,
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_D16_UNORM_S8_UINT,
		VK_FORMAT_D16_UNORM,
	};

	VkFormatProperties props;
	for (auto format : candidateFormats) {
		vkGetPhysicalDeviceFormatProperties(vk_physical_device, format, &props);
		if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
			vk_depth_format = format;
			con_printf(CON_DEBUG, "Vulkan: Selected depth format: %d", static_cast<int>(format));
			return;
		}
	}
	/* Fallback */
	vk_depth_format = VK_FORMAT_D16_UNORM;
}

/* Descriptor set layout creation */
static void init_descriptor_set_layout()
{
	VkDescriptorSetLayoutBinding binding{};
	binding.binding = 0;
	binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	binding.descriptorCount = 1;
	binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	binding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	info.bindingCount = 1;
	info.pBindings = &binding;

	VkResult result = vkCreateDescriptorSetLayout(vk_device, &info, nullptr, &vk_descriptor_set_layout);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create descriptor set layout");;
}

int gr_check_fullscreen(void)
{
	return !!(SDL_GetWindowFlags(g_pRebirthVulkanWindow) & SDL_WINDOW_FULLSCREEN);
}

void gr_toggle_fullscreen()
{
	const auto SDLWindow{g_pRebirthVulkanWindow};
	const bool is_fullscreen = SDL_GetWindowFlags(SDLWindow) & SDL_WINDOW_FULLSCREEN;
	CGameCfg.WindowMode = is_fullscreen;
	if (!is_fullscreen)
		SDL_GetWindowPosition(SDLWindow, &g_iRebirthWindowX, &g_iRebirthWindowY);
	SDL_SetWindowFullscreen(SDLWindow, is_fullscreen ? 0 : SDL_WINDOW_FULLSCREEN_DESKTOP);
	if (is_fullscreen) {
		const auto mode{Game_screen_mode};
		SDL_SetWindowPosition(SDLWindow, g_iRebirthWindowX, g_iRebirthWindowY);
		SDL_SetWindowSize(SDLWindow, SM_W(mode), SM_H(mode));
	}
	gr_set_mode_from_window_size();
}

uint_fast32_t gr_list_modes(std::array<screen_mode, 50> &gsmodes)
{
	int modeCount = SDL_GetNumDisplayModes(0);
	uint_fast32_t modesnum = 0;
	for (int i = 0; i < modeCount && modesnum < gsmodes.size(); i++) {
		SDL_DisplayMode mode;
		if (SDL_GetDisplayMode(0, i, &mode) == 0) {
			if (mode.w > 320 && mode.h > 200 && mode.w < 4096 && mode.h < 4096) {
				gsmodes[modesnum].width = mode.w;
				gsmodes[modesnum].height = mode.h;
				modesnum++;
			}
		}
	}
	return modesnum;
}

} /* namespace dcx */

namespace dsx {

void gr_set_mode_from_window_size()
{
	/* Stub — full implementation requires window resize logic */
}

void vulkan_init_surface_from_window(void *window)
{
#ifdef _WIN32
	VkWin32SurfaceCreateInfoEXT createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_EXT;
	createInfo.hwnd = GetWindowLongPtr(window, GWLP_HWNDPRC) ? reinterpret_cast<HWND>(GetWindowLongPtr(window, GWLP_HWNDPRC)) : nullptr;
	VkSurfaceKHR surface;
	VkResult result = vkCreateWin32SurfaceKHR(::dcx::vk_instance, &createInfo, nullptr, &surface);
	if (result == VK_SUCCESS) {
		::dcx::vk_surface = surface;
	}
#elif defined(__APPLE__)
	VkIOSSurfaceCreateInfoMVK createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_IO_SURFACE_CREATE_INFO_MVK;
	createInfo.view = static_cast<MTLTexture *>(nullptr);  /* Simplified */
	VkSurfaceKHR surface;
	VkResult result = vkCreateIOSurfaceSurfaceMVK(::dcx::vk_instance, &createInfo, nullptr, &surface);
	if (result == VK_SUCCESS) {
		::dcx::vk_surface = surface;
	}
#else
	/* Use SDL_Vulkan functions */
	VkSurfaceKHR surface{};
	SDL_bool result = SDL_Vulkan_CreateSurface(reinterpret_cast<SDL_Window *>(window), ::dcx::vk_instance, &surface);
	if (result == SDL_TRUE) {
		::dcx::vk_surface = surface;
	}
#endif
	(void)surface; /* used above but may be unused depending on platform */
}

int gr_init()
{
	if (gr_installed)
		return -1;

	/* Initialize Vulkan instance */
	::dcx::vks_init_instance();

	/* Create surface from SDL window */
	assert(g_pRebirthVulkanWindow);
	::dcx::vks_init_surface(reinterpret_cast<void *>(g_pRebirthVulkanWindow));

	/* Initialize physical device */
	::dcx::vks_init_physical_device();
	::dcx::vks_init_device();

	/* Find depth format */
	::dcx::find_depth_format();

	/* Create descriptor set layout */
	::dcx::init_descriptor_set_layout();

	/* Set up sync helper */
	vulkan_sync_helper.init();

	gr_installed = 1;
	vulkan_initialized = 1;

	return 0;
}

int gr_set_mode(screen_mode mode)
{
	const uint32_t w = SM_W(mode), h = SM_H(mode);

	/* Reallocate screen buffer */
	uint8_t *gr_bm_data = grd_curscreen->sc_canvas.cv_bitmap.get_bitmap_data();
	auto gr_new_bm_data = reinterpret_cast<uint8_t *>(d_realloc(gr_bm_data, w * h));
	if (!gr_new_bm_data)
		return 0;

	/* Reset screen state */
	*grd_curscreen = {};
	grd_curscreen->set_screen_width_height(w, h);
	grd_curscreen->sc_aspect = fixdiv(grd_curscreen->get_screen_width() * CGameCfg.AspectX,
		grd_curscreen->get_screen_height() * CGameCfg.AspectY);
	gr_init_canvas(grd_curscreen->sc_canvas, gr_new_bm_data, bm_mode::vulkan, w, h);

	/* Reinitialize Vulkan swapchain */
	::dcx::vk_surface_extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};

	/* Destroy old swapchain and related resources */
	if (::dcx::vk_swapchain) {
		vulkan_sync_helper.deinit();
		vkDeviceWaitIdle(::dcx::vk_device);

		for (auto &fence : ::dcx::vk_in_flight_fences)
			vkDestroyFence(::dcx::vk_device, fence, nullptr);
		for (auto &sem : ::dcx::vk_render_finished_semaphores)
			vkDestroySemaphore(::dcx::vk_device, sem, nullptr);
		for (auto &sem : ::dcx::vk_image_available_semaphores)
			vkDestroySemaphore(::dcx::vk_device, sem, nullptr);
		for (auto &fb : ::dcx::vk_framebuffers)
			vkDestroyFramebuffer(::dcx::vk_device, fb, nullptr);
		if (::dcx::vk_render_pipeline)
			vkDestroyPipeline(::dcx::vk_device, ::dcx::vk_render_pipeline, nullptr);
		if (::dcx::vk_swapchain)
			vkDestroySwapchainKHR(::dcx::vk_device, ::dcx::vk_swapchain, nullptr);
		if (::dcx::vk_depth_image_view)
			vkDestroyImageView(::dcx::vk_device, ::dcx::vk_depth_image_view, nullptr);
		if (::dcx::vk_depth_image_memory)
			vkFreeMemory(::dcx::vk_device, ::dcx::vk_depth_image_memory, nullptr);
		if (::dcx::vk_depth_image)
			vkDestroyImage(::dcx::vk_device, ::dcx::vk_depth_image, nullptr);
		if (::dcx::vk_render_pass)
			vkDestroyRenderPass(::dcx::vk_device, ::dcx::vk_render_pass, nullptr);
	}

	/* Recreate swapchain */
	::dcx::vks_init_swapchain(w, h);
	::dcx::vks_init_render_pass();
	::dcx::vks_init_pipeline();
	::dcx::vks_init_framebuffers(w, h);
	::dcx::vks_init_command_buffers();
	::dcx::vks_init_sync_objects();

	/* Recreate depth resources */
	::dcx::initDepthResources();

	/* Recreate vertex buffers */
	::dcx::initVertexBuffers();

	/* Update sync helper */
	vulkan_sync_helper.init();

	/* Initialize rendering state */
	::dcx::vks_init_state();
	gamefont_choose_game_font(w, h);
	gr_remap_color_fonts();

	last_width = w;
	last_height = h;

#if DXX_USE_STEREOSCOPIC_RENDER
	// gr_set_stereo_mode_sync();
#endif

	return 0;
}

void gr_set_attributes(void)
{
	/* Vulkan doesn't use SDL GL attributes, but we set window flags */
	/* TODO: Apply these flags when creating the Vulkan window */
	(void)SDL_WINDOW_VULKAN;
	if (CGameArg.SysNoBorders)
		(void)SDL_WINDOW_BORDERLESS;
	if (!CGameCfg.WindowMode && !CGameArg.SysWindow)
		(void)SDL_WINDOW_FULLSCREEN_DESKTOP;
#if defined(__APPLE__) && defined(__MACH__)
	(void)SDL_WINDOW_ALLOW_HIGHDPI;
#endif
}

void gr_close()
{
	if (!gr_installed)
		return;

	if (vulkan_initialized)
	{
		vulkan_sync_helper.deinit();
		::dcx::vks_shutdown_textures();
		::dcx::vks_shutdown();
	}

	if (grd_curscreen)
	{
		if (grd_curscreen->sc_canvas.cv_bitmap.bm_mdata)
			d_free(grd_curscreen->sc_canvas.cv_bitmap.bm_mdata);
		grd_curcanv = nullptr;
		grd_curscreen.reset();
	}

	gr_installed = 0;
	vulkan_initialized = 0;
}

} /* namespace dsx */

#endif /* DXX_USE_VULKAN */
