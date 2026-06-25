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

} /* namespace dcx */

namespace dsx {

void gr_set_mode_from_window_size()
{
	/* Stub — full implementation requires window resize logic */
}

int gr_init()
{
	if (gr_installed)
		return -1;

	/* Create an SDL window for Vulkan surface creation */
	assert(!g_pRebirthVulkanWindow);
	unsigned sdl_window_flags = SDL_WINDOW_VULKAN;
	if (CGameArg.SysNoBorders)
		sdl_window_flags |= SDL_WINDOW_BORDERLESS;
	if (!CGameCfg.WindowMode && !CGameArg.SysWindow)
		sdl_window_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
#if defined(__APPLE__) && defined(__MACH__)
	sdl_window_flags |= SDL_WINDOW_ALLOW_HIGHDPI;
#endif
	const auto mode{Game_screen_mode};
	g_pRebirthVulkanWindow = SDL_CreateWindow(DESCENT_VERSION, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SM_W(mode), SM_H(mode), sdl_window_flags);
	if (!g_pRebirthVulkanWindow) {
		con_printf(CON_URGENT, "Vulkan: SDL_CreateWindow failed: %s", SDL_GetError());
		return -1;
	}
	SDL_GetWindowPosition(g_pRebirthVulkanWindow, &g_iRebirthWindowX, &g_iRebirthWindowY);
	if (const auto window_icon = SDL_LoadBMP(DXX_SDL_WINDOW_ICON_BITMAP))
		SDL_SetWindowIcon(g_pRebirthVulkanWindow, window_icon);

	/* Initialize Vulkan instance (needs window for platform-specific extensions) */
	vks_init_instance(g_pRebirthVulkanWindow);

	/* Create surface from SDL window */
	vks_init_surface(g_pRebirthVulkanWindow);

	/* Initialize physical device */
	vks_init_physical_device();
	vks_init_device();

	/* Find depth format */
	find_depth_format();

	/* Create descriptor set layout */
	init_descriptor_set_layout();

	/* Descriptor pool — persists across resize (textures outlive swapchain) */
	vks_init_descriptor_pool();

	/* Set up sync helper */
	vulkan_sync_helper.init();

	grd_curscreen = std::make_unique<grs_screen>();
	*grd_curscreen = {};
	grd_curscreen->sc_canvas.cv_bitmap.bm_data = NULL;

	// Set the mode.
	grd_curscreen->sc_canvas.cv_fade_level = GR_FADE_OFF;
	grd_curscreen->sc_canvas.cv_font = NULL;
	grd_curscreen->sc_canvas.cv_font_fg_color = 0;
	grd_curscreen->sc_canvas.cv_font_bg_color = 0;
	gr_set_current_canvas(grd_curscreen->sc_canvas);

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


	vks_recreate_swapchain(w, h);

	/* Initialize rendering state */
	vks_init_state();
	gamefont_choose_game_font(w, h);
	gr_remap_color_fonts();

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
		vks_shutdown_textures();
		vks_shutdown();
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
