/*
 * This file is part of the DXX-Rebirth project <https://www.dxx-rebirth.com/>.
 * It is copyright by its individual contributors, as recorded in the
 * project's Git history.  See COPYING.txt at the top level for license
 * terms and a link to the Git history.
 */

#ifndef _VULKAN_INIT_H_
#define _VULKAN_INIT_H_

#include "dxxsconf.h"

#if !DXX_USE_VULKAN
#error "This file can only be included in Vulkan enabled builds."
#endif

#include "d_vk.h"
#include "dsx-ns.h"
#include "fwd-gr.h"
#include "fwd-segment.h"
#include "palette.h"
#include "pstypes.h"
#include "3d.h"
#include "internal.h"
#include "vulkan_textures.h"
#include <array>
#include <span>

constexpr int vulkan_bitmap_use_dst_canvas = -1;

namespace dcx {

/* SDL window handle */
extern SDL_Window *g_pRebirthVulkanWindow;

/* Vulkan renderer state */
extern VkInstance vk_instance;
extern VkPhysicalDevice vk_physical_device;
extern VkDevice vk_device;
extern VkQueue vk_graphics_queue;
extern VkSurfaceKHR vk_surface;
extern VkSwapchainKHR vk_swapchain;
extern std::vector<VkImageView> vk_swapchain_image_views;
extern std::vector<VkImage> vk_swapchain_images;

extern uint32_t vk_graphics_queue_family;
extern uint32_t vk_surface_family;
extern VkExtent2D vk_surface_extent;
extern VkFormat vk_swapchain_format;
extern VkFormat vk_depth_format;

/* Pipeline and render pass */
extern VkPipeline vk_render_pipeline;
extern VkRenderPass vk_render_pass;
extern VkDescriptorSetLayout vk_descriptor_set_layout;
extern VkPipelineLayout vk_pipeline_layout;

/* Shader modules */
extern VkShaderModule vk_vertex_shader;
extern VkShaderModule vk_fragment_shader;

/* Framebuffers, command buffers, and command pools */
extern std::vector<VkFramebuffer> vk_framebuffers;
extern std::vector<VkCommandBuffer> vk_command_buffers;
extern std::vector<VkCommandPool> vk_command_pools;
extern std::vector<VkSemaphore> vk_image_available_semaphores;
extern std::vector<VkSemaphore> vk_render_finished_semaphores;
extern std::vector<VkFence> vk_in_flight_fences;

/* Current frame index */
extern uint32_t vk_current_frame;

/* Initialization */
void vks_init_instance(SDL_Window *sdl_window);
void vks_init_physical_device();
void vks_init_device();
void vks_init_surface(SDL_Window *window_handle);
void vks_init_swapchain(uint32_t width, uint32_t height);
void vks_init_swapchain_image_views();
void vks_record_initial_barriers();
void vks_destroy_swapchain_image_views();
void vks_init_render_pass();
void vks_init_pipeline();
void vks_init_framebuffers(uint32_t width, uint32_t height);
void vks_init_command_buffers();
void vks_init_sync_objects();
void vks_init_state();

/* Depth resources */
extern VkImage vk_depth_image;
extern VkDeviceMemory vk_depth_image_memory;
extern VkImageView vk_depth_image_view;
void initDepthResources();
void initVertexBuffers();


/* Shutdown */
void vks_shutdown();

/* Frame management */
void vks_start_frame(grs_canvas &);
#if DXX_USE_STEREOSCOPIC_RENDER
void vks_stereo_frame(bool left_eye, int xoff);
#endif
void vks_end_frame();
void vks_set_screen_mode();
int vks_acquire_next_image();
void vks_wait_for_flight_fence(uint32_t frame);

/* Color palette conversion (Vulkan equivalent of ogl_colors) */
struct vks_colors
{
	using array_type = std::array<float, 4>;
	static const array_type white;
	const array_type &init(int c);
	const array_type &init_palette(unsigned c);
private:
	const array_type &init_maybe_white(int c);
	array_type a;
};

/* 2D drawing primitives */
void vks_urect(grs_canvas &, int left, int top, int right, int bot, color_palette_index color);
bool vks_ubitmapm_cs(grs_canvas &, int x, int y, int dw, int dh, grs_bitmap &bm, int c);
bool vks_ubitmapm_cs(grs_canvas &, int x, int y, int dw, int dh, grs_bitmap &bm, const vks_colors::array_type &c);
bool vks_ubitmapm_cs(grs_canvas &, int x, int y, int dw, int dh, grs_bitmap &bm, const vks_colors::array_type &c, bool fill);
bool vks_ubitblt_cs(grs_canvas &, int dw, int dh, int dx, int dy, int sx, int sy);
bool vks_ubitblt_i(unsigned dw, unsigned dh, unsigned dx, unsigned dy, unsigned sw, unsigned sh, unsigned sx, unsigned sy, const grs_bitmap &src, grs_bitmap &dest, vulkan_texture_filter texfilt);
bool vks_ubitblt(unsigned w, unsigned h, unsigned dx, unsigned dy, unsigned sx, unsigned sy, const grs_bitmap &src, grs_bitmap &dest);
void vks_upixelc(const grs_bitmap &, unsigned x, unsigned y, color_palette_index c);
color_palette_index vks_ugpixel(const grs_bitmap &bitmap, unsigned x, unsigned y);
void vks_ulinec(grs_canvas &, int left, int top, int right, int bot, int c);


/* UI elements */
void vks_draw_vertex_reticle(grs_canvas &, int cross, int primary, int secondary, int color, int alpha, int size_offs);
void vks_set_blending(gr_blend);

/* Font rendering (Vulkan equivalent of ogl_internal_string) */
void vks_internal_string(grs_canvas &, const grs_font &cv_font, int entry_x, int yy, const char *const s);
void vks_init_font(const grs_font *font);

} /* namespace dcx */

#ifdef DXX_BUILD_DESCENT
namespace dsx {
void vks_cache_level_textures();
}
#endif

#endif /* _VULKAN_INIT_H_ */
