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
/* maxSamplerAnisotropy when the samplerAnisotropy feature is enabled, else 0. */
extern float vk_max_sampler_anisotropy;
extern VkExtent2D vk_surface_extent;
extern VkFormat vk_swapchain_format;
extern VkFormat vk_depth_format;

/* Pipeline and render pass */
extern VkPipeline vk_2d_pipeline;
/* 2D full-screen palette-flash overlay pipelines (ogl_do_palfx): additive
 * (ONE, ONE) and darkening (ZERO, ONE_MINUS_SRC_COLOR). */
extern VkPipeline vk_2d_pipeline_additive;
extern VkPipeline vk_2d_pipeline_darken;
extern VkRenderPass vk_render_pass;
extern VkDescriptorSetLayout vk_descriptor_set_layout;
extern VkPipelineLayout vk_2d_pipeline_layout;
extern VkPipeline vk_3d_pipeline;
extern VkPipeline vk_3d_pipeline_additive_a;
extern VkPipeline vk_3d_pipeline_additive_c;
extern VkPipeline vk_3d_line_pipeline;
/* Current 3D blend mode (set via gr_settransblend); selects the 3D pipeline
 * so additive draws (weapon cores, explosions, glows) combine rather than
 * replace. Mirrors ogl_set_blending. */
void vks_set_blend(gr_blend b);
gr_blend vks_get_blend();
/* Push-constant offset of the fragment shader's alpha-test reference. */
constexpr uint32_t VKS_PUSH_ALPHA_REF_OFFSET = 16;
/* Fragments whose alpha falls below `ref` are discarded; the counterpart of
 * glAlphaFunc(GL_GEQUAL, ref). The default reference is 0.02. */
void vks_set_alpha_test(float ref);
extern VkDescriptorPool vk_descriptor_pool;

/* 1x1 white texture descriptor set, bound for flat primitives (rect/line/pixel)
 * so a single textured pipeline serves every 2D draw. */
extern VkDescriptorSet vk_white_descriptor_set;

/* Shader modules */
extern VkShaderModule vk_vertex_shader;
extern VkShaderModule vk_fragment_shader;
extern VkShaderModule vk_3d_vertex_shader;

/* Framebuffers, command buffers, and command pools */
extern std::vector<VkFramebuffer> vk_framebuffers;
extern std::vector<VkCommandBuffer> vk_command_buffers;
extern std::vector<VkCommandPool> vk_command_pools;
extern std::vector<VkSemaphore> vk_image_available_semaphores;
extern std::vector<VkSemaphore> vk_present_semaphores;
extern std::vector<VkFence> vk_in_flight_fences;

/* Current in-flight frame index (cycles 0..MAX_FRAMES-1, owns fence/semaphores/cmd buffer) */
extern uint32_t vk_current_frame;
/* Index of the swapchain image acquired for the current frame (differs from vk_current_frame) */
extern uint32_t vk_image_index;

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
void vks_init_descriptor_pool();
void vks_destroy_descriptor_pool();

/* Depth resources: one image per swapchain image, so concurrent frames
 * rendering to different swapchain images never alias a depth attachment. */
extern std::vector<VkImage> vk_depth_images;
extern std::vector<VkDeviceMemory> vk_depth_image_memories;
extern std::vector<VkImageView> vk_depth_image_views;
void initDepthResources();
void destroyDepthResources();

/* Multisample colour resolve resources (allocated only when vk_msaa_samples > 1). */
void initColorResources();
void destroyColorResources();

/* Per-frame vertex buffers (host-visible, persistently mapped). 2D draw
 * functions append vertices into the current frame's buffer; the write cursor
 * resets at the start of each frame once that frame's fence has retired. */
struct vks_vertex
{
	float x, y;       /* absolute screen pixel coordinates */
	float u, v;       /* texture coordinates */
	float r, g, b, a; /* modulating color */
};
/* 3D vertex: viewer-relative position (f2fl), texture coords, color+alpha.
 * 36 bytes (9 floats); shares the per-frame vertex buffer with the 2D vertex
 * via vks_alloc_bytes (raw bytes, so the differing stride is fine). The alpha
 * carries the canvas fade level for sprites (g3_draw_bitmap); opaque (1.0) for
 * every other 3D draw. */
struct vks_vertex3d
{
	float x, y, z; /* viewer-relative position (f2fl) */
	float u, v;    /* texture coordinates (f2fl) */
	float r, g, b; /* modulating light/color */
	float a;       /* vertex alpha (sprite fade; 1.0 = opaque) */
};
struct vks_vertex_alloc
{
	VkBuffer buffer;
	VkDeviceSize offset;
	void *data; /* writable mapped pointer */
};
void vks_init_vertex_buffers();
void vks_destroy_vertex_buffers();
/* Reserve `count` 2D vertices (32 bytes each) in the current frame's buffer. */
vks_vertex_alloc vks_alloc_vertices(uint32_t count);
/* Reserve `bytes` in the current frame's buffer. Used by draw paths whose
 * vertex stride differs from vks_vertex (e.g. 3D). */
vks_vertex_alloc vks_alloc_bytes(uint32_t bytes);
/* The command buffer being recorded for the current in-flight frame. */
VkCommandBuffer vks_get_command_buffer();
/* True while a frame's command buffer is open (between begin and present). */
bool vks_is_frame_recording();


/* Shutdown */
void vks_shutdown();

/* Frame management */
void vks_start_frame(grs_canvas &);
#if DXX_USE_STEREOSCOPIC_RENDER
void vks_stereo_frame(bool left_eye, int xoff);
#endif
/* Begin a frame if none is recording. Returns false if the swapchain is
 * unavailable (out of date); callers must skip drawing when it returns false. */
bool vks_ensure_frame();
void vks_present_frame();
/* Tear down and rebuild swapchain-dependent resources (swapchain, render pass,
 * depth, pipeline, framebuffers, sync objects, vertex buffers) for a new
 * extent. Called on startup, resize, and when the swapchain goes out of date. */
void vks_recreate_swapchain(uint32_t w, uint32_t h);

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
bool vks_ubitblt_i(unsigned dw, unsigned dh, unsigned dx, unsigned dy, unsigned sw, unsigned sh, unsigned sx, unsigned sy, const grs_bitmap &src, grs_bitmap &dest, vulkan_texture_filter texfilt);
void vks_upixelc(const grs_bitmap &, unsigned x, unsigned y, color_palette_index c);
color_palette_index vks_ugpixel(const grs_bitmap &bitmap, unsigned x, unsigned y);
void vks_ulinec(grs_canvas &, int left, int top, int right, int bot, int c);
void _g3_draw_tmap_2(grs_canvas &, std::span<g3_draw_tmap_point *const> pointlist, std::span<const g3s_uvl, 4> uvl_list, std::span<const g3s_lrgb, 4> light_rgb, grs_bitmap &bmbot, grs_bitmap &bm, texture2_rotation_low orient, tmap_drawer_type tmap_drawer_ptr);

/* UI elements */
void vks_draw_vertex_reticle(grs_canvas &, int cross, int primary, int secondary, int color, int alpha, int size_offs);

/* Font rendering (Vulkan equivalent of ogl_internal_string) */
void vks_internal_string(grs_canvas &, const grs_font &cv_font, int entry_x, int yy, const char *const s);
/* vks_init_font is file-local in font.cpp (called from gr_init_font). */

} /* namespace dcx */

template <std::size_t N>
static inline void g3_draw_tmap_2(grs_canvas &canvas, const unsigned nv, const std::array<g3_draw_tmap_point *, N> &pointlist, const std::array<g3s_uvl, N> &uvl_list, const std::array<g3s_lrgb, N> &light_rgb, grs_bitmap &bmbot, grs_bitmap &bm, const texture2_rotation_low orient, const tmap_drawer_type tmap_drawer_ptr)
{
	static_assert(N <= MAX_POINTS_PER_POLY, "too many points in tmap");
#ifdef DXX_CONSTANT_TRUE
	if (DXX_CONSTANT_TRUE(nv > N))
		DXX_ALWAYS_ERROR_FUNCTION("reading beyond array");
#endif
	if (nv > N)
		return;
	_g3_draw_tmap_2(canvas, std::span(pointlist).first(nv), uvl_list, light_rgb, bmbot, bm, orient, tmap_drawer_ptr);
}

#ifdef DXX_BUILD_DESCENT
namespace dsx {
void vks_cache_level_textures();
}
#endif

#endif /* _VULKAN_INIT_H_ */
