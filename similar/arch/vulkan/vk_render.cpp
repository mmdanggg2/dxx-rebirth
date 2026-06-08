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
#include "palette.h"
#include "3d.h"

#include <SDL.h>
#include <cstring>
#include <algorithm>
#include <array>
#include <vector>
#include <span>

namespace dcx {

using std::min;
using std::max;

extern vks_sync vulkan_sync_helper;

/* Screen dimensions */
unsigned last_width = 640;
unsigned last_height = 480;

/* Color conversion (analogous to OpenGL CPAL2Tr/G/B) */
static inline float CPAL2Tr(int c) { return gr_current_pal[c].r / 63.0f; }
static inline float CPAL2Tg(int c) { return gr_current_pal[c].g / 63.0f; }
static inline float CPAL2Tb(int c) { return gr_current_pal[c].b / 63.0f; }

/* Color palette for rendering */
static std::array<std::array<float, 4>, 256> vks_palette_colors;

static void vks_init_palette()
{
	for (int i = 0; i < 256; i++) {
		vks_palette_colors[i][0] = gr_current_pal[i].r / 63.0f;
		vks_palette_colors[i][1] = gr_current_pal[i].g / 63.0f;
		vks_palette_colors[i][2] = gr_current_pal[i].b / 63.0f;
		vks_palette_colors[i][3] = 1.0f;
	}
}

void vks_init_state()
{
	/* Select clearing color */
	VkClearValue clearValues[2]{};
	clearValues[0].color = {0.0f, 0.0f, 0.0f, 0.0f};
	clearValues[1].depthStencil = {1.0f, 0};

	/* Initialize palette */
	vks_init_palette();
}

/* Swap buffers */
void vks_swap_buffers_internal(void)
{
	vulkan_sync_helper.before_swap();

	/* SDL_Vulkan_SwapWindow not available in all SDL2 versions.
	 * Vulkan presentation is handled by vkQueuePresentKHR in the render pipeline. */
	(void)g_pRebirthVulkanWindow;

	vulkan_sync_helper.after_swap();
}

/* gr_flip — the main display function */
void gr_flip(void)
{
#if DXX_USE_STEREOSCOPIC_RENDER
	/* Handle stereo rendering */
	if (VR_stereo != StereoFormat::None) {
		// Simplified — would need proper stereo implementation
	}
#endif

	vks_swap_buffers_internal();
}

/* Pixel drawing */
void vks_upixelc(const grs_bitmap &cv_bitmap, unsigned x, unsigned y, const color_palette_index c)
{
	/* Vulkan would render this via a draw call with a small quad or point */
	(void)cv_bitmap;
	(void)x;
	(void)y;
	(void)c;
	/* Placeholder: actual implementation would record to command buffer */
}

color_palette_index vks_ugpixel(const grs_bitmap &bitmap, unsigned x, unsigned y)
{
	(void)bitmap;
	(void)x;
	(void)y;
	/* Placeholder: would need readback from swapchain image */
	return 0;
}

/* Rectangle drawing */
void vks_urect(grs_canvas &canvas, int left, int top, int right, int bot, color_palette_index c)
{
	(void)canvas;
	(void)left;
	(void)top;
	(void)right;
	(void)bot;
	(void)c;
	/* Placeholder: would submit a quad draw */
}

/* Bitmap drawing */
bool vks_ubitmapm_cs(grs_canvas &canvas, int x, int y, int dw, int dh, grs_bitmap &bm, int c)
{
	(void)canvas;
	(void)x;
	(void)y;
	(void)dw;
	(void)dh;
	(void)bm;
	(void)c;
	return true;
}

bool vks_ubitmapm_cs(grs_canvas &canvas, int x, int y, int dw, int dh, grs_bitmap &bm, const vks_colors::array_type &c, bool /*fill*/)
{
	(void)canvas;
	(void)x;
	(void)y;
	(void)dw;
	(void)dh;
	(void)bm;
	(void)c;
	return true;
}

bool vks_ubitblt_cs(grs_canvas &canvas, int dw, int dh, int dx, int dy, int sx, int sy)
{
	(void)canvas;
	(void)dw;
	(void)dh;
	(void)dx;
	(void)dy;
	(void)sx;
	(void)sy;
	return true;
}

bool vks_ubitblt_i(unsigned dw, unsigned dh, unsigned dx, unsigned dy, unsigned sw, unsigned sh, unsigned sx, unsigned sy, const grs_bitmap &src, grs_bitmap &dest, vulkan_texture_filter /*texfilt*/)
{
	(void)dw;
	(void)dh;
	(void)dx;
	(void)dy;
	(void)sw;
	(void)sh;
	(void)sx;
	(void)sy;
	(void)src;
	(void)dest;
	return true;
}

bool vks_ubitblt(unsigned w, unsigned h, unsigned dx, unsigned dy, unsigned sx, unsigned sy, const grs_bitmap &src, grs_bitmap &dest)
{
	(void)w;
	(void)h;
	(void)dx;
	(void)dy;
	(void)sx;
	(void)sy;
	(void)src;
	(void)dest;
	return true;
}

/* Line drawing */
void vks_ulinec(grs_canvas &canvas, int left, int top, int right, int bot, int c)
{
	(void)canvas;
	(void)left;
	(void)top;
	(void)right;
	(void)bot;
	(void)c;
	/* Placeholder: would submit a line segment draw */
}

/* 3D texture-mapped polygon drawing */
void _vks_draw_tmap_2(grs_canvas & /*canvas*/, std::span<g3_draw_tmap_point *const> /*pointlist*/,
	std::span<const g3s_uvl, 4> /*uvl_list*/, std::span<const g3s_lrgb, 4> /*light_rgb*/,
	grs_bitmap & /*bmbot*/, grs_bitmap & /*bm*/, texture2_rotation_low /*orient*/, tmap_drawer_type /*tmap_drawer_ptr*/)
{
	/* Placeholder: would submit a textured polygon draw */
}

/* Color palette conversion */
const vks_colors::array_type vks_colors::white = {1.0f, 1.0f, 1.0f, 1.0f};

const vks_colors::array_type &vks_colors::init(int c)
{
	return init_maybe_white(c);
}

const vks_colors::array_type &vks_colors::init_palette(unsigned c)
{
	a[0] = gr_current_pal[c].r / 63.0f;
	a[1] = gr_current_pal[c].g / 63.0f;
	a[2] = gr_current_pal[c].b / 63.0f;
	a[3] = 1.0f;
	return a;
}

const vks_colors::array_type &vks_colors::init_maybe_white(int c)
{
	if (c == -1)
		return white;
	return init_palette(static_cast<unsigned>(c));
}

/* UI elements */
void vks_draw_vertex_reticle(grs_canvas & /*canvas*/, int /*cross*/, int /*primary*/, int /*secondary*/,
	int /*color*/, int /*alpha*/, int /*size_offs*/)
{
	/* Placeholder: would draw the crosshair */
}

void vks_set_blending(gr_blend /*blend*/)
{
	/* Placeholder: would set VkPipelineColorBlendState */
}

/* Palette animation */
void vks_do_palfx(void)
{
	(void)last_width;
	(void)last_height;
	/* Placeholder: handle palette effects */
}

/* Brightness/gamma controls */
static int do_pal_step = 0;
static float last_r = 0, last_g = 0, last_b = 0;

void gr_palette_step_up(int r, int g, int b)
{
	(void)r;
	(void)g;
	(void)b;
	do_pal_step = (r || g || b);
	last_r = r / 63.0f;
	last_g = g / 63.0f;
	last_b = b / 63.0f;
}

void gr_palette_load(const palette_array_t &pal)
{
	copy_bound_palette(gr_current_pal, pal);
	gr_palette_step_up(0, 0, 0);
	reset_computed_colors();
	vks_init_palette();
}

/* Font rendering — Vulkan equivalent of ogl_internal_string */
void vks_internal_string(grs_canvas &canvas, const grs_font &cv_font, int entry_x, int yy, const char *const s)
{
	/* Placeholder: would render text using Vulkan-textured quads */
	(void)canvas;
	(void)cv_font;
	(void)entry_x;
	(void)yy;
	(void)s;
}

void vks_init_font(const grs_font * /*font*/)
{
	/* Placeholder: Vulkan font texture initialization */
}

} /* namespace dcx */

namespace dcx {

/* 3D drawing stubs */
void _g3_draw_poly(grs_canvas &, std::span<g3_draw_tmap_point *const>, uint8_t)
{
}

void _g3_draw_tmap(grs_canvas &, std::span<g3_draw_tmap_point *const>, const g3s_uvl *, const g3s_lrgb *, grs_bitmap &, void (*)(grs_canvas &, const grs_bitmap &, std::span<g3_draw_tmap_point *const>))
{
}

void g3_draw_sphere(grs_canvas &, g3_draw_sphere_point &, fix, uint8_t)
{
}

void g3_draw_line(const g3_draw_line_context &, g3_draw_line_point &, g3_draw_line_point &)
{
}

void g3_draw_line(const g3_draw_line_context &, g3_draw_line_point &, g3_draw_line_point &, temporary_points_t &)
{
}

void g3_draw_bitmap(grs_canvas &, const vms_vector &, fix, fix, grs_bitmap &)
{
}

/* 2D drawing stubs */
void gr_ubitmapm(grs_canvas &canvas, unsigned x, unsigned y, grs_bitmap &bm)
{
	(void)canvas; (void)x; (void)y; (void)bm;
}

void gr_bitmapm(grs_canvas &canvas, unsigned x, unsigned y, const grs_bitmap &bm)
{
	(void)canvas; (void)x; (void)y; (void)bm;
}

int gr_ucircle(grs_canvas &, fix, fix, fix, color_palette_index)
{
	return 0;
}

int gr_disk(grs_canvas &, fix, fix, fix, color_palette_index)
{
	return 0;
}

/* Texture mapping globals */
int Interpolation_method = 0;
int Lighting_on = 0;
unsigned Current_seg_depth = 0;

/* Palette */
void gr_palette_read(palette_array_t &)
{
}

/* Draw mode stubs */
void draw_tmap(grs_canvas &, const grs_bitmap &, std::span<const g3_draw_tmap_point *const>)
{
}

void draw_tmap_flat(grs_canvas &, const grs_bitmap &, std::span<const g3_draw_tmap_point *const>)
{
}

} /* namespace dcx */

namespace dsx {

void vks_cache_level_textures()
{
	/* Placeholder: cache level textures for Vulkan */
}

} /* namespace dsx */

#endif /* DXX_USE_VULKAN */
