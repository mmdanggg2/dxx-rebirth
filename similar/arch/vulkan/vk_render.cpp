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
#include <cmath>
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
	/* Initialize palette */
	vks_init_palette();
}

/* --- 2D draw helpers -------------------------------------------------------- */

/* Bind the 2D pipeline and set the full-screen viewport, scissor, and
 * pixel->NDC push constant. Canvas-local coordinates are folded into the
 * vertex positions by each caller (adding canvas.cv_bitmap.bm_x/y), matching
 * the OpenGL backend which normalizes absolute coordinates against the full
 * screen via glOrtho(0,1). */
static bool vks_prepare_2d()
{
	if (!vks_ensure_frame())
		return false;
	VkCommandBuffer cmd = vks_get_command_buffer();
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_2d_pipeline);
	const float w = static_cast<float>(last_width);
	const float h = static_cast<float>(last_height);
	const VkViewport viewport{0.0f, 0.0f, w, h, 0.0f, 1.0f};
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	const VkRect2D scissor{{0, 0}, {last_width, last_height}};
	vkCmdSetScissor(cmd, 0, 1, &scissor);
	const float push[4] = {2.0f / w, 2.0f / h, -1.0f, -1.0f};
	vkCmdPushConstants(cmd, vk_2d_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), push);
	return true;
}

/* Allocate `count` vertices from the current frame's buffer, copy them in,
 * bind the descriptor set + vertex buffer, and draw. */
static void vks_emit(VkDescriptorSet ds, const vks_vertex *src, uint32_t count)
{
	VkCommandBuffer cmd = vks_get_command_buffer();
	vks_vertex_alloc alloc = vks_alloc_vertices(count);
	std::memcpy(alloc.vertices, src, count * sizeof(vks_vertex));
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_2d_pipeline_layout, 0, 1, &ds, 0, nullptr);
	VkDeviceSize offset = alloc.offset;
	vkCmdBindVertexBuffers(cmd, 0, 1, &alloc.buffer, &offset);
	vkCmdDraw(cmd, count, 1, 0, 0);
}

/* gr_flip — the main display function. Closes and submits the current frame's
 * command buffer, presents the swapchain image, and advances to the next
 * in-flight frame. All draw submission/synchronization lives in
 * vks_present_frame() (vulkan_init.cpp). */
void gr_flip(void)
{
#if DXX_USE_STEREOSCOPIC_RENDER
	/* Handle stereo rendering */
	if (VR_stereo != StereoFormat::None) {
		// Simplified — would need proper stereo implementation
	}
#endif

	vks_present_frame();
}

/* Pixel drawing */
void vks_upixelc(const grs_bitmap &/*cv_bitmap*/, unsigned x, unsigned y, const color_palette_index c)
{
	const auto &col = vks_palette_colors[c];
	const float fx = static_cast<float>(x);
	const float fy = static_cast<float>(y);
	const float cr = col[0], cg = col[1], cb = col[2], ca = col[3];
	const vks_vertex v[6] = {
		{fx,     fy,     0.f, 0.f, cr, cg, cb, ca},
		{fx + 1.f, fy,     0.f, 0.f, cr, cg, cb, ca},
		{fx + 1.f, fy + 1.f, 0.f, 0.f, cr, cg, cb, ca},
		{fx,     fy,     0.f, 0.f, cr, cg, cb, ca},
		{fx + 1.f, fy + 1.f, 0.f, 0.f, cr, cg, cb, ca},
		{fx,     fy + 1.f, 0.f, 0.f, cr, cg, cb, ca},
	};
	if (!vks_prepare_2d())
		return;
	vks_emit(vk_white_descriptor_set, v, 6);
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
	const float ox = static_cast<float>(canvas.cv_bitmap.bm_x);
	const float oy = static_cast<float>(canvas.cv_bitmap.bm_y);
	const float x0 = left + ox, y0 = top + oy;
	const float x1 = right + ox, y1 = bot + oy;
	const auto &col = vks_palette_colors[c];
	const float cr = col[0], cg = col[1], cb = col[2], ca = col[3];
	const vks_vertex v[6] = {
		{x0, y0, 0.f, 0.f, cr, cg, cb, ca},
		{x1, y0, 0.f, 0.f, cr, cg, cb, ca},
		{x1, y1, 0.f, 0.f, cr, cg, cb, ca},
		{x0, y0, 0.f, 0.f, cr, cg, cb, ca},
		{x1, y1, 0.f, 0.f, cr, cg, cb, ca},
		{x0, y1, 0.f, 0.f, cr, cg, cb, ca},
	};
	if (!vks_prepare_2d())
		return;
	vks_emit(vk_white_descriptor_set, v, 6);
}

/* Bitmap drawing. The 2-arg (array) overload is the real implementation; the
 * int and fill variants delegate to it (stereo is not implemented). */
bool vks_ubitmapm_cs(grs_canvas &canvas, int x, int y, int dw, int dh, grs_bitmap &bm, int c)
{
	vks_colors colors;
	return vks_ubitmapm_cs(canvas, x, y, dw, dh, bm, colors.init(c));
}

bool vks_ubitmapm_cs(grs_canvas &canvas, int x0, int y0, int dw, int dh, grs_bitmap &bm, const vks_colors::array_type &c, bool /*fill*/)
{
	return vks_ubitmapm_cs(canvas, x0, y0, dw, dh, bm, c);
}

bool vks_ubitmapm_cs(grs_canvas &canvas, const int x0, const int y0, const int dw, const int dh, grs_bitmap &bm, const vks_colors::array_type &color_array)
{
	/* Upload on demand (mirrors ogl_bindbmtex). The upload walks to the root
	 * bitmap, so the live texture hangs off the root — look it up there. */
	grs_bitmap *root = &bm;
	while (root->bm_parent)
		root = root->bm_parent;
	if (!root->vktexture)
		vks_loadbmtexture_f(bm, vulkan_texture_filter::classic, false, false);
	vks_texture *tex = root->vktexture;
	if (!tex)
		return false;

	/* Destination size: -1 => fill canvas, 0 => source bitmap size. */
	const int ew = (dw == vulkan_bitmap_use_dst_canvas) ? canvas.cv_bitmap.bm_w
		: (dw == 0) ? bm.bm_w : dw;
	const int eh = (dh == vulkan_bitmap_use_dst_canvas) ? canvas.cv_bitmap.bm_h
		: (dh == 0) ? bm.bm_h : dh;

	const float ox = static_cast<float>(canvas.cv_bitmap.bm_x);
	const float oy = static_cast<float>(canvas.cv_bitmap.bm_y);
	const float xa = x0 + ox, ya = y0 + oy;
	const float xb = xa + ew, yb = ya + eh;

	/* UVs map bm's sub-rectangle into the root-sized texture image. */
	const float tw = static_cast<float>(tex->width);
	const float th = static_cast<float>(tex->height);
	const float u0 = bm.bm_x / tw;
	const float u1 = (bm.bm_x + bm.bm_w) / tw;
	const float v0 = bm.bm_y / th;
	const float v1 = (bm.bm_y + bm.bm_h) / th;

	const float cr = color_array[0], cg = color_array[1], cb = color_array[2], ca = color_array[3];
	const vks_vertex verts[6] = {
		{xa, ya, u0, v0, cr, cg, cb, ca},
		{xb, ya, u1, v0, cr, cg, cb, ca},
		{xb, yb, u1, v1, cr, cg, cb, ca},
		{xa, ya, u0, v0, cr, cg, cb, ca},
		{xb, yb, u1, v1, cr, cg, cb, ca},
		{xa, yb, u0, v1, cr, cg, cb, ca},
	};
	if (!vks_prepare_2d())
		return false;
	vks_emit(tex->descriptor_set, verts, 6);
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
	const auto &col = vks_palette_colors[c];
	const float cr = col[0], cg = col[1], cb = col[2], ca = col[3];
	const float ox = static_cast<float>(canvas.cv_bitmap.bm_x);
	const float oy = static_cast<float>(canvas.cv_bitmap.bm_y);
	const float x0 = left + ox, y0 = top + oy;
	const float x1 = right + ox, y1 = bot + oy;

	float dx = x1 - x0, dy = y1 - y0;
	const float len = std::sqrt(dx * dx + dy * dy);
	if (len < 0.5f) {
		/* Degenerate: render a single pixel. */
		const vks_vertex v[6] = {
			{x0, y0, 0.f, 0.f, cr, cg, cb, ca},
			{x0 + 1.f, y0, 0.f, 0.f, cr, cg, cb, ca},
			{x0 + 1.f, y0 + 1.f, 0.f, 0.f, cr, cg, cb, ca},
			{x0, y0, 0.f, 0.f, cr, cg, cb, ca},
			{x0 + 1.f, y0 + 1.f, 0.f, 0.f, cr, cg, cb, ca},
			{x0, y0 + 1.f, 0.f, 0.f, cr, cg, cb, ca},
		};
		if (!vks_prepare_2d())
			return;
		vks_emit(vk_white_descriptor_set, v, 6);
		return;
	}
	/* Unit perpendicular, half-width 0.5 => 1px-thick line. */
	const float px = -dy / len * 0.5f;
	const float py = dx / len * 0.5f;
	const vks_vertex v[6] = {
		{x0 + px, y0 + py, 0.f, 0.f, cr, cg, cb, ca},
		{x1 + px, y1 + py, 0.f, 0.f, cr, cg, cb, ca},
		{x1 - px, y1 - py, 0.f, 0.f, cr, cg, cb, ca},
		{x0 + px, y0 + py, 0.f, 0.f, cr, cg, cb, ca},
		{x1 - px, y1 - py, 0.f, 0.f, cr, cg, cb, ca},
		{x0 - px, y0 - py, 0.f, 0.f, cr, cg, cb, ca},
	};
	if (!vks_prepare_2d())
		return;
	vks_emit(vk_white_descriptor_set, v, 6);
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

/* Font string rendering (vks_internal_string) lives in font.cpp, alongside
 * ogl_internal_string, so it can reuse the shared glyph-layout helpers
 * (FONTSCALE_X, get_char_width, font_character_extent, ...). */

/* vks_init_font lives in font.cpp (it needs the glyph-atlas helpers). */

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
