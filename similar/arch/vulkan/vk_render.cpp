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
#include "texmap.h"
#include "common/3d/globvars.h"
#include "segment.h"
#include "textures.h"
#include "texmerge.h"
#include "effects.h"
#include "weapon.h"
#include "powerup.h"
#include "laser.h"
#include "player.h"
#include "robot.h"
#include "object.h"
#include "polyobj.h"
#include "piggy.h"
#include "d_levelstate.h"
#include "d_zip.h"
#include "partial_range.h"

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

/* Vertex alpha for a canvas fade level, as every OpenGL draw path computes
 * it: opaque at GR_FADE_OFF, otherwise 1 - fade/(GR_FADE_LEVELS-1). */
static float vks_fade_alpha(const grs_canvas &canvas)
{
	return canvas.cv_fade_level >= GR_FADE_OFF
		? 1.0f
		: 1.0f - static_cast<float>(canvas.cv_fade_level) / (static_cast<float>(GR_FADE_LEVELS) - 1.0f);
}

/* Line width / point size in pixels, scaled with resolution like the OpenGL
 * backend's linedotscale (glLineWidth/glPointSize). */
static float vks_linedotscale()
{
	const auto min_wh = std::min(last_width / 640, last_height / 480);
	return min_wh < 1 ? 1.0f : static_cast<float>(min_wh);
}

/* --- 2D draw helpers -------------------------------------------------------- */

/* Bind the 2D pipeline and set the full-screen viewport, scissor, and
 * pixel->NDC push constant. Canvas-local coordinates are folded into the
 * vertex positions by each caller (adding canvas.cv_bitmap.bm_x/y), matching
 * the OpenGL backend which normalizes absolute coordinates against the full
 * screen via glOrtho(0,1). */
static bool vks_prepare_2d(const VkPipeline pipeline = vk_2d_pipeline)
{
	if (!vks_ensure_frame())
		return false;
	VkCommandBuffer cmd = vks_get_command_buffer();
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
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
	std::memcpy(alloc.data, src, count * sizeof(vks_vertex));
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_2d_pipeline_layout, 0, 1, &ds, 0, nullptr);
	VkDeviceSize offset = alloc.offset;
	vkCmdBindVertexBuffers(cmd, 0, 1, &alloc.buffer, &offset);
	vkCmdDraw(cmd, count, 1, 0, 0);
}

/* Bind the 3D pipeline and set the viewport/scissor to the canvas rect. The 3D
 * vertex shader projects viewer-relative coordinates itself (90-degree
 * perspective), so no push constant is needed. */
static bool vks_prepare_3d(grs_canvas &canvas)
{
	if (!vks_ensure_frame())
		return false;
	VkCommandBuffer cmd = vks_get_command_buffer();
	VkPipeline pipe = vk_3d_pipeline;
	switch (vks_get_blend()) {
		case gr_blend::additive_a: pipe = vk_3d_pipeline_additive_a; break;
		case gr_blend::additive_c: pipe = vk_3d_pipeline_additive_c; break;
		default: break;
	}
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
	const float x = static_cast<float>(canvas.cv_bitmap.bm_x);
	const float y = static_cast<float>(canvas.cv_bitmap.bm_y);
	const float w = static_cast<float>(canvas.cv_bitmap.bm_w);
	const float h = static_cast<float>(canvas.cv_bitmap.bm_h);
	const VkViewport viewport{x, y, w, h, 0.0f, 1.0f};
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	const VkRect2D scissor{{static_cast<int32_t>(canvas.cv_bitmap.bm_x), static_cast<int32_t>(canvas.cv_bitmap.bm_y)},
		{static_cast<uint32_t>(canvas.cv_bitmap.bm_w), static_cast<uint32_t>(canvas.cv_bitmap.bm_h)}};
	vkCmdSetScissor(cmd, 0, 1, &scissor);
	return true;
}

/* Allocate 3D vertices, copy, bind descriptor set + vertex buffer, draw. */
static void vks_emit_3d(VkDescriptorSet ds, const vks_vertex3d *src, uint32_t count)
{
	VkCommandBuffer cmd = vks_get_command_buffer();
	vks_vertex_alloc alloc = vks_alloc_bytes(count * sizeof(vks_vertex3d));
	std::memcpy(alloc.data, src, count * sizeof(vks_vertex3d));
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_2d_pipeline_layout, 0, 1, &ds, 0, nullptr);
	VkDeviceSize offset = alloc.offset;
	vkCmdBindVertexBuffers(cmd, 0, 1, &alloc.buffer, &offset);
	vkCmdDraw(cmd, count, 1, 0, 0);
}

/* Palette flash / brightness state (damage, pickup and invulnerability
 * flashes, the gamma setting). As in the OpenGL backend, it is applied as a
 * full-screen overlay just before the frame is presented. */
static bool do_pal_step;
static float last_r, last_g, last_b;

static int gr_apply_gamma_clamp(const int v)
{
	if (v >= 0)
		return std::max(v + gr_palette_gamma, 0);
	else
		return std::min(v + gr_palette_gamma, 0);
}

void gr_palette_step_up(int r, int g, int b)
{
	last_r = gr_apply_gamma_clamp(r) / 63.0f;
	last_g = gr_apply_gamma_clamp(g) / 63.0f;
	last_b = gr_apply_gamma_clamp(b) / 63.0f;
	do_pal_step = (r || g || b || gr_palette_gamma);
}

/* Mirrors ogl_do_palfx: positive steps add the colour (ONE, ONE); an
 * all-negative step darkens, scaled by 2.5 to match D1/D2, with
 * (ZERO, ONE_MINUS_SRC_COLOR). */
static void vks_do_palfx()
{
	if (!do_pal_step || !vks_is_frame_recording())
		return;
	float r = last_r, g = last_g, b = last_b;
	VkPipeline pipeline;
	if (last_r <= 0 && last_g <= 0 && last_b <= 0)
	{
		r = last_r * -2.5f;
		g = last_g * -2.5f;
		b = last_b * -2.5f;
		pipeline = vk_2d_pipeline_darken;
	}
	else
		pipeline = vk_2d_pipeline_additive;
	const float w = static_cast<float>(last_width);
	const float h = static_cast<float>(last_height);
	const vks_vertex v[6] = {
		{0.f, 0.f, 0.f, 0.f, r, g, b, 1.f},
		{w, 0.f, 0.f, 0.f, r, g, b, 1.f},
		{w, h, 0.f, 0.f, r, g, b, 1.f},
		{0.f, 0.f, 0.f, 0.f, r, g, b, 1.f},
		{w, h, 0.f, 0.f, r, g, b, 1.f},
		{0.f, h, 0.f, 0.f, r, g, b, 1.f},
	};
	if (!vks_prepare_2d(pipeline))
		return;
	vks_emit(vk_white_descriptor_set, v, 6);
}

/* gr_flip — the main display function. Draws the palette-flash overlay, then
 * closes and submits the current frame's command buffer, presents the
 * swapchain image, and advances to the next in-flight frame. All draw
 * submission/synchronization lives in vks_present_frame() (vulkan_init.cpp). */
void gr_flip(void)
{
	vks_do_palfx();
	vks_present_frame();
}

/* Pixel drawing */
void vks_upixelc(const grs_bitmap &cv_bitmap, unsigned x, unsigned y, const color_palette_index c)
{
	const auto &col = vks_palette_colors[c];
	/* Centre a linedotscale-sized square on the pixel, offset by the canvas
	 * origin, as ogl_upixelc's GL_POINTS with glPointSize(linedotscale). */
	const float half = vks_linedotscale() * 0.5f;
	const float cx = static_cast<float>(x + cv_bitmap.bm_x) + 0.5f;
	const float cy = static_cast<float>(y + cv_bitmap.bm_y) + 0.5f;
	const float x0 = cx - half, x1 = cx + half;
	const float y0 = cy - half, y1 = cy + half;
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
	const float x1 = (right + 1) + ox, y1 = (bot + 1) + oy;
	const auto &col = vks_palette_colors[c];
	const float cr = col[0], cg = col[1], cb = col[2];
	/* Cloak/transparency fade: when cv_fade_level is active the software
	 * renderer darkens existing framebuffer pixels via gr_fade_table; like
	 * ogl_urect, approximate that with a semi-transparent fill. */
	const float ca = vks_fade_alpha(canvas);
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

/* Emit a screen-aligned textured quad covering [xa,xb)x[ya,yb) in absolute
 * pixel coordinates with texture coordinates [u0,u1]x[v0,v1]. */
static bool vks_draw_textured_quad(VkDescriptorSet ds, const float xa, const float ya, const float xb, const float yb, const float u0, const float v0, const float u1, const float v1, const vks_colors::array_type &color_array)
{
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
	vks_emit(ds, verts, 6);
	return true;
}

bool vks_ubitmapm_cs(grs_canvas &canvas, const int x0, const int y0, const int dw, const int dh, grs_bitmap &bm, const vks_colors::array_type &color_array)
{
	/* Upload on demand (mirrors ogl_bindbmtex). */
	vks_texture *const tex = vks_get_bmtexture(bm);
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

	return vks_draw_textured_quad(tex->descriptor_set, xa, ya, xb, yb, u0, v0, u1, v1, color_array);
}

/* Blit the `sw`x`sh` region of `src` at (`sx`,`sy`) scaled to `dw`x`dh` at
 * (`dx`,`dy`) of `dest`. Like ogl_ubitblt_i, the source is uploaded to a
 * scratch texture through gr_current_pal on every call (movie frames and other
 * bitmaps whose pixels change between draws), released after the frame. */
bool vks_ubitblt_i(unsigned dw, unsigned dh, unsigned dx, unsigned dy, unsigned sw, unsigned sh, unsigned sx, unsigned sy, const grs_bitmap &src, grs_bitmap &dest, vulkan_texture_filter /*texfilt*/)
{
	vks_texture *const tex = vks_load_temporary_texture(src, sx, sy, sw, sh);
	if (!tex)
		return false;
	const float xa = static_cast<float>(dx + dest.bm_x);
	const float ya = static_cast<float>(dy + dest.bm_y);
	const bool drawn = vks_draw_textured_quad(tex->descriptor_set, xa, ya, xa + dw, ya + dh, 0.f, 0.f, 1.f, 1.f, vks_colors::white);
	vks_free_texture(*tex);
	return drawn;
}

/* Line drawing */

/* Write a `width`-pixel-thick quad (two triangles, 6 vertices) from (x0,y0)
 * to (x1,y1) into `out`, coloured c0 at the first end and c1 at the second
 * (GL_LINES' per-vertex colours). The quad extends half a width past both
 * endpoints so the end pixels are covered. Returns the next free vertex. */
static vks_vertex *vks_line_quad(vks_vertex *out, const float x0, const float y0, const vks_colors::array_type &c0, const float x1, const float y1, const vks_colors::array_type &c1, const float width)
{
	const float half = width * 0.5f;
	const float dx = x1 - x0, dy = y1 - y0;
	const float len = std::sqrt(dx * dx + dy * dy);
	/* Unit direction (any direction for a degenerate line) and its
	 * perpendicular. */
	const float ux = len < 0.5f ? 1.f : dx / len;
	const float uy = len < 0.5f ? 0.f : dy / len;
	const float ex = ux * half, ey = uy * half;
	const float px = -uy * half, py = ux * half;
	const float ax = x0 - ex, ay = y0 - ey;
	const float bx = x1 + ex, by = y1 + ey;
	const vks_vertex a0{ax + px, ay + py, 0.f, 0.f, c0[0], c0[1], c0[2], c0[3]};
	const vks_vertex a1{ax - px, ay - py, 0.f, 0.f, c0[0], c0[1], c0[2], c0[3]};
	const vks_vertex b0{bx + px, by + py, 0.f, 0.f, c1[0], c1[1], c1[2], c1[3]};
	const vks_vertex b1{bx - px, by - py, 0.f, 0.f, c1[0], c1[1], c1[2], c1[3]};
	*out++ = a0;
	*out++ = b0;
	*out++ = b1;
	*out++ = a0;
	*out++ = b1;
	*out++ = a1;
	return out;
}

void vks_ulinec(grs_canvas &canvas, int left, int top, int right, int bot, int c)
{
	const auto &col = vks_palette_colors[c];
	const vks_colors::array_type color{{col[0], col[1], col[2], vks_fade_alpha(canvas)}};
	/* Endpoints at pixel centres, offset by the canvas origin. */
	const float ox = static_cast<float>(canvas.cv_bitmap.bm_x) + 0.5f;
	const float oy = static_cast<float>(canvas.cv_bitmap.bm_y) + 0.5f;
	std::array<vks_vertex, 6> v;
	vks_line_quad(v.data(), left + ox, top + oy, color, right + ox, bot + oy, color, vks_linedotscale());
	if (!vks_prepare_2d())
		return;
	vks_emit(vk_white_descriptor_set, v.data(), v.size());
}

/* Circle outline / filled disk centred on (xc,yc) (canvas-relative, fixed
 * point) with radius r pixels, as the OpenGL gr_ucircle / gr_disk. */
static void vks_circle(grs_canvas &canvas, const fix xc, const fix yc, const fix r, const color_palette_index c, const bool filled)
{
	const auto &col = vks_palette_colors[c];
	const vks_colors::array_type color{{col[0], col[1], col[2], vks_fade_alpha(canvas)}};
	const float cx = f2fl(xc) + canvas.cv_bitmap.bm_x + 0.5f;
	const float cy = f2fl(yc) + canvas.cv_bitmap.bm_y + 0.5f;
	const float rad = f2fl(r);
	const unsigned nsides = 10 + 2 * static_cast<unsigned>(M_PI * rad / 19);
	const auto rim = [=](const unsigned i) {
		const float ang = 2.0f * static_cast<float>(M_PI) * static_cast<float>(i % nsides) / static_cast<float>(nsides);
		return std::array<float, 2>{{cx + rad * cosf(ang), cy + rad * sinf(ang)}};
	};
	std::vector<vks_vertex> v(nsides * (filled ? 3 : 6));
	auto out = v.data();
	const float lw = vks_linedotscale();
	for (unsigned i = 0; i != nsides; ++i)
	{
		const auto p0 = rim(i), p1 = rim(i + 1);
		if (filled)
		{
			*out++ = {cx, cy, 0.f, 0.f, color[0], color[1], color[2], color[3]};
			*out++ = {p0[0], p0[1], 0.f, 0.f, color[0], color[1], color[2], color[3]};
			*out++ = {p1[0], p1[1], 0.f, 0.f, color[0], color[1], color[2], color[3]};
		}
		else
			out = vks_line_quad(out, p0[0], p0[1], color, p1[0], p1[1], color, lw);
	}
	if (!vks_prepare_2d())
		return;
	vks_emit(vk_white_descriptor_set, v.data(), static_cast<uint32_t>(v.size()));
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

/* "Classic Reboot" vector reticle, a port of ogl_draw_vertex_reticle. The
 * shapes are authored in reticle units around the canvas centre (+y up);
 * one unit is `size` fixed-point screen fractions, corrected for aspect so
 * the reticle stays round. */
void vks_draw_vertex_reticle(grs_canvas &canvas, const int cross, const int primary, const int secondary, const int color, const int alpha, const int size_offs)
{
	int size = 270 + (size_offs * 20);
	const float scale = static_cast<float>(SWIDTH) / SHEIGHT;
	const auto &rgb = gr_palette[color];
	const vks_colors::array_type bright{{
		rgb.r / 63.0f,
		rgb.g / 63.0f,
		rgb.b / 63.0f,
		1.0f - (static_cast<float>(alpha) / static_cast<float>(GR_FADE_LEVELS))
	}}, dark{{bright[0] / 2, bright[1] / 2, bright[2] / 2, bright[3] / 2}};

	float sx, sy;
	if (scale >= 1)
	{
		size /= scale;
		sx = f2fl(size);
		sy = f2fl(size * scale);
	}
	else
	{
		size *= scale;
		sx = f2fl(size / scale);
		sy = f2fl(size);
	}
	const float ux = sx * last_width, uy = sy * last_height;
	const float cx = canvas.cv_bitmap.bm_w / 2 + canvas.cv_bitmap.bm_x;
	const float cy = canvas.cv_bitmap.bm_h / 2 + canvas.cv_bitmap.bm_y;
	const auto px = [=](const float x) { return cx + x * ux; };
	const auto py = [=](const float y) { return cy - y * uy; };
	const float lw = vks_linedotscale() * 2;

	/* 4 cross lines + 4 primary bar strips (2 triangles each) + up to two
	 * 16-segment secondary rings. */
	std::array<vks_vertex, 4 * 6 + 4 * 6 + 2 * 16 * 6> v;
	auto out = v.data();

	//cross
	static constexpr std::array<float, 8 * 2> cross_lva{{
		-4.0, 2.0, -2.0, 0, -3.0, -4.0, -2.0, -3.0, 4.0, 2.0, 2.0, 0, 3.0, -4.0, 2.0, -3.0,
	}};
	for (unsigned i = 0; i != cross_lva.size(); i += 4)
		out = vks_line_quad(out, px(cross_lva[i]), py(cross_lva[i + 1]), dark, px(cross_lva[i + 2]), py(cross_lva[i + 3]), cross ? bright : dark, lw);

	/* Primary bars: each a 4-vertex triangle strip; the first two vertices
	 * take `c0`, the last two `c1`. */
	const auto strip = [&](const std::array<float, 4 * 2> &lva, const vks_colors::array_type &c0, const vks_colors::array_type &c1) {
		const auto vtx = [&](const unsigned i, const vks_colors::array_type &c) {
			return vks_vertex{px(lva[i * 2]), py(lva[i * 2 + 1]), 0.f, 0.f, c[0], c[1], c[2], c[3]};
		};
		const auto v0 = vtx(0, c0), v1 = vtx(1, c0), v2 = vtx(2, c1), v3 = vtx(3, c1);
		*out++ = v0;
		*out++ = v1;
		*out++ = v2;
		*out++ = v1;
		*out++ = v3;
		*out++ = v2;
	};
	static constexpr std::array<float, 4 * 2> primary_lva0{{
		-5.5, -5.0, -6.5, -7.5, -10.0, -7.0, -10.0, -8.7
	}};
	static constexpr std::array<float, 4 * 2> primary_lva1{{
		-10.0, -7.0, -10.0, -8.7, -15.0, -8.5, -15.0, -9.5
	}};
	static constexpr std::array<float, 4 * 2> primary_lva2{{
		5.5, -5.0, 6.5, -7.5, 10.0, -7.0, 10.0, -8.7
	}};
	static constexpr std::array<float, 4 * 2> primary_lva3{{
		10.0, -7.0, 10.0, -8.7, 15.0, -8.5, 15.0, -9.5
	}};
	const auto &inner0 = primary == 0 ? dark : bright;
	const auto &inner1 = dark;
	const auto &outer0 = dark;
	const auto &outer1 = primary == 2 ? bright : dark;
	strip(primary_lva0, inner0, inner1);
	strip(primary_lva1, outer0, outer1);
	strip(primary_lva2, inner0, inner1);
	strip(primary_lva3, outer0, outer1);

	/* Secondary indicator: a 16-segment circle of radius 2 units. */
	const auto ring = [&](const float ox, const float oy, const vks_colors::array_type &c) {
		for (unsigned i = 0; i != 16; ++i)
		{
			const float a0 = 2.0f * static_cast<float>(M_PI) * i / 16;
			const float a1 = 2.0f * static_cast<float>(M_PI) * (i + 1) / 16;
			out = vks_line_quad(out, px(cosf(a0) * 2.0f + ox), py(sinf(a0) * 2.0f + oy), c, px(cosf(a1) * 2.0f + ox), py(sinf(a1) * 2.0f + oy), c, lw);
		}
	};
	if (secondary <= 2)
	{
		ring(-10.0f, -2.0f, secondary != 1 ? dark : bright);
		ring(10.0f, -2.0f, secondary != 2 ? dark : bright);
	}
	else
		ring(0.0f, -8.0f, secondary != 4 ? dark : bright);
	if (!vks_prepare_2d())
		return;
	vks_emit(vk_white_descriptor_set, v.data(), static_cast<uint32_t>(out - v.data()));
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
/* draw_tmap / draw_tmap_flat (declared in texmap.h) are sentinels: the mine
 * renderer passes them as tmap_drawer_type function pointers so _g3_draw_tmap
 * can distinguish textured vs. cloaked faces. They are defined (as no-ops)
 * below. */

/* 3D polygon drawing (mine walls, objects). Vertices are viewer-relative
 * g3_rotated_point coords; the 3D pipeline projects them with a 90-degree
 * perspective frustum and writes depth. */
void _g3_draw_poly(grs_canvas &canvas, std::span<g3_draw_tmap_point *const> pointlist, uint8_t color)
{
	const auto nv = pointlist.size();
	if (nv < 3 || nv > MAX_POINTS_PER_POLY)
		return;
	if (!vks_prepare_3d(canvas))
		return;
	/* Flat-shaded: every vertex gets the palette color (gr_palette, as the
	 * OpenGL PAL2T) and the canvas fade alpha (cloaked walls); UVs unused. */
	const auto &rgb = gr_palette[color];
	const float cr = rgb.r / 63.0f, cg = rgb.g / 63.0f, cb = rgb.b / 63.0f;
	const float ca = vks_fade_alpha(canvas);
	std::array<vks_vertex3d, MAX_POINTS_PER_POLY> verts;
	for (uint32_t i = 0; i < nv; i++) {
		const auto &pv = pointlist[i]->p3_vec;
		verts[i] = {f2fl(pv.x), f2fl(pv.y), f2fl(pv.z), 0.f, 0.f, cr, cg, cb, ca};
	}
	vks_emit_3d(vk_white_descriptor_set, verts.data(), static_cast<uint32_t>(nv));
}

void _g3_draw_tmap(grs_canvas &canvas, std::span<g3_draw_tmap_point *const> pointlist, const g3s_uvl *const uvl_list, const g3s_lrgb *const light_rgb, grs_bitmap &bm, const tmap_drawer_type tmap_drawer_ptr)
{
	const auto nv = pointlist.size();
	if (nv < 3 || nv > MAX_POINTS_PER_POLY)
		return;

	/* draw_tmap => textured; draw_tmap_flat => untextured black silhouette
	 * for cloaked faces. Any other drawer is unsupported, as in OpenGL. */
	const bool textured = (tmap_drawer_ptr == draw_tmap);
	if (!textured && tmap_drawer_ptr != draw_tmap_flat)
		return;
	VkDescriptorSet ds = vk_white_descriptor_set;
	float alpha;
	if (textured)
	{
		/* On-demand upload (mirrors ogl_bindbmtex). */
		vks_texture *const tex = vks_get_bmtexture(bm);
		if (!tex)
			return;
		ds = tex->descriptor_set;
		alpha = vks_fade_alpha(canvas);
	}
	else
		alpha = 1.0f - static_cast<float>(canvas.cv_fade_level) / static_cast<float>(NUM_LIGHTING_LEVELS);
	if (!vks_prepare_3d(canvas))
		return;

	const bool no_light = bm.get_flag_mask(BM_FLAG_NO_LIGHTING);
	std::array<vks_vertex3d, MAX_POINTS_PER_POLY> verts;
	for (uint32_t i = 0; i < nv; i++) {
		const auto &pv = pointlist[i]->p3_vec;
		auto &vt = verts[i];
		vt.x = f2fl(pv.x);
		vt.y = f2fl(pv.y);
		vt.z = f2fl(pv.z);
		vt.u = f2fl(uvl_list[i].u);
		vt.v = f2fl(uvl_list[i].v);
		if (!textured)
			vt.r = vt.g = vt.b = 0.f;		/* cloaked silhouette */
		else if (no_light)
			vt.r = vt.g = vt.b = 1.f;		/* full-bright */
		else {
			vt.r = f2fl(light_rgb[i].r);
			vt.g = f2fl(light_rgb[i].g);
			vt.b = f2fl(light_rgb[i].b);
		}
		vt.a = alpha;
	}
	vks_emit_3d(ds, verts.data(), static_cast<uint32_t>(nv));
}

/* Wall with an overlay texture (tmap_num2): draw the base, then the overlay
 * over the same polygon with its texture coordinates rotated by `orient`, as
 * the OpenGL _g3_draw_tmap_2 does. The overlay pass has the same depth, so
 * the LESS_OR_EQUAL depth test lets it through; transparent overlay texels
 * are discarded. */
void _g3_draw_tmap_2(grs_canvas &canvas, const std::span<g3_draw_tmap_point *const> pointlist, const std::span<const g3s_uvl, 4> uvl_list, const std::span<const g3s_lrgb, 4> light_rgb, grs_bitmap &bmbot, grs_bitmap &bm, const texture2_rotation_low orient, const tmap_drawer_type tmap_drawer_ptr)
{
	_g3_draw_tmap(canvas, pointlist, uvl_list.data(), light_rgb.data(), bmbot, tmap_drawer_ptr);
	std::array<g3s_uvl, 4> rotated;
	for (auto &&[r, uvl] : zip(rotated, uvl_list))
	{
		r.l = uvl.l;
		switch (orient)
		{
			case texture2_rotation_low::_1:
				r.u = F1_0 - uvl.v;
				r.v = uvl.u;
				break;
			case texture2_rotation_low::_2:
				r.u = F1_0 - uvl.u;
				r.v = F1_0 - uvl.v;
				break;
			case texture2_rotation_low::_3:
				r.u = uvl.v;
				r.v = F1_0 - uvl.u;
				break;
			default:
				r.u = uvl.u;
				r.v = uvl.v;
				break;
		}
	}
	_g3_draw_tmap(canvas, pointlist, rotated.data(), light_rgb.data(), bm, draw_tmap);
}

void g3_draw_sphere(grs_canvas &canvas, g3_draw_sphere_point &pnt, fix rad, uint8_t color)
{
	/* Untextured filled circle ("sphere") facing the viewer, flat at the
	 * point's depth -- mirrors ogl_drawcircle(20, GL_TRIANGLE_FAN) in the
	 * OpenGL backend. Drawn through the 3D pipeline with the white texture
	 * (so fragment colour = vertex colour); a 20-gon triangle fan (centre +
	 * rim, with the rim closed back to its first point) fills it. */
	if (!vks_prepare_3d(canvas))
		return;
	const float cx = f2fl(pnt.p3_vec.x);
	const float cy = f2fl(pnt.p3_vec.y);
	const float cz = f2fl(pnt.p3_vec.z);
	/* The projection maps view x across the canvas width and y across its
	 * height, so correct the radii by the canvas aspect to draw a round disc,
	 * exactly as the OpenGL g3_draw_sphere's glScalef(gl1, gl2, ...). */
	const float scale = static_cast<float>(canvas.cv_bitmap.bm_w) / static_cast<float>(canvas.cv_bitmap.bm_h);
	const float rad_f = f2fl(rad);
	const float rx = scale >= 1.0f ? rad_f / scale : rad_f;
	const float ry = scale >= 1.0f ? rad_f : rad_f * scale;
	const float cr = CPAL2Tr(color), cg = CPAL2Tg(color), cb = CPAL2Tb(color);
	constexpr unsigned nsides = 20;
	std::array<vks_vertex3d, nsides + 2> verts;
	verts[0] = {cx, cy, cz, 0.f, 0.f, cr, cg, cb, 1.f};
	for (unsigned i = 0; i <= nsides; ++i) {
		const float ang = 2.0f * 3.14159265358979323846f * static_cast<float>(i) / static_cast<float>(nsides);
		verts[i + 1] = {cx + rx * cosf(ang), cy + ry * sinf(ang), cz, 0.f, 0.f, cr, cg, cb, 1.f};
	}
	vks_emit_3d(vk_white_descriptor_set, verts.data(), static_cast<uint32_t>(verts.size()));
}

void g3_draw_line(const g3_draw_line_context &context, g3_draw_line_point &p0, g3_draw_line_point &p1)
{
	/* 3D wireframe line. Endpoints arrive as rotated viewer-space points
	 * (p3_vec); emit them through the 3D pipeline as a LINE_LIST, mirroring
	 * the OpenGL backend's GL_LINES. The per-vertex colour is pre-baked in
	 * context.color_array by g3_draw_line_colors. */
	if (!vks_prepare_3d(context.canvas))
		return;
	vkCmdBindPipeline(vks_get_command_buffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, vk_3d_line_pipeline);
	const auto &ca = context.color_array;
	const vks_vertex3d verts[2] = {
		{f2fl(p0.p3_vec.x), f2fl(p0.p3_vec.y), f2fl(p0.p3_vec.z), 0.f, 0.f, ca[0], ca[1], ca[2], 1.f},
		{f2fl(p1.p3_vec.x), f2fl(p1.p3_vec.y), f2fl(p1.p3_vec.z), 0.f, 0.f, ca[4], ca[5], ca[6], 1.f},
	};
	vks_emit_3d(vk_white_descriptor_set, verts, 2);
}

void g3_draw_line(const g3_draw_line_context &context, g3_draw_line_point &p0, g3_draw_line_point &p1, temporary_points_t &)
{
	g3_draw_line(context, p0, p1);
}

void g3_draw_bitmap(grs_canvas &canvas, const vms_vector &pos, const fix iwidth, const fix iheight, grs_bitmap &bm)
{
	/* 2d Sprites (fireballs, powerups, explosions, blob weapons): a textured
	 * billboard that always faces the viewer. The center is rotated into
	 * viewer-relative space, then four corners are offset by ±width/±height in
	 * viewer x/y — flat at the center's depth — exactly the coordinates the 3D
	 * pipeline projects. Mirrors the OpenGL backend's g3_draw_bitmap. */
	g3s_point pnt;
	if ((g3_rotate_point(pnt, pos) & clipping_code::behind) != clipping_code::None)
		return;

	/* On-demand upload (mirrors ogl_bindbmtex). */
	vks_texture *const tex = vks_get_bmtexture(bm);
	if (!tex)
		return;
	if (!vks_prepare_3d(canvas))
		return;

	const float cx = f2fl(pnt.p3_vec.x);
	const float cy = f2fl(pnt.p3_vec.y);
	const float cz = f2fl(pnt.p3_vec.z);
	const float w = f2fl(fixmul(iwidth, Matrix_scale.x));
	const float h = f2fl(fixmul(iheight, Matrix_scale.y));

	/* UVs: the bm occupies a sub-rect of the root-sized texture image (no POT
	 * padding in the Vulkan path), so normalize against tex->width/height. */
	const float tw = static_cast<float>(tex->width);
	const float th = static_cast<float>(tex->height);
	const float u0 = static_cast<float>(bm.bm_x) / tw;
	const float u1 = static_cast<float>(bm.bm_x + bm.bm_w) / tw;
	const float v0 = static_cast<float>(bm.bm_y) / th;
	const float v1 = static_cast<float>(bm.bm_y + bm.bm_h) / th;

	/* Sprite fade: dim the billboard with the canvas fade level, mirroring the
	 * OpenGL backend (cv_fade_level < GR_FADE_OFF => semi-transparent). White
	 * vertex colour: the sprite texture carries its own colour (no lighting
	 * modulation). +y viewer = up on screen; v=0 is the top row of the bitmap,
	 * so the sprite appears upright. Triangle fan: TL, TR, BR, BL. */
	const float alpha = vks_fade_alpha(canvas);
	const vks_vertex3d verts[4] = {
		{cx - w, cy + h, cz, u0, v0, 1.f, 1.f, 1.f, alpha},
		{cx + w, cy + h, cz, u1, v0, 1.f, 1.f, 1.f, alpha},
		{cx + w, cy - h, cz, u1, v1, 1.f, 1.f, 1.f, alpha},
		{cx - w, cy - h, cz, u0, v1, 1.f, 1.f, 1.f, alpha},
	};
	vks_emit_3d(tex->descriptor_set, verts, 4);
}

/* 2D bitmap drawing */
void gr_ubitmapm(grs_canvas &canvas, unsigned x, unsigned y, grs_bitmap &bm)
{
	/* Masked bitmap blit at native size (gauge icons, weapon indicators).
	 * Delegates to the textured-quad blit; color-key transparency (alpha-0
	 * discard in the fragment shader) handles the masked pixels. */
	vks_ubitmapm_cs(canvas, static_cast<int>(x), static_cast<int>(y), 0, 0, bm, vks_colors::white);
}

int gr_ucircle(grs_canvas &canvas, const fix xc1, const fix yc1, const fix r1, const color_palette_index c)
{
	vks_circle(canvas, xc1, yc1, r1, c, false);
	return 0;
}

int gr_disk(grs_canvas &canvas, const fix x, const fix y, const fix r, const color_palette_index c)
{
	vks_circle(canvas, x, y, r, c, true);
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

/* Texture precaching at level start, a port of ogl_cache_level_textures:
 * upload walls (stepping animated effects through every frame so texmerged
 * and animated sides are covered), powerups, weapons, polymodels and the
 * player's effects up front rather than on first sight mid-frame. */
namespace dsx {

void vks_cache_polymodel_textures(const polygon_model_index model_num)
{
	auto &Polygon_models = LevelSharedPolygonModelState.Polygon_models;
	if (model_num == polygon_model_index::None)
		return;
	const auto &po = Polygon_models[model_num];
	unsigned i = po.first_texture;
	const unsigned last_texture = i + po.n_textures;
	for (; i != last_texture; ++i)
	{
		const auto objbitmap = ObjBitmaps[ObjBitmapPtrs[i]];
		PIGGY_PAGE_IN(objbitmap);
		vks_get_bmtexture(GameBitmaps[objbitmap]);
	}
}

}

namespace {

static void vks_cache_vclip_textures(const vclip &vc)
{
	for (const auto i : partial_const_range(vc.frames, vc.num_frames))
	{
		PIGGY_PAGE_IN(i);
		vks_get_bmtexture(GameBitmaps[i]);
	}
}

static void vks_cache_vclipn_textures(const d_vclip_array &Vclip, const vclip_index i)
{
	if (Vclip.valid_index(i))
		vks_cache_vclip_textures(Vclip[i]);
}

static void vks_cache_weapon_textures(const d_vclip_array &Vclip, const weapon_info_array &Weapon_info, const weapon_id_type weapon_type)
{
	if (weapon_type >= Weapon_info.size())
		return;
	const auto &w = Weapon_info[weapon_type];
	vks_cache_vclipn_textures(Vclip, w.flash_vclip);
	vks_cache_vclipn_textures(Vclip, w.robot_hit_vclip);
	vks_cache_vclipn_textures(Vclip, w.wall_hit_vclip);
	if (w.render == weapon_info::render_type::vclip)
		vks_cache_vclipn_textures(Vclip, w.weapon_vclip);
	else if (w.render == weapon_info::render_type::polymodel)
	{
		vks_cache_polymodel_textures(w.model_num);
		vks_cache_polymodel_textures(w.model_num_inner);
	}
}

}

namespace dsx {

void vks_cache_level_textures()
{
	auto &Effects = LevelUniqueEffectsClipState.Effects;
	auto &Objects = LevelUniqueObjectState.Objects;
	auto &vcobjptridx = Objects.vcptridx;
	int max_efx{0};

	for (auto &ec : partial_const_range(Effects, Num_effects))
	{
		vks_cache_vclipn_textures(Vclip, ec.dest_vclip);
		if (ec.changing_wall_texture == texture_index{UINT16_MAX} && ec.changing_object_texture.dsx == object_bitmap_index::None)
			continue;
		if (ec.vc.num_frames > max_efx)
			max_efx = ec.vc.num_frames;
	}
	for (int ef = 0; ef < max_efx; ef++)
	{
		for (eclip &ec : partial_range(Effects, Num_effects))
		{
			if (ec.changing_wall_texture == texture_index{UINT16_MAX} && ec.changing_object_texture.dsx == object_bitmap_index::None)
				continue;
			ec.time_left = -1;
		}
		do_special_effects();

		for (const unique_segment &seg : vcsegptr)
		{
			for (auto &side : seg.sides)
			{
				const auto tmap1 = side.tmap_num;
				const auto tmap2 = side.tmap_num2;
				const auto tmap1idx = get_texture_index(tmap1);
				if (tmap1idx >= NumTextures || tmap1idx >= Textures.size()) [[unlikely]]
					continue;
				const auto texture1{Textures[tmap1idx]};
				if (!GameBitmaps.valid_index(texture1)) [[unlikely]]
					continue;
				PIGGY_PAGE_IN(texture1);
				grs_bitmap *bm = &GameBitmaps[texture1];
				if (tmap2 != texture2_value::None)
				{
					const auto tmap2idx{get_texture_index(tmap2)};
					if (tmap2idx >= Textures.size()) [[unlikely]]
						continue;
					const auto texture2{Textures[tmap2idx]};
					if (!GameBitmaps.valid_index(texture2))
						continue;
					PIGGY_PAGE_IN(texture2);
					auto &bm2 = GameBitmaps[texture2];
					/* Mirrors render_face: super-transparent overlays are
					 * texmerged, others drawn as a second pass. */
					if (bm2.get_flag_mask(BM_FLAG_SUPER_TRANSPARENT))
						bm = &texmerge_get_cached_bitmap(GameBitmaps, Textures, tmap1, tmap2);
					else
						vks_get_bmtexture(bm2);
				}
				vks_get_bmtexture(*bm);
			}
		}
	}
	reset_special_effects();
	init_special_effects();
	{
		auto &Robot_info = LevelSharedRobotInfoState.Robot_info;
		// always have lasers, concs, flares.  Always shows player appearance, and at least concs are always available to disappear.
		vks_cache_weapon_textures(Vclip, Weapon_info, Primary_weapon_to_weapon_info[primary_weapon_index::laser]);
		vks_cache_weapon_textures(Vclip, Weapon_info, Secondary_weapon_to_weapon_info[secondary_weapon_index::concussion]);
		vks_cache_weapon_textures(Vclip, Weapon_info, weapon_id_type::FLARE_ID);
		vks_cache_vclipn_textures(Vclip, vclip_index::player_appearance);
		vks_cache_vclipn_textures(Vclip, vclip_index::powerup_disappearance);
		vks_cache_polymodel_textures(Player_ship->model_num.dsx);
		vks_cache_vclipn_textures(Vclip, Player_ship->expl_vclip_num);

		for (const auto &&objp : vcobjptridx)
		{
			if (objp->type == object_type::OBJ_POWERUP && objp->render_type == render_type::RT_POWERUP)
			{
				vks_cache_vclipn_textures(Vclip, objp->rtype.vclip_info.vclip_num);
				const auto id = get_powerup_id(objp);
				primary_weapon_index p;
				secondary_weapon_index s;
				weapon_id_type w;
				if (
					(
						(
							(id == powerup_type_t::POW_VULCAN_WEAPON && (p = primary_weapon_index::vulcan, true)) ||
							(id == powerup_type_t::POW_SPREADFIRE_WEAPON && (p = primary_weapon_index::spreadfire, true)) ||
							(id == powerup_type_t::POW_PLASMA_WEAPON && (p = primary_weapon_index::plasma, true)) ||
							(id == powerup_type_t::POW_FUSION_WEAPON && (p = primary_weapon_index::fusion, true))
						) && (w = Primary_weapon_to_weapon_info[p], true)
					) ||
					(
						(
							(id == powerup_type_t::POW_PROXIMITY_WEAPON && (s = secondary_weapon_index::proximity, true)) ||
							((id == powerup_type_t::POW_HOMING_AMMO_1 || id == powerup_type_t::POW_HOMING_AMMO_4) && (s = secondary_weapon_index::homing, true)) ||
							(id == powerup_type_t::POW_SMARTBOMB_WEAPON && (s = secondary_weapon_index::smart, true)) ||
							(id == powerup_type_t::POW_MEGA_WEAPON && (s = secondary_weapon_index::mega, true))
						) && (w = Secondary_weapon_to_weapon_info[s], true)
					)
				)
				{
					vks_cache_weapon_textures(Vclip, Weapon_info, w);
				}
			}
			else if (objp->type != object_type::OBJ_NONE && objp->render_type == render_type::RT_POLYOBJ)
			{
				if (objp->type == object_type::OBJ_ROBOT)
				{
					auto &ri = Robot_info[get_robot_id(objp)];
					vks_cache_vclipn_textures(Vclip, ri.exp1_vclip_num);
					vks_cache_vclipn_textures(Vclip, ri.exp2_vclip_num);
					vks_cache_weapon_textures(Vclip, Weapon_info, ri.weapon_type);
				}
				if (const auto tmap_override{objp->rtype.pobj_info.tmap_override}; tmap_override < Textures.size())
				{
					const auto t{Textures[tmap_override]};
					if (!GameBitmaps.valid_index(t)) [[unlikely]]
						continue;
					PIGGY_PAGE_IN(t);
					vks_get_bmtexture(GameBitmaps[t]);
				}
				else if (tmap_override == texture_index{UINT16_MAX}) [[likely]]
					vks_cache_polymodel_textures(objp->rtype.pobj_info.model_num.dsx);
			}
		}
	}
}

}

#endif /* DXX_USE_VULKAN */
