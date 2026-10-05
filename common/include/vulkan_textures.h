/*
 * This file is part of the DXX-Rebirth project <https://www.dxx-rebirth.com/>.
 * It is copyright by its individual contributors, as recorded in the
 * project's Git history.  See COPYING.txt at the top level for license
 * terms and a link to the Git history.
 */

#ifndef _VULKAN_TEXTURES_H_
#define _VULKAN_TEXTURES_H_

#include "dxxsconf.h"
#if !DXX_USE_VULKAN
#error "This file can only be included in Vulkan enabled builds."
#endif

#include "d_vk.h"
#include "gr.h"
#include <vulkan/vulkan_core.h>

namespace dcx {

/* Vulkan texture wrapper — the Vulkan equivalent of ogl_texture.
 * Holds a VkImage, its view, and descriptor sets binding the view with the
 * shared sampler for its filter mode, one per wrap mode.
 */
struct vks_texture
{
	VkImage image{};
	VkImageView view{};
	VkDeviceMemory memory{};
	/* Size of the source bitmap in texels; the image itself may be larger
	 * (upscale filtering), so texture coordinates use these. */
	uint32_t width{};
	uint32_t height{};
	/* Descriptor sets binding this texture's view+sampler, written at upload
	 * time and bound before any draw that samples it: repeating for mine
	 * geometry (ogl_texwrap GL_REPEAT), clamped for 2D blits and sprites
	 * (GL_CLAMP_TO_EDGE). */
	VkDescriptorSet descriptor_set{};
	VkDescriptorSet descriptor_set_clamp{};
	/* Pixel data of the root bitmap this texture was uploaded from, or
	 * nullptr for a temporary texture or one already released. grs_bitmap is
	 * copied by value (e.g. piggy aliases one GameBitmaps entry to another),
	 * so several bitmaps can hold the same vktexture; a bitmap only uses or
	 * frees it while this still names its own data. */
	const color_palette_index *owner_data{};
};

/* Upper bound on simultaneously live textures; sizes the texture pool and the
 * descriptor pool. Needs headroom over the steady-state working set (D2 levels
 * reference up to ~2600 bitmaps, plus fonts, HUD art and texmerge results)
 * because destruction is deferred until no frame can reference a texture. */
constexpr uint32_t VKS_MAX_TEXTURES = 8192;
/* Descriptor sets per texture (repeat + clamp). */
constexpr uint32_t VKS_DESCRIPTOR_SETS_PER_TEXTURE = 2;

/* Texture filter enum — mirrors opengl_texture_filter */
enum class vulkan_texture_filter : uint8_t
{
	classic,
	upscale,
	trilinear,
};

/* Texture management */
/* The live texture for `bm` (resolved through its root bitmap), uploading it
 * on first use with the configured filtering, as ogl_bindbmtex does.
 * `edgepad` bleeds colour into transparent texels when filtering, as
 * ogl_loadtexture does for overlays and polygon models. nullptr if no
 * texture can be created. */
vks_texture *vks_get_bmtexture(grs_bitmap &bm, bool edgepad);
/* Upload the `w`x`h` region of `src` at (`sx`,`sy`) through gr_current_pal
 * into a texture not attached to any bitmap (the Vulkan counterpart of the
 * scratch texture in ogl_ubitblt_i). Release it with vks_free_texture. */
vks_texture *vks_load_temporary_texture(const grs_bitmap &src, uint32_t sx, uint32_t sy, uint32_t w, uint32_t h, vulkan_texture_filter texfilt);
/* Destroy a texture once no recorded or in-flight frame can reference it. */
void vks_free_texture(vks_texture &tex);
void vks_freebmtexture(grs_bitmap &bm);
/* Release every bitmap texture so each re-uploads on next use, e.g. after
 * the texture filtering settings change (ogl_smash_texture_list_internal). */
void vks_release_bitmap_textures();
/* Destroy textures released before frame slot `frame` was last submitted
 * (call at frame begin, after that slot's fence wait). */
void vks_flush_pending_texture_frees(uint32_t frame);
/* End the current frame slot's recorded texture uploads and return their
 * command buffer (VK_NULL_HANDLE if none), to submit ahead of the frame. */
VkCommandBuffer vks_end_pending_uploads();

/* Destroy all Vulkan resources (called during shutdown) */
void vks_shutdown_textures();

/* 1x1 opaque-white texture + descriptor set, bound for flat 2D primitives. */
void vks_init_white_texture();
void vks_destroy_white_texture();

} /* namespace dcx */

#endif /* _VULKAN_TEXTURES_H_ */
