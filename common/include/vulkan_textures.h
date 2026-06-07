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
 * Holds aVk image, its view, and staging resources for upload.
 */
struct vks_texture
{
	VkImage image{};
	VkImageView view{};
	VkDeviceMemory memory{};
	VkFormat format{};
	VkImageLayout layout{};
	uint32_t width{};
	uint32_t height{};
	uint32_t mip_levels{};
	VkSampler sampler{};
	VkBuffer staging_buffer{};
	VkDeviceMemory staging_memory{};
	size_t staging_size{};
	int bytes_per_pixel{};
	int wrapstate{};
	unsigned long numrend{};
};

/* Texture filter enum — mirrors opengl_texture_filter */
enum class vulkan_texture_filter : uint8_t
{
	classic,
	upscale,
	trilinear,
};

#define VKS_FLAG_MIPMAP (1 << 0)
#define VKS_FLAG_NOCOLOR (1 << 1)

/* Texture management */
vks_texture* vks_get_free_texture();
void vks_init_texture(vks_texture &t, uint32_t w, uint32_t h, int flags);
void vks_loadbmtexture_f(grs_bitmap &bm, vulkan_texture_filter texfilt, bool texanis, bool edgepad);
void vks_freebmtexture(grs_bitmap &bm);

/* Destroy all Vulkan resources (called during shutdown) */
void vks_shutdown_textures();

} /* namespace dcx */

#endif /* _VULKAN_TEXTURES_H_ */
