/*
 * This file is part of the DXX-Rebirth project <https://www.dxx-rebirth.com/>.
 * It is copyright by its individual contributors, as recorded in the
 * project's Git history.  See COPYING.txt at the top level for license
 * terms and a link to the Git history.
 */

#include "dxxsconf.h"
#if DXX_USE_VULKAN

#include "vulkan_textures.h"
#include "vulkan_init.h"
#include "gr.h"
#include "error.h"
#include "u_mem.h"
#include "rle.h"
#include "console.h"

#include <array>
#include <algorithm>
#include <cstring>
#include <vector>

namespace dcx {

static std::array<vks_texture, VKS_MAX_TEXTURES> texture_pool{};
/* Slots released by vks_destroy_texture, reused before untouched ones. */
static std::vector<uint32_t> free_textures;
/* Slots at or above this index have never been handed out. */
static uint32_t next_unused_texture;

/* Textures released while frames may still reference them cannot be destroyed
 * at once: their descriptor sets may be recorded into the current command
 * buffer or one still executing. Destruction is deferred to a per-frame
 * pending list drained at that slot's next frame begin, after its fence wait
 * and a device idle, so every referencing command buffer has completed.
 * Indexed by vk_current_frame; size must match VK_MAX_FRAMES_IN_FLIGHT. */
constexpr uint32_t VKS_PENDING_SLOTS = 2;
static std::array<std::vector<vks_texture *>, VKS_PENDING_SLOTS> pending_free;

/* All textures share one sampler: drivers may cap the number of live samplers
 * (maxSamplerAllocationCount can be as low as 4000), far below the number of
 * textures a level uses. */
static VkSampler vk_texture_sampler;

static bool is_pool_texture(const vks_texture &tex)
{
	return &tex >= texture_pool.data() && &tex < texture_pool.data() + texture_pool.size();
}

static void vks_destroy_texture(vks_texture &tex)
{
	if (!tex.image)
		return;
	if (tex.descriptor_set)
		vkFreeDescriptorSets(vk_device, vk_descriptor_pool, 1, &tex.descriptor_set);
	if (tex.view)
		vkDestroyImageView(vk_device, tex.view, nullptr);
	vkDestroyImage(vk_device, tex.image, nullptr);
	if (tex.memory)
		vkFreeMemory(vk_device, tex.memory, nullptr);
	tex = {};
	if (is_pool_texture(tex))
		free_textures.push_back(static_cast<uint32_t>(&tex - texture_pool.data()));
}

/* Destroy pending textures from every slot whose frames have been submitted.
 * The open frame's slot is kept while it records: its command buffer is not
 * submitted yet, so a device idle does not cover it. */
static bool vks_reclaim_pending_textures()
{
	bool any = false;
	for (uint32_t i = 0; i != VKS_PENDING_SLOTS; ++i)
		if (!(vks_is_frame_recording() && i == vk_current_frame % VKS_PENDING_SLOTS) && !pending_free[i].empty())
			any = true;
	if (!any)
		return false;
	vkDeviceWaitIdle(vk_device);
	for (uint32_t i = 0; i != VKS_PENDING_SLOTS; ++i)
	{
		if (vks_is_frame_recording() && i == vk_current_frame % VKS_PENDING_SLOTS)
			continue;
		for (auto *const tex : pending_free[i])
			vks_destroy_texture(*tex);
		pending_free[i].clear();
	}
	return true;
}

static vks_texture *allocate_texture_slot()
{
	if (!free_textures.empty()) {
		const auto slot = free_textures.back();
		free_textures.pop_back();
		return &texture_pool[slot];
	}
	if (next_unused_texture < VKS_MAX_TEXTURES)
		return &texture_pool[next_unused_texture++];
	return nullptr;
}

static VkSampler vks_get_sampler()
{
	if (!vk_texture_sampler)
	{
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_NEAREST;
		samplerInfo.minFilter = VK_FILTER_NEAREST;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_FALSE;
		samplerInfo.maxAnisotropy = 1.0f;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;
		if (vkCreateSampler(vk_device, &samplerInfo, nullptr, &vk_texture_sampler) != VK_SUCCESS)
			Error("Vulkan: Failed to create texture sampler");
	}
	return vk_texture_sampler;
}

static VkDescriptorSet allocate_descriptor_set()
{
	VkDescriptorSetAllocateInfo dsAlloc{};
	dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAlloc.descriptorPool = vk_descriptor_pool;
	dsAlloc.descriptorSetCount = 1;
	dsAlloc.pSetLayouts = &vk_descriptor_set_layout;
	VkDescriptorSet ds = VK_NULL_HANDLE;
	if (vkAllocateDescriptorSets(vk_device, &dsAlloc, &ds) != VK_SUCCESS)
		return VK_NULL_HANDLE;
	return ds;
}

/* Create a `w`x`h` RGBA texture in `t`. The descriptor set is reserved first,
 * so an exhausted pool is reported before any image or memory is created. */
static bool vks_init_texture(vks_texture &t, const uint32_t w, const uint32_t h)
{
	t.descriptor_set = allocate_descriptor_set();
	if (!t.descriptor_set)
		return false;
	t.width = w;
	t.height = h;
	constexpr VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

	/* Create image */
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = w;
	imageInfo.extent.height = h;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.format = format;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VkResult result = vkCreateImage(vk_device, &imageInfo, nullptr, &t.image);
	if (!(result == VK_SUCCESS)) Error("Vulkan: Failed to create texture image");

	VkMemoryRequirements memReqs;
	vkGetImageMemoryRequirements(vk_device, t.image, &memReqs);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReqs.size;

	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &memProperties);
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((memReqs.memoryTypeBits & (1 << i)) &&
			(memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
			allocInfo.memoryTypeIndex = i;
			break;
		}
	}

	result = vkAllocateMemory(vk_device, &allocInfo, nullptr, &t.memory);
	if (!(result == VK_SUCCESS)) Error("Vulkan: Failed to allocate texture memory");

	result = vkBindImageMemory(vk_device, t.image, t.memory, 0);
	if (!(result == VK_SUCCESS)) Error("Vulkan: Failed to bind texture memory");

	/* Create image view */
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = t.image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.layerCount = 1;

	result = vkCreateImageView(vk_device, &viewInfo, nullptr, &t.view);
	if (!(result == VK_SUCCESS)) Error("Vulkan: Failed to create texture image view");

	/* Bind this texture's view and the shared sampler to the descriptor set,
	 * so any draw sampling this texture can bind the set directly. */
	VkDescriptorImageInfo descImageInfo{};
	descImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	descImageInfo.imageView = t.view;
	descImageInfo.sampler = vks_get_sampler();

	VkWriteDescriptorSet descriptorWrite{};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = t.descriptor_set;
	descriptorWrite.dstBinding = 0;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pImageInfo = &descImageInfo;
	vkUpdateDescriptorSets(vk_device, 1, &descriptorWrite, 0, nullptr);
	return true;
}

/* Take a texture slot and create a `w`x`h` texture in it. When the pool or
 * the descriptor pool is exhausted, reclaim textures whose deferred
 * destruction is due and retry once; if that still fails, report it and
 * return nullptr so the draw is skipped rather than aborting. */
static vks_texture *vks_create_texture(const uint32_t w, const uint32_t h)
{
	for (bool retried = false;; retried = true)
	{
		if (vks_texture *const tex = allocate_texture_slot())
		{
			if (vks_init_texture(*tex, w, h))
				return tex;
			*tex = {};
			free_textures.push_back(static_cast<uint32_t>(tex - texture_pool.data()));
		}
		if (retried || !vks_reclaim_pending_textures())
			break;
	}
	con_printf(CON_URGENT, "Vulkan: texture pool exhausted; skipping %ux%u texture upload", w, h);
	return nullptr;
}

/* Allocate and begin a one-time-submit command buffer for transfer work
 * (staging uploads, layout transitions). vk_command_pools[0] is created in
 * gr_set_mode, which always runs before any texture is loaded. */
static VkCommandBuffer begin_one_time_commands()
{
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = vk_command_pools[0];
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer cmd;
	vkAllocateCommandBuffers(vk_device, &allocInfo, &cmd);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &beginInfo);
	return cmd;
}

static void submit_one_time_commands(VkCommandBuffer cmd)
{
	vkEndCommandBuffer(cmd);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmd;

	vkQueueSubmit(vk_graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(vk_graphics_queue);

	vkFreeCommandBuffers(vk_device, vk_command_pools[0], 1, &cmd);
}

static void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
	VkCommandBuffer cmd = begin_one_time_commands();

	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.layerCount = 1;
	region.imageExtent.width = width;
	region.imageExtent.height = height;
	region.imageExtent.depth = 1;

	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.layerCount = 1;

	/* UNDEFINED -> TRANSFER_DST_OPTIMAL */
	barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.srcAccessMask = 0;
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	vkCmdPipelineBarrier(cmd,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, nullptr, 0, nullptr, 1, &barrier);

	vkCmdCopyBufferToImage(cmd, buffer, image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	/* TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL */
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(cmd,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		0, 0, nullptr, 0, nullptr, 1, &barrier);

	submit_one_time_commands(cmd);
}

/* Find a host-visible, host-coherent memory type satisfying `requirements'. */
static uint32_t find_host_visible_memory_type(VkMemoryRequirements requirements)
{
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &memProperties);
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((requirements.memoryTypeBits & (1u << i)) &&
			(memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
			(memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
			return i;
	}
	Error("Vulkan: No suitable host-visible memory type for staging buffer");
}

/* Upload `rgba` (`w`x`h` RGBA8 texels) into `image` through a host-visible
 * staging buffer. */
static void vks_upload_rgba(const VkImage image, const uint8_t *const rgba, const uint32_t w, const uint32_t h)
{
	const VkDeviceSize imageSize = static_cast<VkDeviceSize>(w) * h * 4;

	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = imageSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkBuffer stagingBuffer;
	VkResult result = vkCreateBuffer(vk_device, &bufferInfo, nullptr, &stagingBuffer);
	if (!(result == VK_SUCCESS)) Error("Vulkan: Failed to create staging buffer");

	VkMemoryRequirements memReqs;
	vkGetBufferMemoryRequirements(vk_device, stagingBuffer, &memReqs);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReqs.size;
	allocInfo.memoryTypeIndex = find_host_visible_memory_type(memReqs);

	VkDeviceMemory stagingMemory;
	result = vkAllocateMemory(vk_device, &allocInfo, nullptr, &stagingMemory);
	if (!(result == VK_SUCCESS)) Error("Vulkan: Failed to allocate staging memory");

	result = vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory, 0);
	if (!(result == VK_SUCCESS)) Error("Vulkan: Failed to bind staging memory");

	void *data;
	result = vkMapMemory(vk_device, stagingMemory, 0, imageSize, 0, &data);
	if (!(result == VK_SUCCESS)) Error("Vulkan: Failed to map staging memory");
	std::memcpy(data, rgba, imageSize);
	vkUnmapMemory(vk_device, stagingMemory);

	copyBufferToImage(stagingBuffer, image, w, h);

	/* Staging resources are no longer needed once the upload completed. */
	vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
	vkFreeMemory(vk_device, stagingMemory, nullptr);
}

/* Expand `w`x`h` paletted pixels (row stride `rowsize`) into RGBA via `pal`
 * and upload them into `tex`. Mirrors ogl_filltexbuf: index 255 with
 * BM_FLAG_TRANSPARENT and index 254 with BM_FLAG_SUPER_TRANSPARENT become
 * fully transparent; everything else maps through the palette (entries are
 * 0..63, scaled by 4). */
static void vks_upload_paletted(vks_texture &tex, const uint8_t *src, const uint32_t rowsize, const uint32_t w, const uint32_t h, const uint8_t bmflags, const palette_array_t &pal)
{
	std::vector<uint8_t> rgba(static_cast<size_t>(w) * h * 4);
	uint8_t *out = rgba.data();
	for (uint32_t y = 0; y < h; y++) {
		const uint8_t *row = src + static_cast<size_t>(y) * rowsize;
		for (uint32_t x = 0; x < w; x++) {
			const uint8_t c = row[x];
			if (c == 254 && (bmflags & BM_FLAG_SUPER_TRANSPARENT)) {
				out[0] = 255; out[1] = 255; out[2] = 255; out[3] = 0;
			} else if (c == 255 && (bmflags & BM_FLAG_TRANSPARENT)) {
				out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
			} else {
				const rgb_t &col = pal[c];
				out[0] = col.r * 4;
				out[1] = col.g * 4;
				out[2] = col.b * 4;
				out[3] = 255;
			}
			out += 4;
		}
	}
	vks_upload_rgba(tex.image, rgba.data(), w, h);
}

static grs_bitmap &vks_root_bitmap(grs_bitmap &bm)
{
	grs_bitmap *root = &bm;
	while (root->bm_parent)
		root = root->bm_parent;
	return *root;
}

static void vks_loadbmtexture_f(grs_bitmap &bm, vulkan_texture_filter /*texfilt*/, bool /*texanis*/, bool /*edgepad*/)
{
	/* Sub-bitmaps reference their parent's pixel data; upload from the root, as
	 * ogl_loadbmtexture_f does. */
	grs_bitmap &root = vks_root_bitmap(bm);

	const uint32_t w = root.bm_w;
	const uint32_t h = root.bm_h;

	/* RLE-compressed bitmaps (BM_FLAG_RLE) store a packed run-length stream,
	 * not linear paletted pixels — decode to linear first, exactly as
	 * ogl_loadbmtexture_f does. The decode buffer must outlive the RGBA
	 * expansion below. */
	std::array<uint8_t, 300 * 1024> decodebuf;
	const uint8_t *src;
	uint32_t rowsize;
	if (root.get_flag_mask(BM_FLAG_RLE)) {
		decodebuf = {};
		if (!bm_rle_expand(root).loop(w, bm_rle_expand_range{decodebuf}))
			con_printf(CON_URGENT, "Vulkan: insufficient space to decode %ux%u bitmap", w, h);
		src = decodebuf.data();
		rowsize = w;
	} else {
		src = root.get_bitmap_data();
		rowsize = root.bm_rowsize;
	}

	vks_texture *const tex = vks_create_texture(w, h);
	if (!tex)
		return;	/* leave bm.vktexture null so the draw skips */

	/* Bake through gr_palette -- the selected game/art palette (set by
	 * gr_use_palette_table / the PCX loader), NOT the bound gr_current_pal,
	 * which lags until gr_palette_load. This mirrors
	 * ogl_loadtexture(gr_palette, ...) so fullscreen bitmaps uploaded before
	 * their gr_palette_load (briefing/title screens) get the right colours. */
	vks_upload_paletted(*tex, src, rowsize, w, h, root.get_flags(), gr_palette);

	/* Attach the texture to the root bitmap. The bitmap stays bm_mode::linear —
	 * the 2D blit dispatch (e.g. show_fullscr) selects the Vulkan path from the
	 * destination canvas type, and reads bm.vktexture. Setting the source to
	 * vulkan here would break that dispatch. */
	tex->owner_data = root.get_bitmap_data();
	root.vktexture = tex;
}

vks_texture *vks_get_bmtexture(grs_bitmap &bm)
{
	grs_bitmap &root = vks_root_bitmap(bm);
	/* A pointer copied from another bitmap, or left behind after its owner
	 * released it, does not name this bitmap's data: upload this bitmap's own
	 * texture instead of sampling (or later freeing) somebody else's. */
	if (const auto tex = root.vktexture; tex && (!tex->owner_data || tex->owner_data != root.get_bitmap_data()))
		root.vktexture = nullptr;
	if (!root.vktexture)
		vks_loadbmtexture_f(bm, vulkan_texture_filter::classic, false, false);
	return root.vktexture;
}

vks_texture *vks_load_temporary_texture(const grs_bitmap &src, const uint32_t sx, const uint32_t sy, const uint32_t w, const uint32_t h)
{
	vks_texture *const tex = vks_create_texture(w, h);
	if (!tex)
		return nullptr;
	const uint32_t rowsize = src.bm_rowsize;
	vks_upload_paletted(*tex, src.get_bitmap_data() + static_cast<size_t>(sy) * rowsize + sx, rowsize, w, h, src.get_flags(), gr_current_pal);
	return tex;
}

void vks_free_texture(vks_texture &tex)
{
	/* The device is gone: static destructors (e.g. ~Gamefonts) run during
	 * exit(), after gr_close/vks_shutdown already tore Vulkan down. The
	 * texture's Vulkan resources were freed by shutdown. */
	if (!vk_device)
		return;
	tex.owner_data = nullptr;
	/* Batch destruction: the frame slot's next begin waits for the device
	 * once for everything released in between, rather than idling the
	 * device for each texture (level loads release hundreds). */
	pending_free[vk_current_frame % VKS_PENDING_SLOTS].push_back(&tex);
}

void vks_freebmtexture(grs_bitmap &bm)
{
	vks_texture *const tex = bm.vktexture;
	if (!tex)
		return;
	/* Detach immediately so the bitmap will re-upload on next use rather than
	 * keep serving the about-to-be-destroyed texture. Only the bitmap whose
	 * data the texture was built from releases it; copies of the pointer
	 * (aliases, sub-bitmaps) just drop their reference. */
	bm.vktexture = nullptr;
	if (tex->owner_data && tex->owner_data == bm.get_bitmap_data())
		vks_free_texture(*tex);
}

void vks_flush_pending_texture_frees(uint32_t frame)
{
	auto &v = pending_free[frame % VKS_PENDING_SLOTS];
	if (v.empty())
		return;
	/* Textures released while no frame was recording may be referenced by the
	 * other slot's frame, which this slot's fence does not cover. */
	vkDeviceWaitIdle(vk_device);
	for (auto *tex : v)
		vks_destroy_texture(*tex);
	v.clear();
}

void vks_shutdown_textures()
{
	/* Textures may still be sampled by an in-flight frame (gr_close calls this
	 * before vks_shutdown's vkDeviceWaitIdle). Drain the queue first so we don't
	 * destroy images the GPU is still reading. */
	vkDeviceWaitIdle(vk_device);
	for (auto &tex : texture_pool)
		vks_destroy_texture(tex);
	free_textures.clear();
	next_unused_texture = 0;
	for (auto &v : pending_free)
		v.clear();
	if (vk_texture_sampler)
	{
		vkDestroySampler(vk_device, vk_texture_sampler, nullptr);
		vk_texture_sampler = VK_NULL_HANDLE;
	}
}

/* 1x1 opaque-white texture used as the sampler source for flat primitives
 * (filled rects, lines, pixels): modulating by white passes the vertex color
 * through unchanged. Lives outside the texture pool. */
static vks_texture vk_white_texture;

void vks_init_white_texture()
{
	if (!vks_init_texture(vk_white_texture, 1, 1))
		Error("Vulkan: Failed to create white texture");
	constexpr uint8_t white[4] = {255, 255, 255, 255};
	vks_upload_rgba(vk_white_texture.image, white, 1, 1);
	vk_white_descriptor_set = vk_white_texture.descriptor_set;
}

void vks_destroy_white_texture()
{
	vkDeviceWaitIdle(vk_device);
	vks_destroy_texture(vk_white_texture);
	vk_white_descriptor_set = VK_NULL_HANDLE;
}

} /* namespace dcx */

#endif /* DXX_USE_VULKAN */
