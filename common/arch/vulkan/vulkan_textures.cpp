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
#include <cmath>
#include <cstring>
#include <vector>

namespace dcx {

static std::array<vks_texture, VKS_MAX_TEXTURES> texture_pool{};
static std::vector<size_t> free_textures;

/* Textures destroyed while a frame is recording cannot be freed at once:
 * their descriptor sets may already be recorded into the current (or an
 * in-flight) command buffer, and vkDeviceWaitIdle() does not wait for a
 * command buffer that has not been submitted yet. Defer such destruction to
 * a per-frame pending list drained at frame begin, where the fence wait
 * guarantees every referencing command buffer has completed. Indexed by
 * vk_current_frame; size must match VK_MAX_FRAMES_IN_FLIGHT. */
constexpr uint32_t VKS_PENDING_SLOTS = 2;
static std::array<std::vector<vks_texture *>, VKS_PENDING_SLOTS> pending_free;

static void vks_destroy_texture(vks_texture *tex)
{
	if (tex->descriptor_set)
		vkFreeDescriptorSets(vk_device, vk_descriptor_pool, 1, &tex->descriptor_set);
	if (tex->sampler)
		vkDestroySampler(vk_device, tex->sampler, nullptr);
	if (tex->view)
		vkDestroyImageView(vk_device, tex->view, nullptr);
	if (tex->memory)
		vkFreeMemory(vk_device, tex->memory, nullptr);
	if (tex->image)
		vkDestroyImage(vk_device, tex->image, nullptr);
	if (tex->staging_buffer)
		vkDestroyBuffer(vk_device, tex->staging_buffer, nullptr);
	if (tex->staging_memory)
		vkFreeMemory(vk_device, tex->staging_memory, nullptr);
	tex->descriptor_set = VK_NULL_HANDLE;
	tex->sampler = VK_NULL_HANDLE;
	tex->view = VK_NULL_HANDLE;
	tex->memory = VK_NULL_HANDLE;
	tex->image = VK_NULL_HANDLE;
	tex->staging_buffer = VK_NULL_HANDLE;
	tex->staging_memory = VK_NULL_HANDLE;
	free_textures.push_back(static_cast<size_t>(tex - texture_pool.data()));
}

static size_t allocate_texture_slot()
{
	if (!free_textures.empty()) {
		size_t slot = free_textures.back();
		free_textures.pop_back();
		return slot;
	}
	for (size_t i = 0; i < VKS_MAX_TEXTURES; i++) {
		if (texture_pool[i].image == VK_NULL_HANDLE)
			return i;
	}
	Error(__FILE__, __LINE__, __func__, "Vulkan: No free texture slots available");
	return 0;
}

vks_texture* vks_get_free_texture()
{
	return &texture_pool[allocate_texture_slot()];
}

bool vks_init_texture(vks_texture &t, uint32_t w, uint32_t h, int flags)
{
	t.width = w;
	t.height = h;
	t.format = VK_FORMAT_R8G8B8A8_UNORM;
	t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	t.wrapstate = 0;
	t.bytes_per_pixel = 4;

	if (flags & VKS_FLAG_MIPMAP) {
		/* Calculate mipmap levels */
		t.mip_levels = static_cast<uint32_t>(std::floor(std::log2(std::max(w, h)))) + 1;
	} else {
		t.mip_levels = 1;
	}

	/* Create image */
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = w;
	imageInfo.extent.height = h;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = t.mip_levels;
	imageInfo.arrayLayers = 1;
	imageInfo.format = t.format;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VkResult result = vkCreateImage(vk_device, &imageInfo, nullptr, &t.image);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to create texture image");

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
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to allocate texture memory");

	result = vkBindImageMemory(vk_device, t.image, t.memory, 0);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to bind texture memory");

	/* Create image view */
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = t.image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = t.format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.levelCount = t.mip_levels;
	viewInfo.subresourceRange.layerCount = 1;

	result = vkCreateImageView(vk_device, &viewInfo, nullptr, &t.view);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to create texture image view");

	/* Create sampler */
	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_NEAREST;
	samplerInfo.minFilter = VK_FILTER_NEAREST;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_TRUE;
	samplerInfo.maxAnisotropy = 16.0f;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = static_cast<float>(t.mip_levels);

	result = vkCreateSampler(vk_device, &samplerInfo, nullptr, &t.sampler);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to create texture sampler");

	/* Allocate a descriptor set and bind this texture's view+sampler to it, so
	 * any draw sampling this texture can bind the set directly. */
	VkDescriptorSetAllocateInfo dsAlloc{};
	dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAlloc.descriptorPool = vk_descriptor_pool;
	dsAlloc.descriptorSetCount = 1;
	dsAlloc.pSetLayouts = &vk_descriptor_set_layout;
	result = vkAllocateDescriptorSets(vk_device, &dsAlloc, &t.descriptor_set);
	if (result != VK_SUCCESS) {
		/* Descriptor pool exhausted (e.g. heavy texture churn while frees are
		 * deferred). Tear down the partially-built texture and return its slot
		 * rather than aborting; the caller leaves the bitmap untextured and the
		 * draw path skips it. It will retry the upload on a later frame once
		 * pending frees drain back into the pool. */
		con_printf(CON_URGENT, "Vulkan: descriptor pool full; skipping %ux%u texture upload", w, h);
		vks_destroy_texture(&t);
		return false;
	}

	VkDescriptorImageInfo descImageInfo{};
	descImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	descImageInfo.imageView = t.view;
	descImageInfo.sampler = t.sampler;

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
	Error(__FILE__, __LINE__, __func__, "Vulkan: No suitable host-visible memory type for staging buffer");
}

/* Expand `w`x`h` paletted pixels (row stride `rowsize`) into RGBA via `pal`
 * and upload them into `tex` through a host-visible staging buffer. Mirrors
 * ogl_filltexbuf: index 255 with BM_FLAG_TRANSPARENT and index 254 with
 * BM_FLAG_SUPER_TRANSPARENT become fully transparent; everything else maps
 * through the palette (entries are 0..63, scaled by 4). */
static void vks_upload_paletted(vks_texture &tex, const uint8_t *src, const uint32_t rowsize, const uint32_t w, const uint32_t h, const uint8_t bmflags, const palette_array_t &pal)
{
	const uint32_t pixelCount = w * h;
	std::vector<uint8_t> rgba(static_cast<size_t>(pixelCount) * 4);
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

	/* Upload the RGBA data through a host-visible staging buffer. */
	VkDeviceSize imageSize = static_cast<VkDeviceSize>(pixelCount) * 4;

	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = imageSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkBuffer stagingBuffer;
	VkResult result = vkCreateBuffer(vk_device, &bufferInfo, nullptr, &stagingBuffer);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to create staging buffer");

	VkMemoryRequirements memReqs;
	vkGetBufferMemoryRequirements(vk_device, stagingBuffer, &memReqs);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReqs.size;
	allocInfo.memoryTypeIndex = find_host_visible_memory_type(memReqs);

	VkDeviceMemory stagingMemory;
	result = vkAllocateMemory(vk_device, &allocInfo, nullptr, &stagingMemory);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to allocate staging memory");

	result = vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory, 0);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to bind staging memory");

	void *data;
	result = vkMapMemory(vk_device, stagingMemory, 0, imageSize, 0, &data);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to map staging memory");
	std::memcpy(data, rgba.data(), imageSize);
	vkUnmapMemory(vk_device, stagingMemory);

	copyBufferToImage(stagingBuffer, tex.image, w, h);

	/* Staging resources are no longer needed once the upload completed. */
	vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
	vkFreeMemory(vk_device, stagingMemory, nullptr);
}

void vks_loadbmtexture_f(grs_bitmap &bm, vulkan_texture_filter /*texfilt*/, bool /*texanis*/, bool /*edgepad*/)
{
	/* Sub-bitmaps reference their parent's pixel data; upload from the root, as
	 * ogl_loadbmtexture_f does. */
	grs_bitmap *root = &bm;
	while (root->bm_parent)
		root = root->bm_parent;

	const uint32_t w = root->bm_w;
	const uint32_t h = root->bm_h;

	/* RLE-compressed bitmaps (BM_FLAG_RLE) store a packed run-length stream,
	 * not linear paletted pixels — decode to linear first, exactly as
	 * ogl_loadbmtexture_f does. The decode buffer must outlive the RGBA
	 * expansion below. */
	std::array<uint8_t, 300 * 1024> decodebuf;
	const uint8_t *src;
	uint32_t rowsize;
	if (root->get_flag_mask(BM_FLAG_RLE)) {
		decodebuf = {};
		if (!bm_rle_expand(*root).loop(w, bm_rle_expand_range{decodebuf}))
			con_printf(CON_URGENT, "Vulkan: insufficient space to decode %ux%u bitmap", w, h);
		src = decodebuf.data();
		rowsize = w;
	} else {
		src = root->get_bitmap_data();
		rowsize = root->bm_rowsize;
	}

	vks_texture *tex = vks_get_free_texture();
	if (!vks_init_texture(*tex, w, h, 0))
		return;	/* slot already returned by vks_destroy_texture; leave bm.vktexture null so the draw skips */

	/* Bake through gr_palette -- the selected game/art palette (set by
	 * gr_use_palette_table / the PCX loader), NOT the bound gr_current_pal,
	 * which lags until gr_palette_load. This mirrors
	 * ogl_loadtexture(gr_palette, ...) so fullscreen bitmaps uploaded before
	 * their gr_palette_load (briefing/title screens) get the right colours. */
	vks_upload_paletted(*tex, src, rowsize, w, h, root->get_flags(), gr_palette);

	/* Attach the texture to the root bitmap. The bitmap stays bm_mode::linear —
	 * the 2D blit dispatch (e.g. show_fullscr) selects the Vulkan path from the
	 * destination canvas type, and reads bm.vktexture. Setting the source to
	 * vulkan here would break that dispatch. */
	root->vktexture = tex;
	tex->numrend++;
}

vks_texture *vks_load_temporary_texture(const grs_bitmap &src, const uint32_t sx, const uint32_t sy, const uint32_t w, const uint32_t h)
{
	vks_texture *tex = vks_get_free_texture();
	if (!vks_init_texture(*tex, w, h, 0))
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
	if (vks_is_frame_recording())
	{
		/* A command buffer is open and/or another is in flight; the descriptor
		 * set may still be referenced. Defer destruction to the next frame
		 * begin, where the fence wait guarantees all referencing command
		 * buffers have completed. */
		pending_free[vk_current_frame % VKS_PENDING_SLOTS].push_back(&tex);
		return;
	}
	/* Outside the frame loop (e.g. level load): no command buffer is recording.
	 * Wait for in-flight submissions, then destroy at once. */
	vkDeviceWaitIdle(vk_device);
	vks_destroy_texture(&tex);
}

void vks_freebmtexture(grs_bitmap &bm)
{
	vks_texture *tex = bm.vktexture;
	if (!tex)
		return;
	/* Detach immediately so the bitmap will re-upload on next use rather than
	 * keep serving the about-to-be-destroyed texture. */
	bm.vktexture = nullptr;
	vks_free_texture(*tex);
}

/* Drain pending texture destruction for `frame`. Called from vks_begin_frame
 * after the per-frame fence wait, so every command buffer that could reference
 * a pending texture has finished. */
void vks_flush_pending_texture_frees(uint32_t frame)
{
	auto &v = pending_free[frame % VKS_PENDING_SLOTS];
	if (v.empty())
		return;
	vkDeviceWaitIdle(vk_device);
	for (auto *tex : v)
		vks_destroy_texture(tex);
	v.clear();
}

void vks_shutdown_textures()
{
	/* Textures may still be sampled by an in-flight frame (gr_close calls this
	 * before vks_shutdown's vkDeviceWaitIdle). Drain the queue first so we don't
	 * destroy samplers/images the GPU is still reading (VUID-vkDestroySampler-
	 * sampler-01082). */
	vkDeviceWaitIdle(vk_device);
	/* Destroy each occupied slot's Vulkan resources directly. The old code
	 * reinterpret_cast a vks_texture as a grs_bitmap and routed it through
	 * vks_freebmtexture, which is type-confused and corrupts the heap. */
	for (auto &tex : texture_pool) {
		if (tex.image == VK_NULL_HANDLE)
			continue;
		if (tex.sampler)
			vkDestroySampler(vk_device, tex.sampler, nullptr);
		if (tex.view)
			vkDestroyImageView(vk_device, tex.view, nullptr);
		if (tex.memory)
			vkFreeMemory(vk_device, tex.memory, nullptr);
		if (tex.image)
			vkDestroyImage(vk_device, tex.image, nullptr);
		if (tex.staging_buffer)
			vkDestroyBuffer(vk_device, tex.staging_buffer, nullptr);
		if (tex.staging_memory)
			vkFreeMemory(vk_device, tex.staging_memory, nullptr);
		tex = vks_texture{};
	}
	free_textures.clear();
	for (auto &v : pending_free)
		v.clear();
}

/* 1x1 opaque-white texture used as the sampler source for flat primitives
 * (filled rects, lines, pixels): modulating by white passes the vertex color
 * through unchanged. Lives outside the texture pool; its descriptor set is
 * freed with the pool at shutdown. */
static vks_texture vk_white_texture;

void vks_init_white_texture()
{
	if (!vks_init_texture(vk_white_texture, 1, 1, 0))
		Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to create white texture");

	/* Upload a single white texel through a throwaway staging buffer. */
	constexpr uint8_t white[4] = {255, 255, 255, 255};
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = sizeof(white);
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkBuffer stagingBuffer;
	VkResult result = vkCreateBuffer(vk_device, &bufferInfo, nullptr, &stagingBuffer);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to create white-texture staging buffer");

	VkMemoryRequirements memReqs;
	vkGetBufferMemoryRequirements(vk_device, stagingBuffer, &memReqs);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReqs.size;
	allocInfo.memoryTypeIndex = find_host_visible_memory_type(memReqs);

	VkDeviceMemory stagingMemory;
	result = vkAllocateMemory(vk_device, &allocInfo, nullptr, &stagingMemory);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to allocate white-texture staging memory");

	result = vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory, 0);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to bind white-texture staging memory");

	void *data;
	result = vkMapMemory(vk_device, stagingMemory, 0, sizeof(white), 0, &data);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to map white-texture staging memory");
	std::memcpy(data, white, sizeof(white));
	vkUnmapMemory(vk_device, stagingMemory);

	copyBufferToImage(stagingBuffer, vk_white_texture.image, 1, 1);

	vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
	vkFreeMemory(vk_device, stagingMemory, nullptr);

	vk_white_descriptor_set = vk_white_texture.descriptor_set;
}

void vks_destroy_white_texture()
{
	vkDeviceWaitIdle(vk_device);
	if (vk_white_texture.sampler)
		vkDestroySampler(vk_device, vk_white_texture.sampler, nullptr);
	if (vk_white_texture.view)
		vkDestroyImageView(vk_device, vk_white_texture.view, nullptr);
	if (vk_white_texture.memory)
		vkFreeMemory(vk_device, vk_white_texture.memory, nullptr);
	if (vk_white_texture.image)
		vkDestroyImage(vk_device, vk_white_texture.image, nullptr);
	vk_white_texture = vks_texture{};
	vk_white_descriptor_set = VK_NULL_HANDLE;
}

} /* namespace dcx */

#endif /* DXX_USE_VULKAN */
