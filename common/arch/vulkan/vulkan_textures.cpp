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
#include "config.h"

#include <array>
#include <algorithm>
#include <bit>
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

/* Textures share samplers, one per (filter, anisotropy, wrap) combination:
 * drivers may cap the number of live samplers (maxSamplerAllocationCount can
 * be as low as 4000), far below the number of textures a level uses. */
struct vks_sampler_mode
{
	vulkan_texture_filter filter;
	bool anisotropic;
};
static std::array<std::array<std::array<VkSampler, 2>, 2>, 3> vk_texture_samplers;

static bool is_pool_texture(const vks_texture &tex)
{
	return &tex >= texture_pool.data() && &tex < texture_pool.data() + texture_pool.size();
}

static void vks_destroy_texture(vks_texture &tex)
{
	if (!tex.image)
		return;
	if (tex.descriptor_set)
	{
		const std::array<VkDescriptorSet, VKS_DESCRIPTOR_SETS_PER_TEXTURE> sets{{tex.descriptor_set, tex.descriptor_set_clamp}};
		vkFreeDescriptorSets(vk_device, vk_descriptor_pool, sets.size(), sets.data());
	}
	if (tex.view)
		vkDestroyImageView(vk_device, tex.view, nullptr);
	vkDestroyImage(vk_device, tex.image, nullptr);
	if (tex.memory)
		vkFreeMemory(vk_device, tex.memory, nullptr);
	tex = {};
	if (is_pool_texture(tex))
		free_textures.push_back(static_cast<uint32_t>(&tex - texture_pool.data()));
}

static void vks_submit_pending_uploads();

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
	/* Uploads recorded outside a frame may reference the textures about to be
	 * destroyed; while a frame records, they belong to the skipped slot. */
	if (!vks_is_frame_recording())
		vks_submit_pending_uploads();
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

/* Anisotropic filtering is honoured only when the device enabled it. */
static bool vks_anisotropy_available()
{
	return vk_max_sampler_anisotropy > 1.0f;
}

/* Filter states matching ogl_loadtexture: classic is nearest (with a
 * nearest-texel, linear-between-mips minification when anisotropy is on);
 * upscale and trilinear filter linearly with trilinear mipmapping. */
static VkSampler vks_get_sampler(const vks_sampler_mode mode, const bool clamp)
{
	auto &sampler = vk_texture_samplers[static_cast<uint8_t>(mode.filter)][mode.anisotropic][clamp];
	if (!sampler)
	{
		const bool linear = mode.filter != vulkan_texture_filter::classic;
		const auto address = clamp ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
		samplerInfo.minFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
		samplerInfo.addressModeU = address;
		samplerInfo.addressModeV = address;
		samplerInfo.addressModeW = address;
		samplerInfo.anisotropyEnable = mode.anisotropic;
		samplerInfo.maxAnisotropy = mode.anisotropic ? vk_max_sampler_anisotropy : 1.0f;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
		if (vkCreateSampler(vk_device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS)
			Error("Vulkan: Failed to create texture sampler");
	}
	return sampler;
}

/* Mipmaps are generated with linear blits, which the texture format must
 * support; without that, filtered textures fall back to a single level. */
static bool vks_can_generate_mipmaps()
{
	static const bool supported = [] {
		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(vk_physical_device, VK_FORMAT_R8G8B8A8_UNORM, &props);
		constexpr VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
		return (props.optimalTilingFeatures & needed) == needed;
	}();
	return supported;
}

static uint32_t vks_mip_levels(const uint32_t w, const uint32_t h)
{
	return std::bit_width(std::max(w, h));
}

/* Create a `w`x`h` RGBA texture with `mip_levels` levels in `t`, sampled with
 * `mode`. The descriptor sets are reserved first, so an exhausted pool is
 * reported before any image or memory is created. */
static bool vks_init_texture(vks_texture &t, const uint32_t w, const uint32_t h, const uint32_t mip_levels, const vks_sampler_mode mode)
{
	const std::array<VkDescriptorSetLayout, VKS_DESCRIPTOR_SETS_PER_TEXTURE> layouts{{vk_descriptor_set_layout, vk_descriptor_set_layout}};
	std::array<VkDescriptorSet, VKS_DESCRIPTOR_SETS_PER_TEXTURE> sets{};
	VkDescriptorSetAllocateInfo dsAlloc{};
	dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAlloc.descriptorPool = vk_descriptor_pool;
	dsAlloc.descriptorSetCount = layouts.size();
	dsAlloc.pSetLayouts = layouts.data();
	if (vkAllocateDescriptorSets(vk_device, &dsAlloc, sets.data()) != VK_SUCCESS)
		return false;
	t.descriptor_set = sets[0];
	t.descriptor_set_clamp = sets[1];
	constexpr VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

	/* Create image */
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = w;
	imageInfo.extent.height = h;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = mip_levels;
	imageInfo.arrayLayers = 1;
	imageInfo.format = format;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
		(mip_levels > 1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
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
	viewInfo.subresourceRange.levelCount = mip_levels;
	viewInfo.subresourceRange.layerCount = 1;

	result = vkCreateImageView(vk_device, &viewInfo, nullptr, &t.view);
	if (!(result == VK_SUCCESS)) Error("Vulkan: Failed to create texture image view");

	/* Bind this texture's view and the shared samplers to the descriptor
	 * sets, so any draw sampling this texture can bind a set directly. */
	std::array<VkDescriptorImageInfo, VKS_DESCRIPTOR_SETS_PER_TEXTURE> descImageInfo{};
	std::array<VkWriteDescriptorSet, VKS_DESCRIPTOR_SETS_PER_TEXTURE> descriptorWrite{};
	for (uint32_t i = 0; i != VKS_DESCRIPTOR_SETS_PER_TEXTURE; ++i)
	{
		descImageInfo[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		descImageInfo[i].imageView = t.view;
		descImageInfo[i].sampler = vks_get_sampler(mode, i != 0);
		descriptorWrite[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrite[i].dstSet = sets[i];
		descriptorWrite[i].dstBinding = 0;
		descriptorWrite[i].dstArrayElement = 0;
		descriptorWrite[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorWrite[i].descriptorCount = 1;
		descriptorWrite[i].pImageInfo = &descImageInfo[i];
	}
	vkUpdateDescriptorSets(vk_device, descriptorWrite.size(), descriptorWrite.data(), 0, nullptr);
	return true;
}

/* Take a texture slot and create a `w`x`h` texture in it. When the pool or
 * the descriptor pool is exhausted, reclaim textures whose deferred
 * destruction is due and retry once; if that still fails, report it and
 * return nullptr so the draw is skipped rather than aborting. */
static vks_texture *vks_create_texture(const uint32_t w, const uint32_t h, const uint32_t mip_levels, const vks_sampler_mode mode)
{
	for (bool retried = false;; retried = true)
	{
		if (vks_texture *const tex = allocate_texture_slot())
		{
			if (vks_init_texture(*tex, w, h, mip_levels, mode))
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

/* Texture uploads are recorded into a per-frame-slot transfer command buffer
 * and submitted together with that slot's frame (ahead of the frame's own
 * command buffer), instead of a submit + vkQueueWaitIdle per texture. Staging
 * data is sub-allocated from per-slot host-visible chunks, recycled once the
 * slot's fence shows the frame has completed. The command pool is separate
 * from the frame pools, which are rebuilt with the swapchain. */
struct vks_staging_chunk
{
	VkBuffer buffer{};
	VkDeviceMemory memory{};
	uint8_t *mapped{};
	VkDeviceSize size{};
	VkDeviceSize used{};
};

struct vks_upload_slot
{
	VkCommandBuffer cmd{};
	bool recording{};
	std::vector<vks_staging_chunk> chunks;
};

static VkCommandPool vk_upload_pool;
static std::array<vks_upload_slot, VKS_PENDING_SLOTS> upload_slots;
constexpr VkDeviceSize VKS_STAGING_CHUNK_SIZE = 8 << 20;

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

static vks_staging_chunk create_staging_chunk(const VkDeviceSize size)
{
	vks_staging_chunk chunk;
	chunk.size = size;

	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkResult result = vkCreateBuffer(vk_device, &bufferInfo, nullptr, &chunk.buffer);
	if (!(result == VK_SUCCESS)) Error("Vulkan: Failed to create staging buffer");

	VkMemoryRequirements memReqs;
	vkGetBufferMemoryRequirements(vk_device, chunk.buffer, &memReqs);
	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReqs.size;
	allocInfo.memoryTypeIndex = find_host_visible_memory_type(memReqs);
	result = vkAllocateMemory(vk_device, &allocInfo, nullptr, &chunk.memory);
	if (!(result == VK_SUCCESS)) Error("Vulkan: Failed to allocate staging memory");
	result = vkBindBufferMemory(vk_device, chunk.buffer, chunk.memory, 0);
	if (!(result == VK_SUCCESS)) Error("Vulkan: Failed to bind staging memory");
	void *data;
	result = vkMapMemory(vk_device, chunk.memory, 0, size, 0, &data);
	if (!(result == VK_SUCCESS)) Error("Vulkan: Failed to map staging memory");
	chunk.mapped = static_cast<uint8_t *>(data);
	return chunk;
}

static void destroy_staging_chunk(vks_staging_chunk &chunk)
{
	vkUnmapMemory(vk_device, chunk.memory);
	vkDestroyBuffer(vk_device, chunk.buffer, nullptr);
	vkFreeMemory(vk_device, chunk.memory, nullptr);
	chunk = {};
}

/* The upload command buffer of the current frame slot, begun on first use.
 * Beginning it reuses the slot's command buffer and staging, so the slot's
 * previous submission must have completed: vks_begin_frame already waited on
 * the slot's fence when a frame is recording; otherwise wait here. */
static vks_upload_slot &vks_begin_uploads()
{
	const auto frame = vk_current_frame % VKS_PENDING_SLOTS;
	auto &slot = upload_slots[frame];
	if (slot.recording)
		return slot;
	if (!vk_upload_pool)
	{
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = vk_graphics_queue_family;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		if (vkCreateCommandPool(vk_device, &poolInfo, nullptr, &vk_upload_pool) != VK_SUCCESS)
			Error("Vulkan: Failed to create upload command pool");
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = vk_upload_pool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = 1;
		for (auto &s : upload_slots)
			if (vkAllocateCommandBuffers(vk_device, &allocInfo, &s.cmd) != VK_SUCCESS)
				Error("Vulkan: Failed to allocate upload command buffer");
	}
	if (!vks_is_frame_recording() && frame < vk_in_flight_fences.size())
		vkWaitForFences(vk_device, 1, &vk_in_flight_fences[frame], VK_TRUE, UINT64_MAX);
	/* Keep the first chunk for reuse; release overflow chunks. */
	while (slot.chunks.size() > 1)
	{
		destroy_staging_chunk(slot.chunks.back());
		slot.chunks.pop_back();
	}
	if (!slot.chunks.empty())
		slot.chunks.front().used = 0;
	vkResetCommandBuffer(slot.cmd, 0);
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(slot.cmd, &beginInfo);
	slot.recording = true;
	return slot;
}

VkCommandBuffer vks_end_pending_uploads()
{
	auto &slot = upload_slots[vk_current_frame % VKS_PENDING_SLOTS];
	if (!slot.recording)
		return VK_NULL_HANDLE;
	vkEndCommandBuffer(slot.cmd);
	slot.recording = false;
	return slot.cmd;
}

/* Submit recorded uploads on their own and wait for them, for paths that
 * must destroy resources those uploads reference before the next frame
 * would have submitted them. */
static void vks_submit_pending_uploads()
{
	const VkCommandBuffer cmd = vks_end_pending_uploads();
	if (!cmd)
		return;
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmd;
	if (vkQueueSubmit(vk_graphics_queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
		Error("Vulkan: Failed to submit texture uploads");
	vkQueueWaitIdle(vk_graphics_queue);
}

/* Record an upload of `rgba` (`w`x`h` RGBA8 texels) into level 0 of `image`,
 * generate its remaining `mip_levels` - 1 levels with linear blits, and
 * transition every level to be sampleable by the frame that follows. */
static void vks_upload_rgba(const VkImage image, const uint8_t *const rgba, const uint32_t w, const uint32_t h, const uint32_t mip_levels)
{
	auto &slot = vks_begin_uploads();
	const VkDeviceSize imageSize = static_cast<VkDeviceSize>(w) * h * 4;
	/* Texel-aligned (and comfortably above optimalBufferCopyOffsetAlignment)
	 * sub-allocation from the slot's staging chunks. */
	constexpr VkDeviceSize alignment = 16;
	if (slot.chunks.empty() || ((slot.chunks.back().used + alignment - 1) & ~(alignment - 1)) + imageSize > slot.chunks.back().size)
		slot.chunks.emplace_back(create_staging_chunk(std::max(VKS_STAGING_CHUNK_SIZE, imageSize)));
	auto &chunk = slot.chunks.back();
	const VkDeviceSize offset = (chunk.used + alignment - 1) & ~(alignment - 1);
	std::memcpy(chunk.mapped + offset, rgba, imageSize);
	chunk.used = offset + imageSize;

	VkBufferImageCopy region{};
	region.bufferOffset = offset;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.layerCount = 1;
	region.imageExtent.width = w;
	region.imageExtent.height = h;
	region.imageExtent.depth = 1;

	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.levelCount = mip_levels;
	barrier.subresourceRange.layerCount = 1;

	/* UNDEFINED -> TRANSFER_DST_OPTIMAL, every level */
	barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.srcAccessMask = 0;
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	vkCmdPipelineBarrier(slot.cmd,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, nullptr, 0, nullptr, 1, &barrier);

	vkCmdCopyBufferToImage(slot.cmd, chunk.buffer, image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	barrier.subresourceRange.levelCount = 1;
	int32_t mw = static_cast<int32_t>(w), mh = static_cast<int32_t>(h);
	for (uint32_t level = 1; level < mip_levels; ++level)
	{
		/* Level - 1 becomes the blit source, then is final. */
		barrier.subresourceRange.baseMipLevel = level - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		vkCmdPipelineBarrier(slot.cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier);

		const int32_t nw = std::max(mw / 2, 1), nh = std::max(mh / 2, 1);
		VkImageBlit blit{};
		blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1};
		blit.srcOffsets[1] = {mw, mh, 1};
		blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
		blit.dstOffsets[1] = {nw, nh, 1};
		vkCmdBlitImage(slot.cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(slot.cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier);
		mw = nw;
		mh = nh;
	}

	/* The last level was only written: TRANSFER_DST_OPTIMAL ->
	 * SHADER_READ_ONLY_OPTIMAL */
	barrier.subresourceRange.baseMipLevel = mip_levels - 1;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(slot.cmd,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		0, 0, nullptr, 0, nullptr, 1, &barrier);
}

/* Expand `w`x`h` paletted pixels (row stride `rowsize`) into RGBA via `pal`.
 * Mirrors ogl_filltexbuf: index 255 with BM_FLAG_TRANSPARENT and index 254
 * with BM_FLAG_SUPER_TRANSPARENT become fully transparent; everything else
 * maps through the palette (entries are 0..63, scaled by 4). */
static std::vector<uint8_t> vks_expand_paletted(const uint8_t *src, const uint32_t rowsize, const uint32_t w, const uint32_t h, const uint8_t bmflags, const palette_array_t &pal)
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
	return rgba;
}

/* Bleed colour into transparent texels from an opaque neighbour, so linear
 * filtering does not darken the edges of see-through areas ("dark edges
 * problem"). A port of the edge padding in ogl_loadtexture. */
static void vks_edgepad(std::vector<uint8_t> &rgba, const uint32_t w)
{
	uint8_t *p = rgba.data();
	uint8_t *const pdone = p + rgba.size() - 4;
	const ptrdiff_t line = 4 * static_cast<ptrdiff_t>(w);
	p += 4;
	uint8_t *const ptop = p + line;
	uint8_t *const pbottom = pdone - line;
	const auto copy_from = [](uint8_t *const d, const uint8_t *const s) {
		d[0] = s[0];
		d[1] = s[1];
		d[2] = s[2];
	};
	for (; p < pdone; p += 4)
	{
		//offsets 0 to 2 are r, g, b. offset 3 is alpha. 0x00 is transparent, 0xff is opaque.
		if (p[3])
			continue;
		if (p[-1])
			copy_from(p, p - 4);	//from left
		else if (p[7])
			copy_from(p, p + 4);	//from right
		else if (p >= ptop && p[-line + 3])
			copy_from(p, p - line);	//from above
		else if (p < pbottom && p[line + 3])
			copy_from(p, p + line);	//from below
		else if (p < pbottom && p[line - 1])
			copy_from(p, p + line - 4);	//bottom left
		else if (p < pbottom && p[line + 7])
			copy_from(p, p + line + 4);	//bottom right
		else if (p >= ptop && p[-line - 1])
			copy_from(p, p - line - 4);	//top left
		else if (p >= ptop && p[-line + 7])
			copy_from(p, p - line + 4);	//top right
	}
}

/* Nearest-neighbour enlargement for the "upscale" filter, so linear
 * filtering keeps texels blocky but smooths their borders. */
static std::vector<uint8_t> vks_upscale(const std::vector<uint8_t> &rgba, const uint32_t w, const uint32_t h, const uint32_t rescale)
{
	std::vector<uint8_t> out(rgba.size() * rescale * rescale);
	const uint32_t ow = w * rescale;
	for (uint32_t y = 0; y < h * rescale; ++y)
		for (uint32_t x = 0; x < ow; ++x)
			std::memcpy(&out[(static_cast<size_t>(y) * ow + x) * 4], &rgba[(static_cast<size_t>(y / rescale) * w + x / rescale) * 4], 4);
	return out;
}

/* Create and upload a texture for `w`x`h` RGBA texels with filtering
 * `texfilt` (and anisotropy if `texanis`), following ogl_loadtexture's
 * choices: edge padding when filtering, 4x enlargement for "upscale" unless
 * the image exceeds 256 texels (then classic), and a mip chain for the
 * filtered modes or anisotropic classic. */
static vks_texture *vks_create_rgba_texture(std::vector<uint8_t> rgba, const uint32_t w, const uint32_t h, vulkan_texture_filter texfilt, bool texanis, const bool edgepad)
{
	texanis = texanis && vks_anisotropy_available();
	if (texfilt != vulkan_texture_filter::classic && edgepad)
		vks_edgepad(rgba, w);
	uint32_t rescale = 1;
	if (texfilt == vulkan_texture_filter::upscale)
	{
		if (w > 256 || h > 256)
			texfilt = vulkan_texture_filter::classic;
		else
		{
			rescale = 4;
			rgba = vks_upscale(rgba, w, h, rescale);
		}
	}
	const uint32_t iw = w * rescale, ih = h * rescale;
	const bool mipmap = (texfilt != vulkan_texture_filter::classic || texanis) && vks_can_generate_mipmaps();
	const uint32_t mip_levels = mipmap ? vks_mip_levels(iw, ih) : 1;
	vks_texture *const tex = vks_create_texture(iw, ih, mip_levels, {texfilt, texanis});
	if (!tex)
		return nullptr;
	vks_upload_rgba(tex->image, rgba.data(), iw, ih, mip_levels);
	tex->width = w;
	tex->height = h;
	return tex;
}

static grs_bitmap &vks_root_bitmap(grs_bitmap &bm)
{
	grs_bitmap *root = &bm;
	while (root->bm_parent)
		root = root->bm_parent;
	return *root;
}

static void vks_loadbmtexture_f(grs_bitmap &bm, const vulkan_texture_filter texfilt, const bool texanis, const bool edgepad)
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

	/* Bake through gr_palette -- the selected game/art palette (set by
	 * gr_use_palette_table / the PCX loader), NOT the bound gr_current_pal,
	 * which lags until gr_palette_load. This mirrors
	 * ogl_loadtexture(gr_palette, ...) so fullscreen bitmaps uploaded before
	 * their gr_palette_load (briefing/title screens) get the right colours. */
	vks_texture *const tex = vks_create_rgba_texture(vks_expand_paletted(src, rowsize, w, h, root.get_flags(), gr_palette), w, h, texfilt, texanis, edgepad);
	if (!tex)
		return;	/* leave bm.vktexture null so the draw skips */

	/* Attach the texture to the root bitmap. The bitmap stays bm_mode::linear —
	 * the 2D blit dispatch (e.g. show_fullscr) selects the Vulkan path from the
	 * destination canvas type, and reads bm.vktexture. Setting the source to
	 * vulkan here would break that dispatch. */
	tex->owner_data = root.get_bitmap_data();
	root.vktexture = tex;
}

vks_texture *vks_get_bmtexture(grs_bitmap &bm, const bool edgepad)
{
	grs_bitmap &root = vks_root_bitmap(bm);
	/* A pointer copied from another bitmap, or left behind after its owner
	 * released it, does not name this bitmap's data: upload this bitmap's own
	 * texture instead of sampling (or later freeing) somebody else's. */
	if (const auto tex = root.vktexture; tex && (!tex->owner_data || tex->owner_data != root.get_bitmap_data()))
		root.vktexture = nullptr;
	if (!root.vktexture)
		vks_loadbmtexture_f(bm, vulkan_texture_filter{static_cast<uint8_t>(CGameCfg.TexFilt)}, CGameCfg.TexAnisotropy, edgepad);
	return root.vktexture;
}

vks_texture *vks_load_temporary_texture(const grs_bitmap &src, const uint32_t sx, const uint32_t sy, const uint32_t w, const uint32_t h, const vulkan_texture_filter texfilt)
{
	const uint32_t rowsize = src.bm_rowsize;
	return vks_create_rgba_texture(vks_expand_paletted(src.get_bitmap_data() + static_cast<size_t>(sy) * rowsize + sx, rowsize, w, h, src.get_flags(), gr_current_pal), w, h, texfilt, false, false);
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

void vks_release_bitmap_textures()
{
	/* Bitmaps notice the release through owner_data and re-upload. */
	for (auto &tex : texture_pool)
		if (tex.owner_data)
			vks_free_texture(tex);
}

void vks_flush_pending_texture_frees(uint32_t frame)
{
	auto &v = pending_free[frame % VKS_PENDING_SLOTS];
	if (v.empty())
		return;
	/* Textures released while no frame was recording may be referenced by the
	 * other slot's frame, which this slot's fence does not cover, or by
	 * uploads recorded since, which no frame has submitted yet. */
	vks_submit_pending_uploads();
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
	for (auto &by_filter : vk_texture_samplers)
		for (auto &by_anisotropy : by_filter)
			for (auto &sampler : by_anisotropy)
				if (sampler)
				{
					vkDestroySampler(vk_device, sampler, nullptr);
					sampler = VK_NULL_HANDLE;
				}
	for (auto &slot : upload_slots)
	{
		for (auto &chunk : slot.chunks)
			destroy_staging_chunk(chunk);
		slot = {};
	}
	if (vk_upload_pool)
	{
		vkDestroyCommandPool(vk_device, vk_upload_pool, nullptr);
		vk_upload_pool = VK_NULL_HANDLE;
	}
}

/* 1x1 opaque-white texture used as the sampler source for flat primitives
 * (filled rects, lines, pixels): modulating by white passes the vertex color
 * through unchanged. Lives outside the texture pool. */
static vks_texture vk_white_texture;

void vks_init_white_texture()
{
	if (!vks_init_texture(vk_white_texture, 1, 1, 1, {vulkan_texture_filter::classic, false}))
		Error("Vulkan: Failed to create white texture");
	constexpr uint8_t white[4] = {255, 255, 255, 255};
	vks_upload_rgba(vk_white_texture.image, white, 1, 1, 1);
	vk_white_texture.width = vk_white_texture.height = 1;
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
