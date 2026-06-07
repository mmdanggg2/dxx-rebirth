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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace dcx {

static constexpr uint32_t MAX_TEXTURES = 512;
static std::array<vks_texture, MAX_TEXTURES> texture_pool{};
static std::vector<size_t> free_textures;

static size_t allocate_texture_slot()
{
	if (!free_textures.empty()) {
		size_t slot = free_textures.back();
		free_textures.pop_back();
		return slot;
	}
	for (size_t i = 0; i < MAX_TEXTURES; i++) {
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

void vks_init_texture(vks_texture &t, uint32_t w, uint32_t h, int flags)
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
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
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
}

static void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
	VkCommandBuffer commandBuffer = VK_NULL_HANDLE;  // Would be allocated from pool

	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.layerCount = 1;
	region.imageExtent.width = width;
	region.imageExtent.height = height;
	region.imageExtent.depth = 1;

	/* Transition image to TRANSFER_DST_OPTIMAL */
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.layerCount = 1;

	vkCmdPipelineBarrier(commandBuffer,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, nullptr, 0, nullptr,
		1, &barrier);

	vkCmdCopyBufferToImage(commandBuffer, buffer, image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	/* Transition to SHADER_READ_ONLY_OPTIMAL */
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	vkCmdPipelineBarrier(commandBuffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		0, 0, nullptr, 0, nullptr,
		1, &barrier);
}

void vks_loadbmtexture_f(grs_bitmap &bm, vulkan_texture_filter /*texfilt*/, bool /*texanis*/, bool /*edgepad*/)
{
	/* Allocate texture slot and init */
	vks_texture *tex = vks_get_free_texture();

	/* Get pixel data from bitmap */
	uint8_t *pixels = bm.get_bitmap_data();
	uint32_t w = bm.bm_w;
	uint32_t h = bm.bm_h;

	vks_init_texture(*tex, w, h, VKS_FLAG_MIPMAP);

	/* Upload pixels via staging buffer */
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = w * h * 4;
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
	/* Find host-visible memory */
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &memProperties);
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((memReqs.memoryTypeBits & (1 << i)) &&
			(memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
			(memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
			allocInfo.memoryTypeIndex = i;
			break;
		}
	}

	VkDeviceMemory stagingMemory;
	result = vkAllocateMemory(vk_device, &allocInfo, nullptr, &stagingMemory);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to allocate staging memory");

	void *data;
	result = vkMapMemory(vk_device, stagingMemory, 0, bufferInfo.size, 0, &data);
	if (!(result == VK_SUCCESS)) Error(__FILE__, __LINE__, __func__, "Vulkan: Failed to map staging memory");
	std::memcpy(data, pixels, w * h * 4);
	vkUnmapMemory(vk_device, stagingMemory);

	/* Copy buffer to image */
	copyBufferToImage(stagingBuffer, tex->image, w, h);

	/* Cleanup staging */
	vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
	vkFreeMemory(vk_device, stagingMemory, nullptr);

	/* Set bitmap to use Vulkan texture */
	bm.set_type(bm_mode::vulkan);
	tex->numrend++;
}

void vks_freebmtexture(grs_bitmap &bm)
{
	if (bm.get_type() != bm_mode::vulkan)
		return;

	vks_texture *tex = bm.vktexture;
	if (tex) {
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

		/* Return slot to free list */
		free_textures.push_back(static_cast<size_t>(tex - texture_pool.data()));
		bm.vktexture = nullptr;
	}
	bm.set_type(bm_mode::linear);
}

void vks_shutdown_textures()
{
	for (auto &tex : texture_pool) {
		vks_freebmtexture(*reinterpret_cast<grs_bitmap *>(&tex));
	}
	free_textures.clear();
}

} /* namespace dcx */

#endif /* DXX_USE_VULKAN */
