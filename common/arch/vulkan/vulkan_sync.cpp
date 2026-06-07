/*
 * This file is part of the DXX-Rebirth project <https://www.dxx-rebirth.com/>.
 * It is copyright by its individual contributors, as recorded in the
 * project's Git history.  See COPYING.txt at the top level for license
 * terms and a link to the Git history.
 */

#include "dxxsconf.h"
#if DXX_USE_VULKAN

#include "vulkan_sync.h"
#include "vulkan_init.h"
#include "timer.h"
#include "error.h"
#include "args.h"
#include "console.h"
#include "config.h"
#include "game.h"
#include "maths.h"
#include "multi.h"

#include <cstdlib>

namespace dcx {

vks_sync::~vks_sync()
{
	if (current_fence)
		con_puts(CON_URGENT, "DXX-Rebirth: Vulkan: fence was never destroyed!");
}

void vks_sync::before_swap()
{
	/* Wait for the GPU to finish the previous frame */
	if (current_fence) {
		vkWaitForFences(vk_device, 1, &current_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(vk_device, 1, &current_fence);
	}
}

void vks_sync::after_swap()
{
	/* Create a new fence for the next before_swap */
	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	VkResult result = vkCreateFence(vk_device, &fenceInfo, nullptr, &current_fence);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to create fence");;
}

void vks_sync::init()
{
	current_fence = VK_NULL_HANDLE;
}

void vks_sync::deinit()
{
	if (current_fence) {
		vkDestroyFence(vk_device, current_fence, nullptr);
		current_fence = VK_NULL_HANDLE;
	}
}

void vks_sync::record_frame(uint32_t frame_index)
{
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = VK_NULL_HANDLE;  // Would be created from pool
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer cmdBuffer;
	VkResult result = vkAllocateCommandBuffers(vk_device, &allocInfo, &cmdBuffer);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to allocate command buffer");;

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(cmdBuffer, &beginInfo);

	/* Image memory barrier: swapchain image -> COLOR_ATTACHMENT_OPTIMAL */
	VkImageMemoryBarrier imageBarrier{};
	imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	imageBarrier.subresourceRange.levelCount = 1;
	imageBarrier.subresourceRange.layerCount = 1;

	/* Would need the actual swapchain image index */
	(void)frame_index;
	(void)imageBarrier;

	vkEndCommandBuffer(cmdBuffer);

	/* Submit to graphics queue */
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &vk_image_available_semaphores[frame_index];
	VkPipelineStageFlags waitDstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		submitInfo.pWaitDstStageMask = &waitDstStage;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmdBuffer;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &vk_render_finished_semaphores[frame_index];

	result = vkQueueSubmit(vk_graphics_queue, 1, &submitInfo, current_fence);
	if (!( result == VK_SUCCESS )) Error("Vulkan: Failed to submit command buffer");

	vkFreeCommandBuffers(vk_device, VK_NULL_HANDLE, 1, &cmdBuffer);
}

vks_sync vulkan_sync_helper;

} /* namespace dcx */

#endif /* DXX_USE_VULKAN */
