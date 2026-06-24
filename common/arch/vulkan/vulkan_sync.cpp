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
	/* Frame sync is now handled in vks_start_frame() via vk_in_flight_fences.
	 * This is kept as a no-op to maintain the vks_sync interface. */
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

vks_sync vulkan_sync_helper;

} /* namespace dcx */

#endif /* DXX_USE_VULKAN */
