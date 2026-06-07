/*
 * This file is part of the DXX-Rebirth project <https://www.dxx-rebirth.com/>.
 * It is copyright by its individual contributors, as recorded in the
 * project's Git history.  See COPYING.txt at the top level for license
 * terms and a link to the Git history.
 */

#ifndef _VULKAN_SYNC_H_
#define _VULKAN_SYNC_H_

#include "dxxsconf.h"
#if !DXX_USE_VULKAN
#error "This file can only be included in Vulkan enabled builds."
#endif

#include "d_vk.h"
#include <memory>

namespace dcx {

/* Vulkan command buffer submission helper.
 * Manages command buffer recording and submission for a single frame.
 */
class vks_sync
{
public:
	vks_sync() = default;
	~vks_sync();

	/* Called before swap buffers — waits for previous frame's GPU work */
	void before_swap();

	/* Called after swap buffers — creates a fence for next before_swap */
	void after_swap();

	/* Initialize synchronization objects */
	void init();

	/* Destroy all synchronization objects */
	void deinit();

	/* Record one frame's worth of commands and submit to queue */
	void record_frame(uint32_t frame_index);

private:
	VkFence current_fence{};
	VkSemaphore current_image_semaphore{};
	VkSemaphore current_render_semaphore{};
	VkCommandBuffer current_command_buffer{};
};

extern vks_sync vulkan_sync_helper;

} /* namespace dcx */

#endif /* _VULKAN_SYNC_H_ */
