/*
 * This file is part of the DXX-Rebirth project <https://www.dxx-rebirth.com/>.
 * It is copyright by its individual contributors, as recorded in the
 * project's Git history.  See COPYING.txt at the top level for license
 * terms and a link to the Git history.
 */

#ifndef _D_VK_H_
#define _D_VK_H_

#include "dxxsconf.h"
#if !DXX_USE_VULKAN
#error "This file can only be included in Vulkan enabled builds."
#endif

#ifdef _WIN32
#	include <vulkan/vulkan.h>
#elif defined(__APPLE__)
#	include <vulkan/vulkan.h>
#else
#	include <vulkan/vulkan.h>
#endif

#endif /* _D_VK_H_ */
