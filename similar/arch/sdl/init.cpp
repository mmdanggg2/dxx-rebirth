/*
 * This file is part of the DXX-Rebirth project <https://github.com/dxx-rebirth/dxx-rebirth/>.
 * It is copyright by its individual contributors, as recorded in the
 * project's Git history.  See COPYING.txt at the top level for license
 * terms and a link to the Git history.
 */
// Holds the main init and de-init functions for arch-related program parts

#include <SDL.h>
#include <csignal>
#include <cstdlib>
#include "songs.h"
#include "key.h"
#include "digi.h"
#include "mouse.h"
#include "joy.h"
#include "gr.h"
#include "dxxerror.h"
#include "text.h"
#include "args.h"
#include "window.h"
#include "dxxsconf.h"

#if DXX_USE_SDLIMAGE
#include <SDL_image.h>
#endif

#ifndef _WIN32
namespace {
/* Counts quit-signal deliveries (SIGINT and SIGTERM share one counter).
 *
 * SDL's built-in handler turns the first of these into an SDL_QUIT event,
 * which brings up the abort-game dialog. That only works while the main
 * loop is pumping events: if the game is wedged the event is never
 * dispatched and the process hangs with a quit request sitting unread.
 * This replaces SDL's handler with our own. The first signal still
 * enqueues SDL_QUIT, but a second signal terminates immediately from
 * signal context -- so a stuck game can always be forced out (e.g.
 * pressing Ctrl-C twice in the terminal). */
volatile sig_atomic_t quit_signal_count;

static void quit_signal_handler(int)
{
	const auto n = quit_signal_count + 1;
	quit_signal_count = n;
	if (n >= 2)
		std::_Exit(1);
	SDL_Event event{};
	event.type = SDL_QUIT;
	SDL_PushEvent(&event);
}
}
#endif
namespace dsx {

static void arch_close(void)
{
	songs_uninit();

	gr_close();

#if DXX_MAX_JOYSTICKS
	if (!CGameArg.CtlNoJoystick)
	{
		joy_close();
#if SDL_MAJOR_VERSION == 2
		gamecontroller_close();
#endif
	}
#endif

	if (!CGameArg.CtlNoMouse)
		mouse_close();

	if (!CGameArg.SndNoSound)
	{
		digi_close();
	}
#if DXX_USE_SDLIMAGE
	IMG_Quit();
#endif
	SDL_Quit();
}

arch_atexit::~arch_atexit()
{
	arch_close();
}

arch_atexit arch_init()
{
	int t;

	if (SDL_Init(SDL_INIT_VIDEO) < 0)
		Error("SDL library initialisation failed: %s.",SDL_GetError());
#ifndef _WIN32
	/* SDL installs its own SIGINT/SIGTERM handlers during SDL_Init. Replace
	 * both so that a second quit signal force-exits even when the main loop
	 * is stuck. sigaction (rather than signal) keeps the handler installed
	 * across deliveries regardless of SysV/BSD signal() semantics. */
	struct sigaction sa{};
	sa.sa_handler = quit_signal_handler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART;
	sigaction(SIGINT, &sa, nullptr);
	sigaction(SIGTERM, &sa, nullptr);
#endif
#if DXX_USE_SDLIMAGE
	IMG_Init(0);
#endif
#if SDL_MAJOR_VERSION == 2
	/* In SDL1, grabbing input grabbed both the keyboard and the mouse.
	 * Many game management keys assume a keyboard grab.
	 * Tell SDL2 to grab the keyboard.
	 *
	 * Unlike with SDL1, players have the option of overriding this grab
	 * by setting an environment variable.  In SDL1, the only choice was
	 * to skip both the keyboard grab and the mouse grab.  Now, players
	 * can enable grabbing in the UI, but disable keyboard grab with the
	 * environment variable.
	 */
	SDL_SetHint(SDL_HINT_GRAB_KEYBOARD, "1");
	/* Gameplay continues regardless of focus, so keep the window
	 * visible.
	 */
	SDL_SetHint(SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS, "0");
	/* Support the Alt+Shift+F4 hotkey for renaming the Guide-Bot
	 */
	SDL_SetHint(SDL_HINT_WINDOWS_NO_CLOSE_ON_ALT_F4, "1");
#endif

	key_init();

	digi_select_system();

	if (!CGameArg.SndNoSound)
		digi_init();

	if (!CGameArg.CtlNoMouse)
		mouse_init();

#if DXX_MAX_JOYSTICKS
	if (!CGameArg.CtlNoJoystick)
	{
		joy_init();
#if SDL_MAJOR_VERSION == 2
		gamecontroller_init();
#endif
	}
#endif

	if ((t = gr_init()) != 0)
		Error(TXT_CANT_INIT_GFX,t);

	return {};
}

}
