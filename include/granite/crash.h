#pragma once

#include <cstddef>

namespace granite {

using CrashCallback = void(*)(const char* reason);

/// Install Granite's terminate handler and report via the crash callback.
void install_crash_handler();

/// Register a crash callback (called from terminate handler).
void set_crash_callback(CrashCallback callback);

/// Get current crash callback.
CrashCallback get_crash_callback();

/// Invoke the crash callback manually (for app-level crash handlers).
void notify_crash(const char* reason);

}  // namespace granite
