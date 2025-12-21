#include "granite/crash.h"

#include <atomic>
#include <cstdlib>
#include <exception>

namespace granite {

namespace {
std::atomic<CrashCallback> g_crash_callback{nullptr};
std::atomic<bool> g_installed{false};
}

void set_crash_callback(CrashCallback callback) {
    g_crash_callback.store(callback, std::memory_order_release);
}

CrashCallback get_crash_callback() {
    return g_crash_callback.load(std::memory_order_acquire);
}

void notify_crash(const char* reason) {
    auto callback = g_crash_callback.load(std::memory_order_acquire);
    if (callback) {
        callback(reason ? reason : "unknown");
    }
}

void install_crash_handler() {
    bool expected = false;
    if (!g_installed.compare_exchange_strong(expected, true)) {
        return;
    }

    std::set_terminate([]() {
        notify_crash("std::terminate");
        std::abort();
    });
}

}  // namespace granite
