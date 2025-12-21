#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <cstdlib>
#include <string>

namespace granite {

inline spdlog::level::level_enum parse_log_level(std::string level) {
    std::transform(level.begin(), level.end(), level.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (level == "trace") return spdlog::level::trace;
    if (level == "debug") return spdlog::level::debug;
    if (level == "info") return spdlog::level::info;
    if (level == "warn" || level == "warning") return spdlog::level::warn;
    if (level == "error") return spdlog::level::err;
    if (level == "critical") return spdlog::level::critical;
    if (level == "off") return spdlog::level::off;
    return spdlog::level::info;
}

inline spdlog::level::level_enum log_level_from_env() {
    const char* env = std::getenv("GRANITE_LOG_LEVEL");
    if (env && *env) {
        return parse_log_level(env);
    }
    return spdlog::level::info;
}

/// Initialize the logging system
inline void init_logging(spdlog::level::level_enum level = spdlog::level::info) {
    auto logger = spdlog::stdout_color_mt("granite");
    logger->set_level(level);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
    spdlog::set_default_logger(logger);
}

/// Get the granite logger
inline std::shared_ptr<spdlog::logger> get_logger() {
    auto logger = spdlog::get("granite");
    if (!logger) {
        init_logging(log_level_from_env());
        logger = spdlog::get("granite");
    }
    return logger;
}

/// Set the log level
inline void set_log_level(spdlog::level::level_enum level) {
    get_logger()->set_level(level);
}

}  // namespace granite

// Convenience macros
#define GRANITE_LOG_TRACE(...)    SPDLOG_LOGGER_TRACE(::granite::get_logger(), __VA_ARGS__)
#define GRANITE_LOG_DEBUG(...)    SPDLOG_LOGGER_DEBUG(::granite::get_logger(), __VA_ARGS__)
#define GRANITE_LOG_INFO(...)     SPDLOG_LOGGER_INFO(::granite::get_logger(), __VA_ARGS__)
#define GRANITE_LOG_WARN(...)     SPDLOG_LOGGER_WARN(::granite::get_logger(), __VA_ARGS__)
#define GRANITE_LOG_ERROR(...)    SPDLOG_LOGGER_ERROR(::granite::get_logger(), __VA_ARGS__)
