#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace granite {

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
        init_logging();
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
