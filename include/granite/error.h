#pragma once

#include <string>
#include <string_view>
#include <variant>
#include <optional>
#include <source_location>
#include <utility>

namespace granite {

// =============================================================================
// Error Codes
// =============================================================================

enum class ErrorCode {
    Success = 0,

    // File/IO errors
    FileNotFound,
    InvalidFormat,
    IOError,

    // Memory errors
    OutOfMemory,
    AllocationFailed,
    BufferTooSmall,

    // Compute errors
    BackendNotSupported,
    BackendNotInitialized,
    OperatorNotImplemented,
    ShaderCompilationFailed,
    ExecutionFailed,
    Timeout,

    // Validation errors
    InvalidShape,
    ShapeMismatch,
    DTypeMismatch,
    InvalidArgument,
    NullPointer,

    // Graph errors
    InvalidGraph,
    CycleDetected,
    InvalidState,

    // Model errors
    UnsupportedModel,
    CorruptedWeights,
    MissingOperator,

    // Internal errors
    InternalError,
    NotImplemented,
};

/// Returns a string representation of the error code
constexpr const char* error_code_name(ErrorCode code) {
    switch (code) {
        case ErrorCode::Success:              return "Success";
        case ErrorCode::FileNotFound:         return "FileNotFound";
        case ErrorCode::InvalidFormat:        return "InvalidFormat";
        case ErrorCode::IOError:              return "IOError";
        case ErrorCode::OutOfMemory:          return "OutOfMemory";
        case ErrorCode::AllocationFailed:     return "AllocationFailed";
        case ErrorCode::BufferTooSmall:       return "BufferTooSmall";
        case ErrorCode::BackendNotSupported:  return "BackendNotSupported";
        case ErrorCode::BackendNotInitialized: return "BackendNotInitialized";
        case ErrorCode::OperatorNotImplemented: return "OperatorNotImplemented";
        case ErrorCode::ShaderCompilationFailed: return "ShaderCompilationFailed";
        case ErrorCode::ExecutionFailed:      return "ExecutionFailed";
        case ErrorCode::Timeout:              return "Timeout";
        case ErrorCode::InvalidShape:         return "InvalidShape";
        case ErrorCode::ShapeMismatch:        return "ShapeMismatch";
        case ErrorCode::DTypeMismatch:        return "DTypeMismatch";
        case ErrorCode::InvalidArgument:      return "InvalidArgument";
        case ErrorCode::NullPointer:          return "NullPointer";
        case ErrorCode::InvalidGraph:         return "InvalidGraph";
        case ErrorCode::CycleDetected:        return "CycleDetected";
        case ErrorCode::InvalidState:         return "InvalidState";
        case ErrorCode::UnsupportedModel:     return "UnsupportedModel";
        case ErrorCode::CorruptedWeights:     return "CorruptedWeights";
        case ErrorCode::MissingOperator:      return "MissingOperator";
        case ErrorCode::InternalError:        return "InternalError";
        case ErrorCode::NotImplemented:       return "NotImplemented";
    }
    return "Unknown";
}

// =============================================================================
// Error Class
// =============================================================================

class Error {
public:
    Error(ErrorCode code,
          std::string message,
          std::source_location location = std::source_location::current())
        : code_(code)
        , message_(std::move(message))
        , location_(location)
    {}

    Error(ErrorCode code,
          std::source_location location = std::source_location::current())
        : code_(code)
        , message_(error_code_name(code))
        , location_(location)
    {}

    [[nodiscard]] ErrorCode code() const { return code_; }
    [[nodiscard]] const std::string& message() const { return message_; }
    [[nodiscard]] const std::source_location& location() const { return location_; }

    [[nodiscard]] std::string to_string() const {
        return std::string(error_code_name(code_)) + ": " + message_ +
               " (at " + location_.file_name() + ":" +
               std::to_string(location_.line()) + ")";
    }

private:
    ErrorCode code_;
    std::string message_;
    std::source_location location_;
};

// =============================================================================
// Result<T> Template
// =============================================================================

template<typename T>
class Result {
public:
    // Construct from value
    Result(T value) : data_(std::move(value)) {}

    // Construct from error
    Result(Error error) : data_(std::move(error)) {}

    // Construct from error code (convenience)
    Result(ErrorCode code,
           std::source_location loc = std::source_location::current())
        : data_(Error(code, loc)) {}

    [[nodiscard]] bool ok() const {
        return std::holds_alternative<T>(data_);
    }

    explicit operator bool() const { return ok(); }

    [[nodiscard]] T& value() & {
        return std::get<T>(data_);
    }

    [[nodiscard]] const T& value() const& {
        return std::get<T>(data_);
    }

    [[nodiscard]] T&& value() && {
        return std::get<T>(std::move(data_));
    }

    [[nodiscard]] T take() {
        return std::get<T>(std::move(data_));
    }

    [[nodiscard]] const Error& error() const {
        return std::get<Error>(data_);
    }

    [[nodiscard]] T value_or(T default_value) const {
        if (ok()) {
            return std::get<T>(data_);
        }
        return default_value;
    }

    // Monadic map: transform the value if present
    template<typename F>
    auto map(F&& f) -> Result<decltype(f(std::declval<T>()))> {
        using U = decltype(f(std::declval<T>()));
        if (ok()) {
            return Result<U>(f(std::get<T>(data_)));
        }
        return Result<U>(std::get<Error>(data_));
    }

    // Monadic and_then: chain Results
    template<typename F>
    auto and_then(F&& f) -> decltype(f(std::declval<T>())) {
        if (ok()) {
            return f(std::get<T>(data_));
        }
        return decltype(f(std::declval<T>()))(std::get<Error>(data_));
    }

private:
    std::variant<T, Error> data_;
};

// =============================================================================
// Result<void> Specialization
// =============================================================================

template<>
class Result<void> {
public:
    Result() : error_(std::nullopt) {}

    Result(Error error) : error_(std::move(error)) {}

    Result(ErrorCode code,
           std::source_location loc = std::source_location::current())
        : error_(Error(code, loc)) {}

    [[nodiscard]] bool ok() const { return !error_.has_value(); }

    explicit operator bool() const { return ok(); }

    [[nodiscard]] const Error& error() const { return error_.value(); }

private:
    std::optional<Error> error_;
};

// =============================================================================
// Error Propagation Macros
// =============================================================================

// Try an expression and return early if it fails
#define GRANITE_TRY(expr)                           \
    do {                                            \
        auto _granite_result = (expr);              \
        if (!_granite_result.ok()) {                \
            return _granite_result.error();         \
        }                                           \
    } while (0)

// Try an expression and assign to variable if successful, return early if not
#define GRANITE_TRY_ASSIGN(var, expr)               \
    auto _granite_result_##var = (expr);            \
    if (!_granite_result_##var.ok()) {              \
        return _granite_result_##var.error();       \
    }                                               \
    auto var = std::move(_granite_result_##var).take()

// Create an error with the current location
#define GRANITE_ERROR(code, msg) \
    ::granite::Error((code), (msg), std::source_location::current())

// Return an error
#define GRANITE_FAIL(code, msg) \
    return GRANITE_ERROR(code, msg)

}  // namespace granite
