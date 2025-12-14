#pragma once

#include "granite/types.h"
#include "granite/error.h"
#include "granite/backend.h"

#include <vector>
#include <span>
#include <memory>
#include <numeric>
#include <cassert>

namespace granite {

// =============================================================================
// Tensor Class
// =============================================================================

class Tensor {
public:
    // Default constructor creates an empty tensor
    Tensor() = default;

    // -------------------------------------------------------------------------
    // Factory Methods
    // -------------------------------------------------------------------------

    /// Allocate a new tensor with the given shape and dtype
    static Result<Tensor> allocate(std::span<const int64_t> shape,
                                    DataType dtype,
                                    IComputeBackend* backend);

    /// Create a tensor that wraps an existing buffer
    static Tensor from_buffer(BufferHandle buffer,
                              std::span<const int64_t> shape,
                              DataType dtype,
                              IComputeBackend* backend,
                              std::span<const int64_t> strides = {});

    // -------------------------------------------------------------------------
    // Properties
    // -------------------------------------------------------------------------

    /// Get the shape of the tensor
    [[nodiscard]] std::span<const int64_t> shape() const { return shape_; }

    /// Get the strides of the tensor (in elements, not bytes)
    [[nodiscard]] std::span<const int64_t> strides() const { return strides_; }

    /// Get the data type
    [[nodiscard]] DataType dtype() const { return dtype_; }

    /// Get the number of dimensions
    [[nodiscard]] size_t ndim() const { return shape_.size(); }

    /// Get the size of a specific dimension
    [[nodiscard]] int64_t size(size_t dim) const {
        assert(dim < shape_.size());
        return shape_[dim];
    }

    /// Get the total number of elements
    [[nodiscard]] size_t numel() const {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(),
                               int64_t{1}, std::multiplies<>());
    }

    /// Get the size in bytes
    [[nodiscard]] size_t size_bytes() const {
        size_t n = numel();
        if (dtype_ == DataType::INT4 || dtype_ == DataType::UINT4) {
            return (n + 1) / 2;  // 2 elements per byte
        }
        return n * dtype_size(dtype_);
    }

    /// Check if the tensor is contiguous in memory
    [[nodiscard]] bool is_contiguous() const;

    /// Check if the tensor is empty (no elements)
    [[nodiscard]] bool empty() const { return numel() == 0; }

    /// Check if the tensor is valid (has a buffer)
    [[nodiscard]] bool valid() const { return buffer_.valid(); }

    // -------------------------------------------------------------------------
    // Buffer Access
    // -------------------------------------------------------------------------

    /// Get the underlying buffer handle
    [[nodiscard]] BufferHandle buffer() const { return buffer_; }

    /// Get the offset into the buffer (in bytes)
    [[nodiscard]] size_t buffer_offset() const { return offset_; }

    /// Get the backend this tensor belongs to
    [[nodiscard]] IComputeBackend* backend() const { return backend_; }

    // -------------------------------------------------------------------------
    // Data Access (CPU)
    // -------------------------------------------------------------------------

    /// Map the buffer and return a typed pointer (for Shared/Managed memory)
    template<typename T>
    Result<T*> data() {
        if (!backend_) {
            return Error(ErrorCode::NullPointer, "Tensor has no backend");
        }
        auto result = backend_->map_buffer(buffer_);
        if (!result.ok()) {
            return result.error();
        }
        return reinterpret_cast<T*>(static_cast<uint8_t*>(result.value()) + offset_);
    }

    template<typename T>
    Result<const T*> data() const {
        if (!backend_) {
            return Error(ErrorCode::NullPointer, "Tensor has no backend");
        }
        auto result = backend_->map_buffer(buffer_);
        if (!result.ok()) {
            return result.error();
        }
        return reinterpret_cast<const T*>(static_cast<uint8_t*>(result.value()) + offset_);
    }

    /// Unmap the buffer
    void unmap() {
        if (backend_) {
            backend_->unmap_buffer(buffer_);
        }
    }

    // -------------------------------------------------------------------------
    // View Operations (no copy)
    // -------------------------------------------------------------------------

    /// Create a view with a different shape (must have same number of elements)
    [[nodiscard]] Result<Tensor> view(std::span<const int64_t> new_shape) const;

    /// Slice along a dimension
    [[nodiscard]] Result<Tensor> slice(size_t dim, int64_t start, int64_t end) const;

    /// Transpose two dimensions
    [[nodiscard]] Result<Tensor> transpose(size_t dim0, size_t dim1) const;

    /// Squeeze: remove dimensions of size 1
    [[nodiscard]] Tensor squeeze() const;

    /// Unsqueeze: add a dimension of size 1 at the given position
    [[nodiscard]] Result<Tensor> unsqueeze(size_t dim) const;

    // -------------------------------------------------------------------------
    // Copy Operations
    // -------------------------------------------------------------------------

    /// Create a contiguous copy of this tensor
    [[nodiscard]] Result<Tensor> clone() const;

    /// Copy data from another tensor
    Result<void> copy_from(const Tensor& src);

    /// Copy this tensor to a different backend
    [[nodiscard]] Result<Tensor> to(IComputeBackend* target_backend) const;

private:
    BufferHandle buffer_;
    size_t offset_ = 0;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DataType dtype_ = DataType::FP32;
    IComputeBackend* backend_ = nullptr;

    // Compute default (contiguous) strides for a shape
    static std::vector<int64_t> compute_strides(std::span<const int64_t> shape);
};

// =============================================================================
// Utility Functions
// =============================================================================

/// Compute the number of elements from a shape
inline size_t shape_numel(std::span<const int64_t> shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(),
                           int64_t{1}, std::multiplies<>());
}

/// Check if two shapes are broadcast-compatible
bool shapes_broadcastable(std::span<const int64_t> a,
                          std::span<const int64_t> b);

/// Compute the result shape of broadcasting two shapes
std::vector<int64_t> broadcast_shapes(std::span<const int64_t> a,
                                       std::span<const int64_t> b);

}  // namespace granite
