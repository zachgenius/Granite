#include "granite/tensor.h"
#include "granite/log.h"

#include <algorithm>

namespace granite {

// =============================================================================
// Static Helper Functions
// =============================================================================

std::vector<int64_t> Tensor::compute_strides(std::span<const int64_t> shape) {
    if (shape.empty()) {
        return {};
    }

    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;

    // Row-major order: last dimension is contiguous
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}

// =============================================================================
// Factory Methods
// =============================================================================

Result<Tensor> Tensor::allocate(std::span<const int64_t> shape,
                                 DataType dtype,
                                 IComputeBackend* backend) {
    if (!backend) {
        return Error(ErrorCode::NullPointer, "Backend is null");
    }

    if (!backend->is_initialized()) {
        return Error(ErrorCode::BackendNotInitialized, "Backend is not initialized");
    }

    // Calculate buffer size
    size_t numel = shape_numel(shape);
    size_t size_bytes;

    if (dtype == DataType::INT4 || dtype == DataType::UINT4) {
        size_bytes = (numel + 1) / 2;  // 2 elements per byte
    } else {
        size_bytes = numel * dtype_size(dtype);
    }

    // Allocate buffer
    auto buffer_result = backend->create_buffer(BufferDesc::managed(size_bytes));
    if (!buffer_result.ok()) {
        return buffer_result.error();
    }

    Tensor tensor;
    tensor.buffer_ = buffer_result.value();
    tensor.offset_ = 0;
    tensor.shape_ = std::vector<int64_t>(shape.begin(), shape.end());
    tensor.strides_ = compute_strides(shape);
    tensor.dtype_ = dtype;
    tensor.backend_ = backend;

    GRANITE_LOG_DEBUG("Allocated tensor: shape=[{}], dtype={}, size={}",
                      fmt::join(shape, ", "), dtype_name(dtype), size_bytes);

    return tensor;
}

Tensor Tensor::from_buffer(BufferHandle buffer,
                           std::span<const int64_t> shape,
                           DataType dtype,
                           IComputeBackend* backend,
                           std::span<const int64_t> strides) {
    Tensor tensor;
    tensor.buffer_ = buffer;
    tensor.offset_ = 0;
    tensor.shape_ = std::vector<int64_t>(shape.begin(), shape.end());
    tensor.dtype_ = dtype;
    tensor.backend_ = backend;

    if (strides.empty()) {
        tensor.strides_ = compute_strides(shape);
    } else {
        tensor.strides_ = std::vector<int64_t>(strides.begin(), strides.end());
    }

    return tensor;
}

// =============================================================================
// Properties
// =============================================================================

bool Tensor::is_contiguous() const {
    if (shape_.empty()) {
        return true;
    }

    auto expected = compute_strides(shape_);
    return strides_ == expected;
}

// =============================================================================
// View Operations
// =============================================================================

Result<Tensor> Tensor::view(std::span<const int64_t> new_shape) const {
    // Check that the total number of elements matches
    size_t old_numel = numel();
    size_t new_numel = shape_numel(new_shape);

    if (old_numel != new_numel) {
        return Error(ErrorCode::InvalidShape,
                     fmt::format("Cannot view tensor with {} elements as shape with {} elements",
                                 old_numel, new_numel));
    }

    // Views only work on contiguous tensors
    if (!is_contiguous()) {
        return Error(ErrorCode::InvalidShape, "Cannot view non-contiguous tensor");
    }

    Tensor result = *this;
    result.shape_ = std::vector<int64_t>(new_shape.begin(), new_shape.end());
    result.strides_ = compute_strides(new_shape);

    return result;
}

Result<Tensor> Tensor::slice(size_t dim, int64_t start, int64_t end) const {
    if (dim >= shape_.size()) {
        return Error(ErrorCode::InvalidArgument,
                     fmt::format("Dimension {} out of range for tensor with {} dimensions",
                                 dim, shape_.size()));
    }

    int64_t dim_size = shape_[dim];

    // Handle negative indices
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;

    // Clamp to valid range
    start = std::clamp(start, int64_t{0}, dim_size);
    end = std::clamp(end, int64_t{0}, dim_size);

    if (start >= end) {
        return Error(ErrorCode::InvalidArgument,
                     fmt::format("Invalid slice range [{}, {})", start, end));
    }

    Tensor result = *this;
    result.shape_[dim] = end - start;

    // Adjust offset
    size_t element_size = dtype_size(dtype_);
    result.offset_ += start * strides_[dim] * element_size;

    return result;
}

Result<Tensor> Tensor::transpose(size_t dim0, size_t dim1) const {
    if (dim0 >= shape_.size() || dim1 >= shape_.size()) {
        return Error(ErrorCode::InvalidArgument,
                     fmt::format("Transpose dimensions ({}, {}) out of range for tensor with {} dimensions",
                                 dim0, dim1, shape_.size()));
    }

    Tensor result = *this;
    std::swap(result.shape_[dim0], result.shape_[dim1]);
    std::swap(result.strides_[dim0], result.strides_[dim1]);

    return result;
}

Tensor Tensor::squeeze() const {
    Tensor result = *this;
    result.shape_.clear();
    result.strides_.clear();

    for (size_t i = 0; i < shape_.size(); ++i) {
        if (shape_[i] != 1) {
            result.shape_.push_back(shape_[i]);
            result.strides_.push_back(strides_[i]);
        }
    }

    return result;
}

Result<Tensor> Tensor::unsqueeze(size_t dim) const {
    if (dim > shape_.size()) {
        return Error(ErrorCode::InvalidArgument,
                     fmt::format("Unsqueeze dimension {} out of range for tensor with {} dimensions",
                                 dim, shape_.size()));
    }

    Tensor result = *this;
    result.shape_.insert(result.shape_.begin() + dim, 1);

    // Compute stride for the new dimension
    int64_t new_stride = (dim < strides_.size()) ? strides_[dim] : 1;
    result.strides_.insert(result.strides_.begin() + dim, new_stride);

    return result;
}

// =============================================================================
// Copy Operations
// =============================================================================

Result<Tensor> Tensor::clone() const {
    if (!backend_) {
        return Error(ErrorCode::NullPointer, "Tensor has no backend");
    }

    // Allocate new tensor
    auto result = Tensor::allocate(shape_, dtype_, backend_);
    if (!result.ok()) {
        return result.error();
    }

    // Copy data
    auto copy_result = result.value().copy_from(*this);
    if (!copy_result.ok()) {
        return copy_result.error();
    }

    return result;
}

Result<void> Tensor::copy_from(const Tensor& src) {
    if (!backend_ || !src.backend_) {
        return Error(ErrorCode::NullPointer, "Tensor has no backend");
    }

    if (numel() != src.numel()) {
        return Error(ErrorCode::ShapeMismatch,
                     fmt::format("Cannot copy tensor with {} elements to tensor with {} elements",
                                 src.numel(), numel()));
    }

    if (dtype_ != src.dtype_) {
        return Error(ErrorCode::DTypeMismatch,
                     fmt::format("Cannot copy {} tensor to {} tensor",
                                 dtype_name(src.dtype_), dtype_name(dtype_)));
    }

    // TODO: Handle cross-backend copies
    if (backend_ != src.backend_) {
        return Error(ErrorCode::NotImplemented, "Cross-backend tensor copy not yet implemented");
    }

    // Copy buffer data
    return backend_->copy_buffer(src.buffer_, buffer_, size_bytes(),
                                  src.offset_, offset_);
}

Result<Tensor> Tensor::to(IComputeBackend* target_backend) const {
    if (!target_backend) {
        return Error(ErrorCode::NullPointer, "Target backend is null");
    }

    if (target_backend == backend_) {
        // Same backend, just clone
        return clone();
    }

    // Allocate on target backend
    auto result = Tensor::allocate(shape_, dtype_, target_backend);
    if (!result.ok()) {
        return result.error();
    }

    // Copy via CPU (read from source, write to destination)
    std::vector<uint8_t> temp(size_bytes());

    auto read_result = backend_->read_buffer(buffer_, temp.data(), temp.size(), offset_);
    if (!read_result.ok()) {
        return read_result.error();
    }

    auto write_result = target_backend->write_buffer(result.value().buffer_,
                                                      temp.data(), temp.size(), 0);
    if (!write_result.ok()) {
        return write_result.error();
    }

    return result;
}

// =============================================================================
// Utility Functions
// =============================================================================

bool shapes_broadcastable(std::span<const int64_t> a, std::span<const int64_t> b) {
    auto it_a = a.rbegin();
    auto it_b = b.rbegin();

    while (it_a != a.rend() && it_b != b.rend()) {
        if (*it_a != *it_b && *it_a != 1 && *it_b != 1) {
            return false;
        }
        ++it_a;
        ++it_b;
    }

    return true;
}

std::vector<int64_t> broadcast_shapes(std::span<const int64_t> a,
                                       std::span<const int64_t> b) {
    size_t result_size = std::max(a.size(), b.size());
    std::vector<int64_t> result(result_size);

    auto it_a = a.rbegin();
    auto it_b = b.rbegin();
    auto it_r = result.rbegin();

    while (it_r != result.rend()) {
        int64_t dim_a = (it_a != a.rend()) ? *it_a++ : 1;
        int64_t dim_b = (it_b != b.rend()) ? *it_b++ : 1;
        *it_r++ = std::max(dim_a, dim_b);
    }

    return result;
}

}  // namespace granite
