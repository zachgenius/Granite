#include "llm_internal.h"

namespace granite {

void RoPECache::initialize(int max_seq_len, int head_dim, float theta) {
    max_seq_len_ = max_seq_len;
    head_dim_ = head_dim;

    int half_dim = head_dim / 2;
    cos_.resize(max_seq_len * half_dim);
    sin_.resize(max_seq_len * half_dim);

    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim);
            float angle = static_cast<float>(pos) * freq;
            cos_[pos * half_dim + i] = std::cos(angle);
            sin_[pos * half_dim + i] = std::sin(angle);
        }
    }

    GRANITE_LOG_DEBUG("Initialized RoPE cache: max_seq={}, head_dim={}, theta={}",
                      max_seq_len, head_dim, theta);
}

Result<void> RoPECache::apply(Tensor& q, Tensor& k, int start_pos, IComputeBackend* backend) {
    if (!is_initialized()) {
        GRANITE_FAIL(ErrorCode::InvalidState, "RoPE cache not initialized");
    }

    // Q/K shape: [batch, seq_len, num_heads, head_dim]
    int seq_len = static_cast<int>(q.size(1));
    int num_heads_q = static_cast<int>(q.size(2));
    int num_heads_k = static_cast<int>(k.size(2));

    if (start_pos + seq_len > max_seq_len_) {
        GRANITE_FAIL(ErrorCode::InvalidArgument,
                     fmt::format("RoPE position out of range: {} + {} > {}",
                                 start_pos, seq_len, max_seq_len_));
    }

    auto map_q = backend->map_buffer(q.buffer());
    auto map_k = backend->map_buffer(k.buffer());

    if (!map_q.ok() || !map_k.ok()) {
        return Error(ErrorCode::InternalError, "Failed to map buffers");
    }

    auto* q_data = static_cast<float*>(map_q.value());
    auto* k_data = static_cast<float*>(map_k.value());

    int half_dim = head_dim_ / 2;

    // Apply RoPE to Q
    for (int s = 0; s < seq_len; s++) {
        int pos = start_pos + s;
        for (int h = 0; h < num_heads_q; h++) {
            float* q_ptr = q_data + s * num_heads_q * head_dim_ + h * head_dim_;
            for (int i = 0; i < half_dim; i++) {
                float cos_val = cos_[pos * half_dim + i];
                float sin_val = sin_[pos * half_dim + i];

                float q0 = q_ptr[2 * i];
                float q1 = q_ptr[2 * i + 1];

                q_ptr[2 * i]     = q0 * cos_val - q1 * sin_val;
                q_ptr[2 * i + 1] = q0 * sin_val + q1 * cos_val;
            }
        }
    }

    // Apply RoPE to K
    for (int s = 0; s < seq_len; s++) {
        int pos = start_pos + s;
        for (int h = 0; h < num_heads_k; h++) {
            float* k_ptr = k_data + s * num_heads_k * head_dim_ + h * head_dim_;
            for (int i = 0; i < half_dim; i++) {
                float cos_val = cos_[pos * half_dim + i];
                float sin_val = sin_[pos * half_dim + i];

                float k0 = k_ptr[2 * i];
                float k1 = k_ptr[2 * i + 1];

                k_ptr[2 * i]     = k0 * cos_val - k1 * sin_val;
                k_ptr[2 * i + 1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }

    backend->unmap_buffer(q.buffer());
    backend->unmap_buffer(k.buffer());

    return {};
}

}  // namespace granite
