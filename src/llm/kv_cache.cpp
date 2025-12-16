#include "llm_internal.h"

namespace granite {

Result<KVCache> KVCache::allocate(
    const ModelConfig& config,
    int max_seq_len,
    IComputeBackend* backend)
{
    KVCache cache;
    cache.max_seq_len_ = max_seq_len;
    cache.num_kv_heads_ = config.num_kv_heads;
    cache.head_dim_ = config.head_dim;
    cache.backend_ = backend;
    cache.current_len_ = 0;

    // Allocate per-layer cache
    cache.layers_.reserve(config.num_layers);

    std::vector<int64_t> cache_shape = {
        1,
        static_cast<int64_t>(config.num_kv_heads),
        static_cast<int64_t>(max_seq_len),
        static_cast<int64_t>(config.head_dim)
    };

    for (int i = 0; i < config.num_layers; i++) {
        LayerCache layer;

        // Keys: [1, num_kv_heads, max_seq_len, head_dim]
        auto keys_result = Tensor::allocate(cache_shape, DataType::FP16, backend);
        if (!keys_result.ok()) {
            return keys_result.error();
        }
        layer.keys = std::move(keys_result).take();

        // Values: [1, num_kv_heads, max_seq_len, head_dim]
        auto values_result = Tensor::allocate(cache_shape, DataType::FP16, backend);
        if (!values_result.ok()) {
            return values_result.error();
        }
        layer.values = std::move(values_result).take();

        cache.layers_.push_back(std::move(layer));
    }

    size_t total_bytes = cache.memory_bytes();
    GRANITE_LOG_INFO("Allocated KV cache: {} layers, {} max seq len, {:.1f} MB",
                     config.num_layers, max_seq_len,
                     static_cast<float>(total_bytes) / (1024 * 1024));

    return cache;
}

Result<void> KVCache::append(int layer, const Tensor& keys, const Tensor& values) {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) {
        GRANITE_FAIL(ErrorCode::InvalidArgument, "Invalid layer index");
    }

    int new_seq_len = static_cast<int>(keys.size(2));
    if (current_len_ + new_seq_len > max_seq_len_) {
        GRANITE_FAIL(ErrorCode::InvalidArgument,
                     fmt::format("KV cache overflow: {} + {} > {}",
                                 current_len_, new_seq_len, max_seq_len_));
    }

    auto& lc = layers_[layer];

    // Copy new keys/values to cache at current position
    auto map_k_src = backend_->map_buffer(keys.buffer());
    auto map_v_src = backend_->map_buffer(values.buffer());
    auto map_k_dst = backend_->map_buffer(lc.keys.buffer());
    auto map_v_dst = backend_->map_buffer(lc.values.buffer());

    if (!map_k_src.ok() || !map_v_src.ok() || !map_k_dst.ok() || !map_v_dst.ok()) {
        return Error(ErrorCode::InternalError, "Failed to map buffers");
    }

    const auto* k_src = static_cast<const uint16_t*>(map_k_src.value());
    const auto* v_src = static_cast<const uint16_t*>(map_v_src.value());
    auto* k_dst = static_cast<uint16_t*>(map_k_dst.value());
    auto* v_dst = static_cast<uint16_t*>(map_v_dst.value());

    // Copy to the right position in cache
    // Shape: [1, num_kv_heads, seq_len, head_dim]
    size_t stride = static_cast<size_t>(max_seq_len_) * head_dim_;
    for (int h = 0; h < num_kv_heads_; h++) {
        size_t dst_offset = h * stride + current_len_ * head_dim_;
        size_t src_offset = h * new_seq_len * head_dim_;
        std::memcpy(k_dst + dst_offset, k_src + src_offset,
                    new_seq_len * head_dim_ * sizeof(uint16_t));
        std::memcpy(v_dst + dst_offset, v_src + src_offset,
                    new_seq_len * head_dim_ * sizeof(uint16_t));
    }

    backend_->unmap_buffer(keys.buffer());
    backend_->unmap_buffer(values.buffer());
    backend_->unmap_buffer(lc.keys.buffer());
    backend_->unmap_buffer(lc.values.buffer());

    // Only update current_len_ on the last layer to keep consistency
    if (layer == static_cast<int>(layers_.size()) - 1) {
        current_len_ += new_seq_len;
    }

    return {};
}

std::pair<Tensor, Tensor> KVCache::get(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) {
        return {Tensor{}, Tensor{}};
    }

    return {layers_[layer].keys, layers_[layer].values};
}

void KVCache::clear() {
    current_len_ = 0;
}

size_t KVCache::memory_bytes() const {
    size_t total = 0;
    for (const auto& lc : layers_) {
        total += lc.keys.size_bytes() + lc.values.size_bytes();
    }
    return total;
}

}  // namespace granite
