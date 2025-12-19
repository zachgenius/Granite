// TransformerModel - Core LLM inference implementation
//
// File organization:
// 1. Model Loading & Initialization (lines ~15-120)
// 2. Embedding Lookup (lines ~125-205)
// 3. Forward Pass (lines ~210-415)
// 4. Helper Functions: apply_rms_norm, add_tensors (lines ~420-530)
// 5. Transformer Block - main layer processing (lines ~535-775)
// 6. CPU Attention (lines ~780-1100)
// 7. CPU Feed-Forward (lines ~1105-1180)
// 8. GPU/Metal Implementations (lines ~1185-1800):
//    - feed_forward_gpu
//    - allocate_gpu_kv_cache
//    - init_decode_pool
//    - sync_cpu_to_gpu_kv_cache
//    - attention_full_gpu
//    - attention_gpu

#include "llm_internal.h"
#include "math_ops.h"

namespace granite {

// =============================================================================
// SECTION 1: Model Loading & Initialization
// =============================================================================

Result<TransformerModel> TransformerModel::load(
    const std::string& path,
    IComputeBackend* backend)
{
    // Load with default balanced config
    return load(path, backend, Config::Balanced());
}

Result<TransformerModel> TransformerModel::load(
    const std::string& path,
    IComputeBackend* backend,
    const Config& config)
{
    TransformerModel model;
    model.backend_ = backend;
    model.runtime_config_ = config;

    // Open GGUF file
    auto gguf_result = GGUFFile::open(path);
    if (!gguf_result.ok()) {
        return gguf_result.error();
    }
    model.gguf_ = std::make_unique<GGUFFile>(std::move(gguf_result).take());

    // Parse config
    auto config_result = parse_model_config(*model.gguf_);
    if (!config_result.ok()) {
        return config_result.error();
    }
    model.model_config_ = std::move(config_result).take();

    // Initialize RoPE cache
    model.rope_cache_.initialize(
        model.model_config_.max_seq_len,
        model.model_config_.head_dim,
        model.model_config_.rope_theta);

    // Load weights (dequantized for CPU path)
    ModelLoader loader(backend);
    auto weights_result = loader.load_weights(*model.gguf_);
    if (!weights_result.ok()) {
        return weights_result.error();
    }
    model.weights_ = std::move(weights_result).take();

    // Also load raw quantized weights for GPU path
    // Only load quantized weights that are used in matmul operations
    for (const auto& info : model.gguf_->tensors()) {
        // Only keep raw weights for quantized types used in projections
        if (info.type != GGMLType::Q4_K && info.type != GGMLType::Q6_K &&
            info.type != GGMLType::Q5_K && info.type != GGMLType::Q3_K &&
            info.type != GGMLType::Q2_K &&
            info.type != GGMLType::Q8_0 && info.type != GGMLType::Q4_0 &&
            info.type != GGMLType::IQ4_NL && info.type != GGMLType::IQ4_XS &&
            info.type != GGMLType::IQ3_S) {
            continue;
        }

        // Skip non-projection weights (embeddings, norms)
        if (info.name.find(".weight") == std::string::npos ||
            info.name.find("_norm") != std::string::npos ||
            info.name.find("embd") != std::string::npos) {
            continue;
        }

        // Create buffer for raw quantized data
        BufferDesc desc;
        desc.size = info.size_bytes();
        desc.memory_type = MemoryType::Shared;

        auto buf_result = backend->create_buffer(desc);
        if (!buf_result.ok()) {
            GRANITE_LOG_WARN("Failed to create raw weight buffer for {}", info.name);
            continue;
        }

        // Copy raw data to buffer
        auto write_result = backend->write_buffer(
            buf_result.value(),
            model.gguf_->tensor_data(info),
            info.size_bytes());

        if (!write_result.ok()) {
            backend->destroy_buffer(buf_result.value());
            continue;
        }

        // Store raw weight info
        RawWeight raw;
        raw.buffer = buf_result.value();
        raw.quant_type = info.type;
        raw.shape = std::vector<int64_t>(info.dimensions.rbegin(), info.dimensions.rend());
        raw.size_bytes = info.size_bytes();

        model.raw_weights_[info.name] = raw;
    }

    GRANITE_LOG_INFO("Loaded model: {} weights ({} raw for GPU)",
                     model.weights_.size(), model.raw_weights_.size());

    // Determine attention backend based on runtime config and device
    auto device_info = platform::get_device_info();
    auto selected_backend = platform::select_attention_backend(config, device_info);

    GRANITE_LOG_INFO("Attention backend: {} (requested: {})",
                     to_string(selected_backend),
                     to_string(config.attention_backend));

    // Enable GPU based on selected backend
#ifdef GRANITE_HAS_METAL
    if ((selected_backend == AttentionBackend::MetalFlash ||
         selected_backend == AttentionBackend::MetalLegacy) &&
        backend->get_type() == BackendType::Metal &&
        !model.raw_weights_.empty()) {
        model.use_gpu_ = true;
        model.use_coreml_ = false;
        GRANITE_LOG_INFO("GPU acceleration enabled (Metal)");
    }
#ifdef GRANITE_HAS_COREML
    else if (selected_backend == AttentionBackend::CoreML &&
             backend->get_type() == BackendType::Metal) {
        // Initialize CoreML attention backend using MetalCompute's device
        auto* coreml = get_coreml_attention();
        auto* gpu = get_metal_compute();
        if (gpu && gpu->is_initialized() && coreml) {
            auto init_result = coreml->initialize(gpu->device());
            if (init_result.ok()) {
                model.use_coreml_ = true;
                model.use_gpu_ = true;  // Still need GPU for other ops
                GRANITE_LOG_INFO("CoreML/ANE attention enabled (ANE available: {})",
                                CoreMLAttention::is_ane_available());
            } else {
                GRANITE_LOG_WARN("CoreML init failed, falling back to Metal: {}",
                                init_result.error().message());
                model.use_gpu_ = true;
                model.use_coreml_ = false;
            }
        } else {
            GRANITE_LOG_WARN("CoreML not available, falling back to Metal");
            model.use_gpu_ = true;
            model.use_coreml_ = false;
        }
    }
#endif
    else if (selected_backend == AttentionBackend::CPU) {
        model.use_gpu_ = false;
        model.use_coreml_ = false;
        GRANITE_LOG_INFO("Using CPU attention backend");
    }
#else
    model.use_gpu_ = false;
    model.use_coreml_ = false;
#endif

    return model;
}

const Tensor* TransformerModel::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        return nullptr;
    }
    return &it->second;
}

const RawWeight* TransformerModel::get_raw_weight(const std::string& name) const {
    auto it = raw_weights_.find(name);
    if (it == raw_weights_.end()) {
        return nullptr;
    }
    return &it->second;
}

// =============================================================================
// SECTION 2: Embedding Lookup
// =============================================================================

Result<Tensor> TransformerModel::embed(const Tensor& token_ids) {
    const Tensor* emb_weight = get_weight("token_embd.weight");
    if (!emb_weight) {
        GRANITE_FAIL(ErrorCode::InvalidState, "Embedding weight not found");
    }

    // token_ids: [batch, seq_len] INT32
    // emb_weight: [vocab_size, hidden_dim]
    // output: [batch, seq_len, hidden_dim]

    int batch = static_cast<int>(token_ids.size(0));
    int seq_len = static_cast<int>(token_ids.size(1));
    int hidden_dim = model_config_.hidden_dim;
    int total_tokens = batch * seq_len;

    std::vector<int64_t> output_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(hidden_dim)
    };
    auto output_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
    if (!output_result.ok()) {
        return output_result.error();
    }
    auto output = std::move(output_result).take();

#ifdef GRANITE_HAS_METAL
    // GPU path for both decode and prefill
    if (use_gpu_) {
        auto* gpu = get_metal_compute();
        if (gpu && gpu->is_initialized()) {
            auto* ids_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(token_ids.buffer()));
            auto* emb_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(emb_weight->buffer()));
            auto* out_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(output.buffer()));

            if (ids_buf && emb_buf && out_buf) {
                gpu->embedding_lookup(ids_buf, emb_buf, out_buf,
                                     total_tokens, hidden_dim, model_config_.vocab_size);
                // No sync needed - will be synced by subsequent operations
                return output;
            }
        }
    }
#endif

    // CPU path
    auto map_ids = backend_->map_buffer(token_ids.buffer());
    auto map_emb = backend_->map_buffer(emb_weight->buffer());
    auto map_out = backend_->map_buffer(output.buffer());

    if (!map_ids.ok() || !map_emb.ok() || !map_out.ok()) {
        return Error(ErrorCode::InternalError, "Failed to map buffers");
    }

    const auto* ids = static_cast<const int32_t*>(map_ids.value());
    const auto* emb = static_cast<const uint16_t*>(map_emb.value());  // FP16
    auto* out = static_cast<float*>(map_out.value());

    // Embedding lookup
    // GGUF stores embedding with ne0=hidden_dim (innermost), ne1=vocab_size
    // This means token embeddings are contiguous: token t's embedding at t * hidden_dim
    int vocab_size = model_config_.vocab_size;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            int token_id = ids[b * seq_len + s];
            if (token_id < 0 || token_id >= vocab_size) {
                token_id = 0;  // Fallback to first token
            }

            const uint16_t* emb_row = emb + token_id * hidden_dim;
            float* out_row = out + (b * seq_len + s) * hidden_dim;

            for (int d = 0; d < hidden_dim; d++) {
                out_row[d] = fp16_to_fp32(emb_row[d]);
            }
        }
    }

    backend_->unmap_buffer(token_ids.buffer());
    backend_->unmap_buffer(emb_weight->buffer());
    backend_->unmap_buffer(output.buffer());

    return output;
}

// =============================================================================
// SECTION 3: Forward Pass
// =============================================================================

// Helper to print tensor stats for debugging
static void debug_tensor_stats(const std::string& name, const Tensor& t, IComputeBackend* backend) {
    auto map_result = backend->map_buffer(t.buffer());
    if (!map_result.ok()) return;

    const float* data = static_cast<const float*>(map_result.value());
    size_t count = 1;
    for (size_t i = 0; i < t.shape().size(); i++) count *= t.shape()[i];

    float min_val = data[0], max_val = data[0], sum = 0;
    for (size_t i = 0; i < count; i++) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
        sum += data[i];
    }
    float mean = sum / count;

    GRANITE_LOG_INFO("DEBUG {}: min={:.4f}, max={:.4f}, mean={:.4f}, first5=[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]",
                     name, min_val, max_val, mean, data[0], data[1], data[2], data[3], data[4]);

    backend->unmap_buffer(t.buffer());
}

Result<Tensor> TransformerModel::forward(
    const Tensor& token_ids,
    KVCache* kv_cache,
    int start_pos)
{
    // Embed tokens
    auto hidden_result = embed(token_ids);
    if (!hidden_result.ok()) {
        return hidden_result.error();
    }
    auto hidden = std::move(hidden_result).take();

    // Process through transformer layers
    for (int layer = 0; layer < model_config_.num_layers; layer++) {
        auto block_result = transformer_block(hidden, layer, kv_cache, start_pos);
        if (!block_result.ok()) {
            return block_result.error();
        }
        hidden = std::move(block_result).take();
    }

    // Final RMSNorm and output projection
    const Tensor* norm_weight = get_weight("output_norm.weight");
    const Tensor* output_weight = get_weight("output.weight");
    if (!output_weight) {
        // Try tied embeddings
        output_weight = get_weight("token_embd.weight");
    }
    if (!output_weight) {
        GRANITE_FAIL(ErrorCode::InvalidState, "Output weight not found");
    }

    int batch = static_cast<int>(hidden.size(0));
    int seq_len = static_cast<int>(hidden.size(1));
    int total_tokens = batch * seq_len;
    bool is_decode = (total_tokens == 1);

#ifdef GRANITE_HAS_METAL
    // GPU path for decode mode
    if (use_gpu_ && is_decode && norm_weight) {
        auto* gpu = get_metal_compute();
        if (gpu && gpu->is_initialized()) {
            auto* h_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(hidden.buffer()));
            auto* norm_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(norm_weight->buffer()));
            auto* out_w_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(output_weight->buffer()));

            if (h_buf && norm_buf && out_w_buf) {
                // Allocate output tensor for normalized hidden
                std::vector<int64_t> norm_shape = {1, 1, static_cast<int64_t>(model_config_.hidden_dim)};
                auto norm_out_result = Tensor::allocate(norm_shape, DataType::FP32, backend_);

                // Allocate logits tensor
                std::vector<int64_t> logits_shape = {1, 1, static_cast<int64_t>(model_config_.vocab_size)};
                auto logits_result = Tensor::allocate(logits_shape, DataType::FP32, backend_);

                if (norm_out_result.ok() && logits_result.ok()) {
                    auto norm_out = std::move(norm_out_result).take();
                    auto logits = std::move(logits_result).take();

                    auto* norm_out_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(norm_out.buffer()));
                    auto* logits_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(logits.buffer()));

                    // GPU Final RMSNorm
                    bool is_f16 = (norm_weight->dtype() == DataType::FP16);
                    if (is_f16) {
                        gpu->rms_norm_f16(h_buf, norm_buf, norm_out_buf,
                                         model_config_.hidden_dim, model_config_.rms_norm_eps);
                    } else {
                        gpu->rms_norm(h_buf, norm_buf, norm_out_buf,
                                     model_config_.hidden_dim, model_config_.rms_norm_eps);
                    }

                    // GPU Output projection (FP16 weights -> matvec_f16)
                    gpu->matvec_f16(norm_out_buf, out_w_buf, logits_buf,
                                   model_config_.hidden_dim, model_config_.vocab_size);

                    // Sync before returning results
                    gpu->sync();

                    return logits;
                }
            }
        }
    }

    // GPU prefill path: complete GPU pipeline including final norm + output projection
    if (use_gpu_ && !is_decode && norm_weight) {
        auto* gpu = get_metal_compute();
        if (gpu && gpu->is_initialized()) {
            auto* h_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(hidden.buffer()));
            auto* norm_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(norm_weight->buffer()));
            auto* out_w_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(output_weight->buffer()));

            if (h_buf && norm_buf && out_w_buf) {
                // Allocate normalized hidden tensor
                std::vector<int64_t> norm_shape = {batch, seq_len, static_cast<int64_t>(model_config_.hidden_dim)};
                auto norm_out_result = Tensor::allocate(norm_shape, DataType::FP32, backend_);

                // Allocate logits tensor
                std::vector<int64_t> logits_shape = {batch, seq_len, static_cast<int64_t>(model_config_.vocab_size)};
                auto logits_result = Tensor::allocate(logits_shape, DataType::FP32, backend_);

                if (norm_out_result.ok() && logits_result.ok()) {
                    auto norm_out = std::move(norm_out_result).take();
                    auto logits = std::move(logits_result).take();

                    auto* norm_out_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(norm_out.buffer()));
                    auto* logits_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(logits.buffer()));

                    // GPU Final Batched RMSNorm
                    bool is_f16 = (norm_weight->dtype() == DataType::FP16);
                    if (is_f16) {
                        gpu->rms_norm_batch_f16(h_buf, norm_buf, norm_out_buf,
                                               total_tokens, model_config_.hidden_dim, model_config_.rms_norm_eps);
                    } else {
                        gpu->rms_norm_batch(h_buf, norm_buf, norm_out_buf,
                                           total_tokens, model_config_.hidden_dim, model_config_.rms_norm_eps);
                    }

                    // GPU Output projection using batched FP16 matmul
                    // output_weight is [vocab_size, hidden_dim] in FP16
                    gpu->matmul_f16(norm_out_buf, out_w_buf, logits_buf,
                                   total_tokens, model_config_.hidden_dim, model_config_.vocab_size);

                    // Sync before returning results
                    gpu->sync();

                    return logits;
                }
            }
        }
    }
#endif

    // CPU path
    if (norm_weight) {
        // Apply RMSNorm (simplified inline implementation)
        auto map_h = backend_->map_buffer(hidden.buffer());
        auto map_w = backend_->map_buffer(norm_weight->buffer());

        if (map_h.ok() && map_w.ok()) {
            auto* h = static_cast<float*>(map_h.value());
            const void* w = map_w.value();
            DataType w_dtype = norm_weight->dtype();

            int dim = model_config_.hidden_dim;

            for (int b = 0; b < batch; b++) {
                for (int s = 0; s < seq_len; s++) {
                    float* row = h + (b * seq_len + s) * dim;

                    // Compute RMS
                    float sum_sq = 0;
                    for (int d = 0; d < dim; d++) {
                        sum_sq += row[d] * row[d];
                    }
                    float rms = std::sqrt(sum_sq / dim + model_config_.rms_norm_eps);
                    float inv_rms = 1.0f / rms;

                    // Normalize and scale (handle FP32 or FP16 weights)
                    for (int d = 0; d < dim; d++) {
                        float weight_val = (w_dtype == DataType::FP32)
                            ? static_cast<const float*>(w)[d]
                            : fp16_to_fp32(static_cast<const uint16_t*>(w)[d]);
                        row[d] = row[d] * inv_rms * weight_val;
                    }
                }
            }

            backend_->unmap_buffer(hidden.buffer());
            backend_->unmap_buffer(norm_weight->buffer());
        }
    }

    std::vector<int64_t> logits_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(model_config_.vocab_size)
    };
    auto logits_result = Tensor::allocate(logits_shape, DataType::FP32, backend_);
    if (!logits_result.ok()) {
        return logits_result.error();
    }
    auto logits = std::move(logits_result).take();

    // Simple matmul implementation
    auto map_h = backend_->map_buffer(hidden.buffer());
    auto map_w = backend_->map_buffer(output_weight->buffer());
    auto map_l = backend_->map_buffer(logits.buffer());

    if (!map_h.ok() || !map_w.ok() || !map_l.ok()) {
        return Error(ErrorCode::InternalError, "Failed to map buffers");
    }

    const auto* h = static_cast<const float*>(map_h.value());
    const auto* w = static_cast<const uint16_t*>(map_w.value());  // [vocab, hidden]
    auto* l = static_cast<float*>(map_l.value());

    int hidden_dim = model_config_.hidden_dim;
    int vocab_size = model_config_.vocab_size;

    // Use optimized BLAS matmul: logits = hidden @ output_weight.T
    // Output weight shape: [vocab_size, hidden_dim]
    matmul_transb_fp16(h, w, l, total_tokens, vocab_size, hidden_dim);

    backend_->unmap_buffer(hidden.buffer());
    backend_->unmap_buffer(output_weight->buffer());
    backend_->unmap_buffer(logits.buffer());

    return logits;
}

Result<Tensor> TransformerModel::forward_single(int32_t token_id, KVCache& kv_cache) {
    // Create single-token tensor
    std::vector<int64_t> ids_shape = {1, 1};
    auto ids_result = Tensor::allocate(ids_shape, DataType::INT32, backend_);
    if (!ids_result.ok()) {
        return ids_result.error();
    }
    auto ids = std::move(ids_result).take();

    auto map_ids = backend_->map_buffer(ids.buffer());
    if (map_ids.ok()) {
        auto* ptr = static_cast<int32_t*>(map_ids.value());
        ptr[0] = token_id;
        backend_->unmap_buffer(ids.buffer());
    }

    int start_pos = kv_cache.seq_len();
    return forward(ids, &kv_cache, start_pos);
}

Result<Tensor> TransformerModel::forward_batch(
    const std::vector<int32_t>& tokens,
    KVCache* kv_cache,
    int start_pos)
{
    if (tokens.empty()) {
        GRANITE_FAIL(ErrorCode::InvalidArgument, "Empty token batch");
    }

    int num_tokens = static_cast<int>(tokens.size());

    // Create tensor from tokens [1, num_tokens]
    std::vector<int64_t> ids_shape = {1, static_cast<int64_t>(num_tokens)};
    auto ids_result = Tensor::allocate(ids_shape, DataType::INT32, backend_);
    if (!ids_result.ok()) {
        return ids_result.error();
    }
    auto ids = std::move(ids_result).take();

    // Copy tokens to tensor
    auto map_ids = backend_->map_buffer(ids.buffer());
    if (!map_ids.ok()) {
        return Error(ErrorCode::InternalError, "Failed to map token buffer");
    }
    auto* ptr = static_cast<int32_t*>(map_ids.value());
    std::memcpy(ptr, tokens.data(), tokens.size() * sizeof(int32_t));
    backend_->unmap_buffer(ids.buffer());

    // Forward pass - returns logits [1, num_tokens, vocab_size]
    // The forward() method already handles proper causal attention for multiple tokens
    return forward(ids, kv_cache, start_pos);
}

Result<Tensor> TransformerModel::forward_tree(
    const std::vector<int32_t>& tokens,
    const std::vector<int>& parent_indices,
    KVCache* kv_cache,
    int start_pos)
{
    if (tokens.empty()) {
        GRANITE_FAIL(ErrorCode::InvalidArgument, "Empty token batch");
    }

    if (tokens.size() != parent_indices.size()) {
        GRANITE_FAIL(ErrorCode::InvalidArgument,
                     "Tokens and parent_indices must have same size");
    }

    int num_nodes = static_cast<int>(tokens.size());

    // Check if GPU path is available
    bool use_gpu = false;
#ifdef GRANITE_HAS_METAL
    auto* mc = get_metal_compute();
    use_gpu = mc && mc->is_initialized();
#endif

    // Build tree_mask only if using CPU path
    std::vector<std::vector<bool>> tree_mask;
    if (!use_gpu) {
        // Build ancestor mask: ancestors[i] = set of node indices that node i can attend to
        // Node i can attend to itself and all ancestors up to root
        tree_mask.resize(num_nodes, std::vector<bool>(num_nodes, false));

        for (int i = 0; i < num_nodes; i++) {
            // Each node attends to itself
            tree_mask[i][i] = true;

            // Walk up the parent chain to find all ancestors
            int current = parent_indices[i];
            while (current >= 0 && current < num_nodes) {
                tree_mask[i][current] = true;
                current = parent_indices[current];
            }
        }
    }

    // Create tensor from tokens [1, num_nodes]
    std::vector<int64_t> ids_shape = {1, static_cast<int64_t>(num_nodes)};
    auto ids_result = Tensor::allocate(ids_shape, DataType::INT32, backend_);
    if (!ids_result.ok()) {
        return ids_result.error();
    }
    auto ids = std::move(ids_result).take();

    // Copy tokens to tensor
    auto map_ids = backend_->map_buffer(ids.buffer());
    if (!map_ids.ok()) {
        return Error(ErrorCode::InternalError, "Failed to map token buffer");
    }
    auto* ptr = static_cast<int32_t*>(map_ids.value());
    std::memcpy(ptr, tokens.data(), tokens.size() * sizeof(int32_t));
    backend_->unmap_buffer(ids.buffer());

    // Embed tokens
    auto hidden_result = embed(ids);
    if (!hidden_result.ok()) {
        return hidden_result.error();
    }
    auto hidden = std::move(hidden_result).take();

    // Process through transformer layers with tree attention
    for (int layer = 0; layer < model_config_.num_layers; layer++) {
        if (use_gpu) {
            // GPU path: use parent_indices directly
            auto block_result = transformer_block_tree_gpu(hidden, layer, kv_cache, start_pos, parent_indices);
            if (!block_result.ok()) {
                return block_result.error();
            }
            hidden = std::move(block_result).take();
        } else {
            // CPU path: use precomputed tree_mask
            auto block_result = transformer_block_tree(hidden, layer, kv_cache, start_pos, tree_mask);
            if (!block_result.ok()) {
                return block_result.error();
            }
            hidden = std::move(block_result).take();
        }
    }

    // Final RMSNorm and output projection (same as regular forward)
    const Tensor* norm_weight = get_weight("output_norm.weight");
    const Tensor* output_weight = get_weight("output.weight");
    if (!output_weight) {
        output_weight = get_weight("token_embd.weight");
    }
    if (!output_weight) {
        GRANITE_FAIL(ErrorCode::InvalidState, "Output weight not found");
    }

    // CPU path for tree forward (multi-token)
    if (norm_weight) {
        hidden = apply_rms_norm(hidden, norm_weight);
    }

    // Output projection: logits = hidden @ output.T
    int batch = 1;
    int seq_len = num_nodes;
    int hidden_dim = model_config_.hidden_dim;
    int vocab_size = model_config_.vocab_size;

    std::vector<int64_t> logits_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(vocab_size)
    };
    auto logits_result = Tensor::allocate(logits_shape, DataType::FP32, backend_);
    if (!logits_result.ok()) {
        return logits_result.error();
    }
    auto logits = std::move(logits_result).take();

    auto map_h = backend_->map_buffer(hidden.buffer());
    auto map_l = backend_->map_buffer(logits.buffer());
    auto map_w = backend_->map_buffer(output_weight->buffer());

    if (!map_h.ok() || !map_l.ok() || !map_w.ok()) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to map logits buffers");
    }

    const float* h_data = static_cast<const float*>(map_h.value());
    float* l_data = static_cast<float*>(map_l.value());
    const uint16_t* w_data = static_cast<const uint16_t*>(map_w.value());

    // Batch matmul: [num_nodes, hidden_dim] @ [vocab_size, hidden_dim].T
    matmul_transb_fp16(h_data, w_data, l_data, num_nodes, vocab_size, hidden_dim);

    backend_->unmap_buffer(hidden.buffer());
    backend_->unmap_buffer(logits.buffer());
    backend_->unmap_buffer(output_weight->buffer());

    return logits;
}

// =============================================================================
// SECTION 4: Helper Functions
// =============================================================================

// Apply RMSNorm in place
Tensor TransformerModel::apply_rms_norm(const Tensor& input, const Tensor* weight) {
    int batch = static_cast<int>(input.size(0));
    int seq_len = static_cast<int>(input.size(1));
    int dim = static_cast<int>(input.size(2));

    // Allocate output
    std::vector<int64_t> shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(dim)
    };
    auto output_result = Tensor::allocate(shape, DataType::FP32, backend_);
    if (!output_result.ok()) {
        return input;  // Return input unchanged on failure
    }
    auto output = std::move(output_result).take();

    auto map_in = backend_->map_buffer(input.buffer());
    auto map_out = backend_->map_buffer(output.buffer());

    if (!map_in.ok() || !map_out.ok()) {
        return input;
    }

    const float* in_data = static_cast<const float*>(map_in.value());
    float* out_data = static_cast<float*>(map_out.value());

    const void* w_data = nullptr;
    DataType w_dtype = DataType::FP16;
    if (weight) {
        auto map_w = backend_->map_buffer(weight->buffer());
        if (map_w.ok()) {
            w_data = map_w.value();
            w_dtype = weight->dtype();
        }
    }

    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            const float* row = in_data + (b * seq_len + s) * dim;
            float* out_row = out_data + (b * seq_len + s) * dim;

            // Compute RMS
            float sum_sq = 0;
            for (int d = 0; d < dim; d++) {
                sum_sq += row[d] * row[d];
            }
            float rms = std::sqrt(sum_sq / dim + model_config_.rms_norm_eps);
            float inv_rms = 1.0f / rms;

            // Normalize and scale
            for (int d = 0; d < dim; d++) {
                float val = row[d] * inv_rms;
                if (w_data) {
                    // Handle both FP16 and FP32 weights
                    if (w_dtype == DataType::FP32) {
                        val *= static_cast<const float*>(w_data)[d];
                    } else {
                        val *= fp16_to_fp32(static_cast<const uint16_t*>(w_data)[d]);
                    }
                }
                out_row[d] = val;
            }
        }
    }

    backend_->unmap_buffer(input.buffer());
    backend_->unmap_buffer(output.buffer());
    if (weight) {
        backend_->unmap_buffer(weight->buffer());
    }

    return output;
}

// Helper: Add two tensors element-wise
Tensor TransformerModel::add_tensors(const Tensor& a, const Tensor& b) {
    int batch = static_cast<int>(a.size(0));
    int seq_len = static_cast<int>(a.size(1));
    int dim = static_cast<int>(a.size(2));

    std::vector<int64_t> shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(dim)
    };
    auto output_result = Tensor::allocate(shape, DataType::FP32, backend_);
    if (!output_result.ok()) {
        return a;
    }
    auto output = std::move(output_result).take();

    auto map_a = backend_->map_buffer(a.buffer());
    auto map_b = backend_->map_buffer(b.buffer());
    auto map_out = backend_->map_buffer(output.buffer());

    if (!map_a.ok() || !map_b.ok() || !map_out.ok()) {
        return a;
    }

    const float* a_data = static_cast<const float*>(map_a.value());
    const float* b_data = static_cast<const float*>(map_b.value());
    float* out_data = static_cast<float*>(map_out.value());

    size_t total = static_cast<size_t>(batch) * seq_len * dim;
    for (size_t i = 0; i < total; i++) {
        out_data[i] = a_data[i] + b_data[i];
    }

    backend_->unmap_buffer(a.buffer());
    backend_->unmap_buffer(b.buffer());
    backend_->unmap_buffer(output.buffer());

    return output;
}

// =============================================================================
// SECTION 5: Transformer Block (Layer Processing)
// =============================================================================

Result<Tensor> TransformerModel::transformer_block(
    const Tensor& hidden,
    int layer,
    KVCache* kv_cache,
    int start_pos)
{
    std::string prefix = "blk." + std::to_string(layer) + ".";

    int batch = static_cast<int>(hidden.size(0));
    int seq_len = static_cast<int>(hidden.size(1));
    int hidden_dim = model_config_.hidden_dim;
    int total_tokens = batch * seq_len;
    bool is_decode = (total_tokens == 1);

    // Get weights
    const Tensor* attn_norm_weight = get_weight(prefix + "attn_norm.weight");
    const Tensor* ffn_norm_weight = get_weight(prefix + "ffn_norm.weight");

#ifdef GRANITE_HAS_METAL
    // Full GPU path for decode mode - avoid CPU<->GPU transfers
    if (use_gpu_ && is_decode) {
        auto* gpu = get_metal_compute();
        if (gpu && gpu->is_initialized() && attn_norm_weight && ffn_norm_weight) {
            // Get Metal buffers
            auto* h_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(hidden.buffer()));
            auto* attn_norm_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(attn_norm_weight->buffer()));
            auto* ffn_norm_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(ffn_norm_weight->buffer()));

            if (h_buf && attn_norm_buf && ffn_norm_buf) {
                // Use pooled buffers if available
                MTL::Buffer* attn_in_buf;
                MTL::Buffer* post_attn_buf;
                MTL::Buffer* ffn_in_buf;
                MTL::Buffer* out_buf;

                Tensor attn_input, post_attn, ffn_input, output;
                bool use_pool = decode_pool_ && decode_pool_->initialized;

                if (use_pool) {
                    // Use preallocated buffers
                    attn_in_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(decode_pool_->attn_input.buffer()));
                    post_attn_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(decode_pool_->post_attn.buffer()));
                    ffn_in_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(decode_pool_->ffn_input.buffer()));
                    out_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(decode_pool_->block_output.buffer()));

                    // Reference the tensors (not taking ownership)
                    attn_input = decode_pool_->attn_input;
                    post_attn = decode_pool_->post_attn;
                    ffn_input = decode_pool_->ffn_input;
                    output = decode_pool_->block_output;
                } else {
                    // Fallback to allocation
                    std::vector<int64_t> output_shape = {1, 1, static_cast<int64_t>(hidden_dim)};
                    auto attn_in_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
                    auto post_attn_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
                    auto ffn_in_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
                    auto output_result = Tensor::allocate(output_shape, DataType::FP32, backend_);

                    if (!attn_in_result.ok() || !post_attn_result.ok() ||
                        !ffn_in_result.ok() || !output_result.ok()) {
                        goto cpu_path;  // Fallback to CPU
                    }

                    attn_input = std::move(attn_in_result).take();
                    post_attn = std::move(post_attn_result).take();
                    ffn_input = std::move(ffn_in_result).take();
                    output = std::move(output_result).take();

                    attn_in_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(attn_input.buffer()));
                    post_attn_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(post_attn.buffer()));
                    ffn_in_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(ffn_input.buffer()));
                    out_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(output.buffer()));
                }

                // 1. GPU RMSNorm for attention
                bool is_f16_weight = (attn_norm_weight->dtype() == DataType::FP16);
                if (is_f16_weight) {
                    gpu->rms_norm_f16(h_buf, attn_norm_buf, attn_in_buf,
                                     hidden_dim, model_config_.rms_norm_eps);
                } else {
                    gpu->rms_norm(h_buf, attn_norm_buf, attn_in_buf,
                                 hidden_dim, model_config_.rms_norm_eps);
                }

                // 2. GPU Attention
                auto attn_result = attention_gpu(attn_input, layer, kv_cache, start_pos);
                if (!attn_result.ok()) {
                    return attn_result.error();
                }
                auto attn_output = std::move(attn_result).take();
                auto* attn_out_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(attn_output.buffer()));

                // 3. GPU Residual add: post_attn = hidden + attn_output
                gpu->elementwise_add(h_buf, attn_out_buf, post_attn_buf, hidden_dim);

                // 4+5. FUSED GPU Feed Forward using rms_norm_matvec_q4k
                // This eliminates the intermediate normalized buffer by fusing
                // RMSNorm with the gate/up projections
                const RawWeight* w_gate = get_raw_weight(prefix + "ffn_gate.weight");
                const RawWeight* w_up = get_raw_weight(prefix + "ffn_up.weight");
                const RawWeight* w_down = get_raw_weight(prefix + "ffn_down.weight");

                // Check if we can use fused kernels (need Q4_K/Q6_K/Q5_K/Q3_K/Q2_K/Q8_0/Q4_0/IQ4_NL/IQ4_XS/IQ3_S weights and FP16 norm)
                bool can_fuse = w_gate && w_up && w_down &&
                               (w_gate->quant_type == GGMLType::Q4_K ||
                                w_gate->quant_type == GGMLType::Q6_K ||
                                w_gate->quant_type == GGMLType::Q5_K ||
                                w_gate->quant_type == GGMLType::Q3_K ||
                                w_gate->quant_type == GGMLType::Q2_K ||
                                w_gate->quant_type == GGMLType::Q8_0 ||
                                w_gate->quant_type == GGMLType::Q4_0 ||
                                w_gate->quant_type == GGMLType::IQ4_NL ||
                                w_gate->quant_type == GGMLType::IQ4_XS ||
                                w_gate->quant_type == GGMLType::IQ3_S) &&
                               ffn_norm_weight->dtype() == DataType::FP16;
                GGMLType fused_qtype = can_fuse ? w_gate->quant_type : GGMLType::F32;

                Tensor ffn_output;
                bool residual_fused = false;  // Track if residual was fused into down projection
                if (can_fuse) {
                    // Get weight buffers
                    auto* wg_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(w_gate->buffer));
                    auto* wu_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(w_up->buffer));
                    auto* wd_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(w_down->buffer));

                    // Use pooled FFN buffers if available
                    MTL::Buffer* gate_buf;
                    MTL::Buffer* up_buf;
                    bool use_ffn_pool = decode_pool_ && decode_pool_->initialized &&
                                        decode_pool_->ffn_gate_buf && decode_pool_->ffn_up_buf;

                    if (use_ffn_pool) {
                        gate_buf = static_cast<MTL::Buffer*>(decode_pool_->ffn_gate_buf);
                        up_buf = static_cast<MTL::Buffer*>(decode_pool_->ffn_up_buf);
                    } else {
                        gate_buf = gpu->create_buffer(model_config_.intermediate_dim * sizeof(float));
                        up_buf = gpu->create_buffer(model_config_.intermediate_dim * sizeof(float));
                    }

                    // Check if residual will be fused based on DOWN weight's type (not gate)
                    // Only Q4_K/Q3_K/Q2_K have fused down+residual kernels
                    GGMLType down_qtype = w_down->quant_type;
                    bool will_fuse_residual = (down_qtype == GGMLType::Q4_K ||
                                               down_qtype == GGMLType::Q3_K ||
                                               down_qtype == GGMLType::Q2_K);

                    // Only allocate FFN output for non-fused residual paths
                    MTL::Buffer* ffn_out_buf = nullptr;
                    if (!will_fuse_residual) {
                        std::vector<int64_t> ffn_out_shape = {1, 1, static_cast<int64_t>(hidden_dim)};
                        auto ffn_out_result = Tensor::allocate(ffn_out_shape, DataType::FP32, backend_);
                        if (!ffn_out_result.ok()) {
                            if (!use_ffn_pool) {
                                if (gate_buf) gate_buf->release();
                                if (up_buf) up_buf->release();
                            }
                            return ffn_out_result.error();
                        }
                        ffn_output = std::move(ffn_out_result).take();
                        ffn_out_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(ffn_output.buffer()));
                    }

                    // Fused RMSNorm + gate projection: eliminates intermediate norm buffer
                    if (fused_qtype == GGMLType::Q8_0) {
                        gpu->rms_norm_matvec_q8_0(post_attn_buf, ffn_norm_buf, wg_buf, gate_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                        gpu->rms_norm_matvec_q8_0(post_attn_buf, ffn_norm_buf, wu_buf, up_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                    } else if (fused_qtype == GGMLType::Q4_0) {
                        gpu->rms_norm_matvec_q4_0(post_attn_buf, ffn_norm_buf, wg_buf, gate_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                        gpu->rms_norm_matvec_q4_0(post_attn_buf, ffn_norm_buf, wu_buf, up_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                    } else if (fused_qtype == GGMLType::IQ4_NL) {
                        gpu->rms_norm_matvec_iq4_nl(post_attn_buf, ffn_norm_buf, wg_buf, gate_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                        gpu->rms_norm_matvec_iq4_nl(post_attn_buf, ffn_norm_buf, wu_buf, up_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                    } else if (fused_qtype == GGMLType::IQ4_XS) {
                        gpu->rms_norm_matvec_iq4_xs(post_attn_buf, ffn_norm_buf, wg_buf, gate_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                        gpu->rms_norm_matvec_iq4_xs(post_attn_buf, ffn_norm_buf, wu_buf, up_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                    } else if (fused_qtype == GGMLType::IQ3_S) {
                        gpu->rms_norm_matvec_iq3_s(post_attn_buf, ffn_norm_buf, wg_buf, gate_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                        gpu->rms_norm_matvec_iq3_s(post_attn_buf, ffn_norm_buf, wu_buf, up_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                    } else if (fused_qtype == GGMLType::Q6_K) {
                        gpu->rms_norm_matvec_q6_k(post_attn_buf, ffn_norm_buf, wg_buf, gate_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                        gpu->rms_norm_matvec_q6_k(post_attn_buf, ffn_norm_buf, wu_buf, up_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                    } else if (fused_qtype == GGMLType::Q5_K) {
                        gpu->rms_norm_matvec_q5_k(post_attn_buf, ffn_norm_buf, wg_buf, gate_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                        gpu->rms_norm_matvec_q5_k(post_attn_buf, ffn_norm_buf, wu_buf, up_buf,
                                                hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                    } else if (fused_qtype == GGMLType::Q3_K) {
                        // Q3_K - Use Phase 2 fused kernel (RMSNorm computed once!)
                        gpu->rms_norm_dual_matvec_q3k(post_attn_buf, ffn_norm_buf, wg_buf, wu_buf,
                                                     gate_buf, up_buf,
                                                     hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                    } else if (fused_qtype == GGMLType::Q2_K) {
                        // Q2_K - Use Phase 2 fused kernel (RMSNorm computed once!)
                        gpu->rms_norm_dual_matvec_q2k(post_attn_buf, ffn_norm_buf, wg_buf, wu_buf,
                                                     gate_buf, up_buf,
                                                     hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                    } else {
                        // Q4_K (default) - Use Phase 2 fused kernel (RMSNorm computed once!)
                        gpu->rms_norm_dual_matvec_q4k(post_attn_buf, ffn_norm_buf, wg_buf, wu_buf,
                                                     gate_buf, up_buf,
                                                     hidden_dim, model_config_.intermediate_dim, model_config_.rms_norm_eps);
                    }

                    // Fused silu + mul
                    gpu->silu_mul(gate_buf, up_buf, gate_buf, model_config_.intermediate_dim);

                    // Down projection - use w_down's actual quantization type (down_qtype defined above)
                    if (down_qtype == GGMLType::Q8_0) {
                        gpu->matvec_q8_0(gate_buf, wd_buf, ffn_out_buf, model_config_.intermediate_dim, hidden_dim);
                    } else if (down_qtype == GGMLType::Q4_0) {
                        gpu->matvec_q4_0(gate_buf, wd_buf, ffn_out_buf, model_config_.intermediate_dim, hidden_dim);
                    } else if (down_qtype == GGMLType::IQ4_NL) {
                        gpu->matvec_iq4_nl(gate_buf, wd_buf, ffn_out_buf, model_config_.intermediate_dim, hidden_dim);
                    } else if (down_qtype == GGMLType::IQ4_XS) {
                        gpu->matvec_iq4_xs(gate_buf, wd_buf, ffn_out_buf, model_config_.intermediate_dim, hidden_dim);
                    } else if (down_qtype == GGMLType::IQ3_S) {
                        gpu->matvec_iq3_s(gate_buf, wd_buf, ffn_out_buf, model_config_.intermediate_dim, hidden_dim);
                    } else if (down_qtype == GGMLType::Q6_K) {
                        gpu->matvec_q6_k(gate_buf, wd_buf, ffn_out_buf, model_config_.intermediate_dim, hidden_dim);
                    } else if (down_qtype == GGMLType::Q5_K) {
                        gpu->matvec_q5_k(gate_buf, wd_buf, ffn_out_buf, model_config_.intermediate_dim, hidden_dim);
                    } else if (down_qtype == GGMLType::Q3_K) {
                        // Q3_K - Use Phase 2 fused kernel (down proj + residual in one!)
                        gpu->matvec_residual_q3k(gate_buf, wd_buf, post_attn_buf, out_buf,
                                                model_config_.intermediate_dim, hidden_dim);
                        residual_fused = true;
                    } else if (down_qtype == GGMLType::Q2_K) {
                        // Q2_K - Use Phase 2 fused kernel (down proj + residual in one!)
                        gpu->matvec_residual_q2k(gate_buf, wd_buf, post_attn_buf, out_buf,
                                                model_config_.intermediate_dim, hidden_dim);
                        residual_fused = true;
                    } else if (down_qtype == GGMLType::Q4_K) {
                        // Q4_K - Use Phase 2 fused kernel (down proj + residual in one!)
                        gpu->matvec_residual_q4k(gate_buf, wd_buf, post_attn_buf, out_buf,
                                                model_config_.intermediate_dim, hidden_dim);
                        residual_fused = true;
                    } else {
                        // Fallback to Q4_K for unknown types
                        gpu->matvec_q4k(gate_buf, wd_buf, ffn_out_buf, model_config_.intermediate_dim, hidden_dim);
                    }

                    if (!use_ffn_pool) {
                        gate_buf->release();
                        up_buf->release();
                    }
                } else {
                    // Fallback: separate RMSNorm + feed_forward_gpu
                    is_f16_weight = (ffn_norm_weight->dtype() == DataType::FP16);
                    if (is_f16_weight) {
                        gpu->rms_norm_f16(post_attn_buf, ffn_norm_buf, ffn_in_buf,
                                         hidden_dim, model_config_.rms_norm_eps);
                    } else {
                        gpu->rms_norm(post_attn_buf, ffn_norm_buf, ffn_in_buf,
                                     hidden_dim, model_config_.rms_norm_eps);
                    }

                    auto ffn_result = feed_forward_gpu(ffn_input, layer);
                    if (!ffn_result.ok()) {
                        return ffn_result.error();
                    }
                    ffn_output = std::move(ffn_result).take();
                }

                // 6. GPU Residual add: output = post_attn + ffn_output
                // (Skip for Q4_K - already fused into matvec_residual_q4k)
                if (!residual_fused) {
                    auto* ffn_out_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(ffn_output.buffer()));
                    gpu->elementwise_add(post_attn_buf, ffn_out_buf, out_buf, hidden_dim);
                }

                return output;
            }
        }
    }

    // GPU PREFILL PATH: For multi-token prefill (seq_len > 1)
    if (!is_decode && use_gpu_) {
        auto* gpu = get_metal_compute();
        if (gpu && gpu->is_initialized() && attn_norm_weight && ffn_norm_weight) {
            // Check if weights are FP16 (most common for GGUF models)
            bool attn_norm_f16 = (attn_norm_weight->dtype() == DataType::FP16);
            bool ffn_norm_f16 = (ffn_norm_weight->dtype() == DataType::FP16);

            // GPU prefill supports both FP16 and FP32 norm weights
            // Get Metal buffers for hidden and norm weights
            auto* h_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(hidden.buffer()));
            auto* attn_norm_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(attn_norm_weight->buffer()));
            auto* ffn_norm_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(ffn_norm_weight->buffer()));

            if (h_buf && attn_norm_buf && ffn_norm_buf) {
                // Allocate output tensors
                std::vector<int64_t> output_shape = {batch, seq_len, static_cast<int64_t>(hidden_dim)};
                auto attn_in_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
                auto post_attn_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
                auto ffn_in_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
                auto output_result = Tensor::allocate(output_shape, DataType::FP32, backend_);

                if (attn_in_result.ok() && post_attn_result.ok() &&
                    ffn_in_result.ok() && output_result.ok()) {

                    auto attn_input = std::move(attn_in_result).take();
                    auto post_attn = std::move(post_attn_result).take();
                    auto ffn_input = std::move(ffn_in_result).take();
                    auto output = std::move(output_result).take();

                    auto* attn_in_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(attn_input.buffer()));
                    auto* post_attn_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(post_attn.buffer()));
                    auto* ffn_in_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(ffn_input.buffer()));
                    auto* out_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(output.buffer()));

                    // 1. GPU Batched RMSNorm for attention (FP16 or FP32 weights)
                    if (attn_norm_f16) {
                        gpu->rms_norm_batch_f16(h_buf, attn_norm_buf, attn_in_buf,
                                                total_tokens, hidden_dim, model_config_.rms_norm_eps);
                    } else {
                        gpu->rms_norm_batch(h_buf, attn_norm_buf, attn_in_buf,
                                           total_tokens, hidden_dim, model_config_.rms_norm_eps);
                    }

                    // 2. GPU Attention (attention_gpu already handles prefill)
                    auto attn_result = attention_gpu(attn_input, layer, kv_cache, start_pos);
                    if (!attn_result.ok()) {
                        return attn_result.error();
                    }
                    auto attn_output = std::move(attn_result).take();
                    auto* attn_out_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(attn_output.buffer()));

                    // 3. GPU Residual add: post_attn = hidden + attn_output
                    gpu->elementwise_add(h_buf, attn_out_buf, post_attn_buf, total_tokens * hidden_dim);

                    // 4. GPU Batched RMSNorm for FFN (FP16 or FP32 weights)
                    if (ffn_norm_f16) {
                        gpu->rms_norm_batch_f16(post_attn_buf, ffn_norm_buf, ffn_in_buf,
                                                total_tokens, hidden_dim, model_config_.rms_norm_eps);
                    } else {
                        gpu->rms_norm_batch(post_attn_buf, ffn_norm_buf, ffn_in_buf,
                                           total_tokens, hidden_dim, model_config_.rms_norm_eps);
                    }

                    // 5. GPU FFN (feed_forward_gpu already handles prefill)
                    auto ffn_result = feed_forward_gpu(ffn_input, layer);
                    if (!ffn_result.ok()) {
                        return ffn_result.error();
                    }
                    auto ffn_output = std::move(ffn_result).take();
                    auto* ffn_out_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(ffn_output.buffer()));

                    // 6. GPU Residual add: output = post_attn + ffn_output
                    gpu->elementwise_add(post_attn_buf, ffn_out_buf, out_buf, total_tokens * hidden_dim);

                    return output;
                }
            }
        }
    }

cpu_path:
#endif

    // CPU path (or fallback)
    std::vector<int64_t> hidden_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(hidden_dim)
    };
    auto residual_result = Tensor::allocate(hidden_shape, DataType::FP32, backend_);
    if (!residual_result.ok()) {
        return residual_result.error();
    }
    auto residual = std::move(residual_result).take();

    // Copy hidden to residual
    auto map_h = backend_->map_buffer(hidden.buffer());
    auto map_r = backend_->map_buffer(residual.buffer());
    if (map_h.ok() && map_r.ok()) {
        std::memcpy(map_r.value(), map_h.value(), hidden.size_bytes());
        backend_->unmap_buffer(hidden.buffer());
        backend_->unmap_buffer(residual.buffer());
    }

    // 1. RMSNorm for attention
    auto attn_input = apply_rms_norm(hidden, attn_norm_weight);

    // 2. Attention with KV cache (use GPU if available)
#ifdef GRANITE_HAS_METAL
    auto attn_result = use_gpu_ ? attention_gpu(attn_input, layer, kv_cache, start_pos)
                                : attention(attn_input, layer, kv_cache, start_pos);
#else
    auto attn_result = attention(attn_input, layer, kv_cache, start_pos);
#endif
    if (!attn_result.ok()) {
        return attn_result.error();
    }
    auto attn_output = std::move(attn_result).take();

    // 3. Residual add: hidden = residual + attn_output
    auto post_attn = add_tensors(residual, attn_output);

    // 4. RMSNorm for FFN
    auto ffn_input = apply_rms_norm(post_attn, ffn_norm_weight);

    // 5. Feed forward (SwiGLU) - use GPU if available
#ifdef GRANITE_HAS_METAL
    auto ffn_result = use_gpu_ ? feed_forward_gpu(ffn_input, layer)
                               : feed_forward(ffn_input, layer);
#else
    auto ffn_result = feed_forward(ffn_input, layer);
#endif
    if (!ffn_result.ok()) {
        return ffn_result.error();
    }
    auto ffn_output = std::move(ffn_result).take();

    // 6. Residual add: output = post_attn + ffn_output
    auto output = add_tensors(post_attn, ffn_output);
    return output;
}

// Tree-aware transformer block for speculative decoding
Result<Tensor> TransformerModel::transformer_block_tree(
    const Tensor& hidden,
    int layer,
    KVCache* kv_cache,
    int start_pos,
    const std::vector<std::vector<bool>>& tree_mask)
{
    std::string prefix = "blk." + std::to_string(layer) + ".";

    // Get weights
    const Tensor* attn_norm_weight = get_weight(prefix + "attn_norm.weight");
    const Tensor* ffn_norm_weight = get_weight(prefix + "ffn_norm.weight");

    // CPU path for tree attention (multi-token, tree-structured)
    // Save residual for later
    Tensor residual = hidden;

    // 1. RMSNorm for attention
    auto attn_input = apply_rms_norm(hidden, attn_norm_weight);

    // 2. Tree attention (no GPU path for tree attention yet)
    auto attn_result = attention_tree(attn_input, layer, kv_cache, start_pos, tree_mask);
    if (!attn_result.ok()) {
        return attn_result.error();
    }
    auto attn_output = std::move(attn_result).take();

    // 3. Residual add: hidden = residual + attn_output
    auto post_attn = add_tensors(residual, attn_output);

    // 4. RMSNorm for FFN
    auto ffn_input = apply_rms_norm(post_attn, ffn_norm_weight);

    // 5. Feed forward (SwiGLU) - CPU path
    auto ffn_result = feed_forward(ffn_input, layer);
    if (!ffn_result.ok()) {
        return ffn_result.error();
    }
    auto ffn_output = std::move(ffn_result).take();

    // 6. Residual add: output = post_attn + ffn_output
    auto output = add_tensors(post_attn, ffn_output);
    return output;
}

// =============================================================================
// SECTION 6: CPU Attention
// =============================================================================

Result<Tensor> TransformerModel::attention(
    const Tensor& hidden,
    int layer,
    KVCache* kv_cache,
    int start_pos)
{
    std::string prefix = "blk." + std::to_string(layer) + ".";

    // Get attention weights
    const Tensor* wq = get_weight(prefix + "attn_q.weight");
    const Tensor* wk = get_weight(prefix + "attn_k.weight");
    const Tensor* wv = get_weight(prefix + "attn_v.weight");
    const Tensor* wo = get_weight(prefix + "attn_output.weight");

    if (!wq || !wk || !wv || !wo) {
        GRANITE_FAIL(ErrorCode::InvalidState,
                     "Missing attention weights for layer " + std::to_string(layer));
    }

    int batch = static_cast<int>(hidden.size(0));
    int seq_len = static_cast<int>(hidden.size(1));
    int hidden_dim = model_config_.hidden_dim;
    int num_heads = model_config_.num_heads;
    int num_kv_heads = model_config_.num_kv_heads;
    int head_dim = model_config_.head_dim;
    int kv_dim = num_kv_heads * head_dim;

    // Allocate Q, K, V tensors
    std::vector<int64_t> q_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(num_heads),
        static_cast<int64_t>(head_dim)
    };
    std::vector<int64_t> kv_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(num_kv_heads),
        static_cast<int64_t>(head_dim)
    };

    auto q_result = Tensor::allocate(q_shape, DataType::FP32, backend_);
    auto k_result = Tensor::allocate(kv_shape, DataType::FP32, backend_);
    auto v_result = Tensor::allocate(kv_shape, DataType::FP32, backend_);

    if (!q_result.ok() || !k_result.ok() || !v_result.ok()) {
        GRANITE_FAIL(ErrorCode::AllocationFailed, "Failed to allocate Q/K/V tensors");
    }

    auto q = std::move(q_result).take();
    auto k = std::move(k_result).take();
    auto v = std::move(v_result).take();

    // Project Q, K, V
    // hidden: [batch, seq_len, hidden_dim]
    // wq: [num_heads * head_dim, hidden_dim]
    // output: [batch, seq_len, num_heads, head_dim]
    auto map_h = backend_->map_buffer(hidden.buffer());
    auto map_q = backend_->map_buffer(q.buffer());
    auto map_k = backend_->map_buffer(k.buffer());
    auto map_v = backend_->map_buffer(v.buffer());
    auto map_wq = backend_->map_buffer(wq->buffer());
    auto map_wk = backend_->map_buffer(wk->buffer());
    auto map_wv = backend_->map_buffer(wv->buffer());

    if (!map_h.ok() || !map_q.ok() || !map_k.ok() || !map_v.ok() ||
        !map_wq.ok() || !map_wk.ok() || !map_wv.ok()) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to map attention buffers");
    }

    const float* h_data = static_cast<const float*>(map_h.value());
    float* q_data = static_cast<float*>(map_q.value());
    float* k_data = static_cast<float*>(map_k.value());
    float* v_data = static_cast<float*>(map_v.value());
    const uint16_t* wq_data = static_cast<const uint16_t*>(map_wq.value());
    const uint16_t* wk_data = static_cast<const uint16_t*>(map_wk.value());
    const uint16_t* wv_data = static_cast<const uint16_t*>(map_wv.value());

    // Use optimized BLAS for Q, K, V projections
    // Q projection: [batch*seq, hidden] @ W_q.T -> [batch*seq, q_out_dim]
    int total_tokens = batch * seq_len;
    int q_out_dim = num_heads * head_dim;

    // Q = hidden @ W_q^T
    matmul_transb_fp16(h_data, wq_data, q_data, total_tokens, q_out_dim, hidden_dim);

    // K = hidden @ W_k^T
    matmul_transb_fp16(h_data, wk_data, k_data, total_tokens, kv_dim, hidden_dim);

    // V = hidden @ W_v^T
    matmul_transb_fp16(h_data, wv_data, v_data, total_tokens, kv_dim, hidden_dim);

    backend_->unmap_buffer(hidden.buffer());
    backend_->unmap_buffer(wq->buffer());
    backend_->unmap_buffer(wk->buffer());
    backend_->unmap_buffer(wv->buffer());

    // Unmap Q, K, V before RoPE (RoPE will remap them)
    backend_->unmap_buffer(q.buffer());
    backend_->unmap_buffer(k.buffer());
    backend_->unmap_buffer(v.buffer());

    // Apply RoPE to Q and K
    auto rope_result = rope_cache_.apply(q, k, start_pos, backend_);
    if (!rope_result.ok()) {
        GRANITE_LOG_WARN("RoPE failed: {}", rope_result.error().message());
    }
    // Note: RoPE::apply unmaps Q and K internally

    // Update KV cache if provided
    if (kv_cache) {
        // Convert K, V to FP16 for cache
        std::vector<int64_t> kv_cache_shape = {
            1,
            static_cast<int64_t>(num_kv_heads),
            static_cast<int64_t>(seq_len),
            static_cast<int64_t>(head_dim)
        };
        auto k_fp16_result = Tensor::allocate(kv_cache_shape, DataType::FP16, backend_);
        auto v_fp16_result = Tensor::allocate(kv_cache_shape, DataType::FP16, backend_);

        if (k_fp16_result.ok() && v_fp16_result.ok()) {
            auto k_fp16 = std::move(k_fp16_result).take();
            auto v_fp16 = std::move(v_fp16_result).take();

            // Convert to FP16 (simplified - assumes batch=1)
            auto map_k_src = backend_->map_buffer(k.buffer());
            auto map_v_src = backend_->map_buffer(v.buffer());
            auto map_k_dst = backend_->map_buffer(k_fp16.buffer());
            auto map_v_dst = backend_->map_buffer(v_fp16.buffer());

            if (map_k_src.ok() && map_v_src.ok() && map_k_dst.ok() && map_v_dst.ok()) {
                const float* k_src = static_cast<const float*>(map_k_src.value());
                const float* v_src = static_cast<const float*>(map_v_src.value());
                uint16_t* k_dst = static_cast<uint16_t*>(map_k_dst.value());
                uint16_t* v_dst = static_cast<uint16_t*>(map_v_dst.value());

                // Transpose and convert: [batch, seq, num_kv_heads, head_dim] -> [1, num_kv_heads, seq, head_dim]
                for (int h = 0; h < num_kv_heads; h++) {
                    for (int s = 0; s < seq_len; s++) {
                        for (int d = 0; d < head_dim; d++) {
                            int src_idx = s * num_kv_heads * head_dim + h * head_dim + d;
                            int dst_idx = h * seq_len * head_dim + s * head_dim + d;
                            k_dst[dst_idx] = fp32_to_fp16(k_src[src_idx]);
                            v_dst[dst_idx] = fp32_to_fp16(v_src[src_idx]);
                        }
                    }
                }

                backend_->unmap_buffer(k.buffer());
                backend_->unmap_buffer(v.buffer());
                backend_->unmap_buffer(k_fp16.buffer());
                backend_->unmap_buffer(v_fp16.buffer());

                kv_cache->append(layer, k_fp16, v_fp16);
            }
        }
    }

    // Compute attention scores and output
    // For now, simplified single-sequence attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Total sequence length (including cached)
    int total_seq = kv_cache ? kv_cache->seq_len() : seq_len;

    // Allocate attention output
    std::vector<int64_t> output_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(hidden_dim)
    };
    auto output_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
    if (!output_result.ok()) {
        return output_result.error();
    }
    auto output = std::move(output_result).take();

    // Map output and weight
    auto map_out = backend_->map_buffer(output.buffer());
    auto map_wo = backend_->map_buffer(wo->buffer());
    map_q = backend_->map_buffer(q.buffer());

    if (!map_out.ok() || !map_wo.ok() || !map_q.ok()) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to map output buffers");
    }

    float* out_data = static_cast<float*>(map_out.value());
    const uint16_t* wo_data = static_cast<const uint16_t*>(map_wo.value());
    q_data = static_cast<float*>(map_q.value());

    // Get K, V from cache or use current
    const float* k_all = nullptr;
    const float* v_all = nullptr;
    std::vector<float> k_buffer, v_buffer;

    if (kv_cache && kv_cache->seq_len() > 0) {
        auto [cached_k, cached_v] = kv_cache->get(layer);
        auto map_ck = backend_->map_buffer(cached_k.buffer());
        auto map_cv = backend_->map_buffer(cached_v.buffer());

        if (map_ck.ok() && map_cv.ok()) {
            // Convert cached FP16 to FP32 and transpose
            const uint16_t* ck = static_cast<const uint16_t*>(map_ck.value());
            const uint16_t* cv = static_cast<const uint16_t*>(map_cv.value());

            k_buffer.resize(total_seq * num_kv_heads * head_dim);
            v_buffer.resize(total_seq * num_kv_heads * head_dim);

            // [1, num_kv_heads, total_seq, head_dim] -> [total_seq, num_kv_heads, head_dim]
            for (int h = 0; h < num_kv_heads; h++) {
                for (int s = 0; s < total_seq; s++) {
                    for (int d = 0; d < head_dim; d++) {
                        int src_idx = h * kv_cache->max_seq_len() * head_dim + s * head_dim + d;
                        int dst_idx = s * num_kv_heads * head_dim + h * head_dim + d;
                        k_buffer[dst_idx] = fp16_to_fp32(ck[src_idx]);
                        v_buffer[dst_idx] = fp16_to_fp32(cv[src_idx]);
                    }
                }
            }

            backend_->unmap_buffer(cached_k.buffer());
            backend_->unmap_buffer(cached_v.buffer());

            k_all = k_buffer.data();
            v_all = v_buffer.data();
        }
    } else {
        map_k = backend_->map_buffer(k.buffer());
        map_v = backend_->map_buffer(v.buffer());
        k_all = static_cast<const float*>(map_k.value());
        v_all = static_cast<const float*>(map_v.value());
        total_seq = seq_len;
    }

    // Compute attention for each head (parallelized)
    int heads_per_kv = num_heads / num_kv_heads;  // For GQA

    // Temporary buffer for context vectors
    std::vector<float> context(batch * seq_len * num_heads * head_dim);

    // Parallelize over batch * num_heads * seq_len
    int total_work = batch * num_heads * seq_len;
    size_t num_threads = get_num_threads();

    parallel_for(static_cast<size_t>(total_work), num_threads, [&](size_t work_idx_u) {
        int work_idx = static_cast<int>(work_idx_u);
        int b = work_idx / (num_heads * seq_len);
        int rem = work_idx % (num_heads * seq_len);
        int h = rem / seq_len;
        int q_pos = rem % seq_len;

        int kv_h = h / heads_per_kv;  // GQA: which KV head to use
        int abs_pos = start_pos + q_pos;

        // Thread-local scores buffer
        std::vector<float> scores(total_seq);
        const float* q_vec = q_data + (b * seq_len + q_pos) * num_heads * head_dim + h * head_dim;

        // Compute attention scores
        for (int k_pos = 0; k_pos < total_seq; k_pos++) {
            if (k_pos > abs_pos) {
                scores[k_pos] = -std::numeric_limits<float>::infinity();
            } else {
                const float* k_vec = k_all + k_pos * num_kv_heads * head_dim + kv_h * head_dim;
                float dot = 0;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[k_pos] = dot * scale;
            }
        }

        // Softmax
        float max_score = *std::max_element(scores.begin(), scores.begin() + total_seq);
        float sum = 0;
        for (int i = 0; i < total_seq; i++) {
            scores[i] = std::exp(scores[i] - max_score);
            sum += scores[i];
        }
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < total_seq; i++) {
            scores[i] *= inv_sum;
        }

        // Weighted sum of values
        float* ctx = context.data() + (b * seq_len + q_pos) * num_heads * head_dim + h * head_dim;
        std::fill(ctx, ctx + head_dim, 0.0f);

        for (int v_pos = 0; v_pos < total_seq; v_pos++) {
            const float* v_vec = v_all + v_pos * num_kv_heads * head_dim + kv_h * head_dim;
            float w = scores[v_pos];
            for (int d = 0; d < head_dim; d++) {
                ctx[d] += w * v_vec[d];
            }
        }
    });  // End parallel_for

    // Unmap K, V if not using cache
    if (!kv_cache || kv_cache->seq_len() == 0) {
        backend_->unmap_buffer(k.buffer());
        backend_->unmap_buffer(v.buffer());
    }

    backend_->unmap_buffer(q.buffer());

    // Output projection: context @ wo.T using optimized BLAS
    // context: [batch*seq, num_heads * head_dim]
    // wo shape: [hidden_dim, q_out_dim] with ne0=q_out_dim
    int total_tokens2 = batch * seq_len;
    int attn_out_dim = num_heads * head_dim;

    // output = context @ wo.T
    matmul_transb_fp16(context.data(), wo_data, out_data,
                       total_tokens2, hidden_dim, attn_out_dim);

    backend_->unmap_buffer(output.buffer());
    backend_->unmap_buffer(wo->buffer());

    return output;
}

// Tree attention: uses tree_mask instead of causal mask
// tree_mask[q_idx][k_idx] = true if query at tree position q_idx can attend to k_idx
// Note: tree positions are relative to the tree (0 to num_nodes-1)
// For KV cache positions (< start_pos), we allow all tree nodes to attend
Result<Tensor> TransformerModel::attention_tree(
    const Tensor& hidden,
    int layer,
    KVCache* kv_cache,
    int start_pos,
    const std::vector<std::vector<bool>>& tree_mask)
{
    std::string prefix = "blk." + std::to_string(layer) + ".";

    // Get attention weights
    const Tensor* wq = get_weight(prefix + "attn_q.weight");
    const Tensor* wk = get_weight(prefix + "attn_k.weight");
    const Tensor* wv = get_weight(prefix + "attn_v.weight");
    const Tensor* wo = get_weight(prefix + "attn_output.weight");

    if (!wq || !wk || !wv || !wo) {
        GRANITE_FAIL(ErrorCode::InvalidState,
                     "Missing attention weights for layer " + std::to_string(layer));
    }

    int batch = static_cast<int>(hidden.size(0));
    int num_nodes = static_cast<int>(hidden.size(1));  // Tree nodes
    int hidden_dim = model_config_.hidden_dim;
    int num_heads = model_config_.num_heads;
    int num_kv_heads = model_config_.num_kv_heads;
    int head_dim = model_config_.head_dim;
    int kv_dim = num_kv_heads * head_dim;

    // Allocate Q, K, V tensors for tree nodes
    std::vector<int64_t> q_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(num_nodes),
        static_cast<int64_t>(num_heads),
        static_cast<int64_t>(head_dim)
    };
    std::vector<int64_t> kv_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(num_nodes),
        static_cast<int64_t>(num_kv_heads),
        static_cast<int64_t>(head_dim)
    };

    auto q_result = Tensor::allocate(q_shape, DataType::FP32, backend_);
    auto k_result = Tensor::allocate(kv_shape, DataType::FP32, backend_);
    auto v_result = Tensor::allocate(kv_shape, DataType::FP32, backend_);

    if (!q_result.ok() || !k_result.ok() || !v_result.ok()) {
        GRANITE_FAIL(ErrorCode::AllocationFailed, "Failed to allocate Q/K/V tensors");
    }

    auto q = std::move(q_result).take();
    auto k = std::move(k_result).take();
    auto v = std::move(v_result).take();

    // Project Q, K, V
    auto map_h = backend_->map_buffer(hidden.buffer());
    auto map_q = backend_->map_buffer(q.buffer());
    auto map_k = backend_->map_buffer(k.buffer());
    auto map_v = backend_->map_buffer(v.buffer());
    auto map_wq = backend_->map_buffer(wq->buffer());
    auto map_wk = backend_->map_buffer(wk->buffer());
    auto map_wv = backend_->map_buffer(wv->buffer());

    if (!map_h.ok() || !map_q.ok() || !map_k.ok() || !map_v.ok() ||
        !map_wq.ok() || !map_wk.ok() || !map_wv.ok()) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to map attention buffers");
    }

    const float* h_data = static_cast<const float*>(map_h.value());
    float* q_data = static_cast<float*>(map_q.value());
    float* k_data = static_cast<float*>(map_k.value());
    float* v_data = static_cast<float*>(map_v.value());
    const uint16_t* wq_data = static_cast<const uint16_t*>(map_wq.value());
    const uint16_t* wk_data = static_cast<const uint16_t*>(map_wk.value());
    const uint16_t* wv_data = static_cast<const uint16_t*>(map_wv.value());

    // Q, K, V projections
    int q_out_dim = num_heads * head_dim;
    matmul_transb_fp16(h_data, wq_data, q_data, num_nodes, q_out_dim, hidden_dim);
    matmul_transb_fp16(h_data, wk_data, k_data, num_nodes, kv_dim, hidden_dim);
    matmul_transb_fp16(h_data, wv_data, v_data, num_nodes, kv_dim, hidden_dim);

    backend_->unmap_buffer(hidden.buffer());
    backend_->unmap_buffer(wq->buffer());
    backend_->unmap_buffer(wk->buffer());
    backend_->unmap_buffer(wv->buffer());

    // Unmap before RoPE
    backend_->unmap_buffer(q.buffer());
    backend_->unmap_buffer(k.buffer());
    backend_->unmap_buffer(v.buffer());

    // Apply RoPE - all tree nodes get positions starting at start_pos
    // Since tree nodes are speculative, they all share the same "position" conceptually
    // But for correct attention, each node gets its own position
    auto rope_result = rope_cache_.apply(q, k, start_pos, backend_);
    if (!rope_result.ok()) {
        GRANITE_LOG_WARN("RoPE failed: {}", rope_result.error().message());
    }

    // Remap after RoPE
    map_q = backend_->map_buffer(q.buffer());
    map_k = backend_->map_buffer(k.buffer());
    map_v = backend_->map_buffer(v.buffer());
    q_data = static_cast<float*>(map_q.value());
    k_data = static_cast<float*>(map_k.value());
    v_data = static_cast<float*>(map_v.value());

    // Compute attention with tree mask
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Total KV sequence = cached + tree nodes
    int cached_len = kv_cache ? kv_cache->seq_len() : 0;
    int total_kv = cached_len + num_nodes;

    // Allocate attention output
    std::vector<int64_t> output_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(num_nodes),
        static_cast<int64_t>(hidden_dim)
    };
    auto output_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
    if (!output_result.ok()) {
        return output_result.error();
    }
    auto output = std::move(output_result).take();

    auto map_out = backend_->map_buffer(output.buffer());
    auto map_wo = backend_->map_buffer(wo->buffer());

    if (!map_out.ok() || !map_wo.ok()) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to map output buffers");
    }

    float* out_data = static_cast<float*>(map_out.value());
    const uint16_t* wo_data = static_cast<const uint16_t*>(map_wo.value());

    // Get cached K, V if available
    const float* k_cached = nullptr;
    const float* v_cached = nullptr;
    std::vector<float> k_cache_buf, v_cache_buf;

    if (kv_cache && cached_len > 0) {
        auto [cached_k, cached_v] = kv_cache->get(layer);
        auto map_ck = backend_->map_buffer(cached_k.buffer());
        auto map_cv = backend_->map_buffer(cached_v.buffer());

        if (map_ck.ok() && map_cv.ok()) {
            const uint16_t* ck = static_cast<const uint16_t*>(map_ck.value());
            const uint16_t* cv = static_cast<const uint16_t*>(map_cv.value());

            k_cache_buf.resize(cached_len * num_kv_heads * head_dim);
            v_cache_buf.resize(cached_len * num_kv_heads * head_dim);

            // [1, num_kv_heads, cached_len, head_dim] -> [cached_len, num_kv_heads, head_dim]
            for (int h = 0; h < num_kv_heads; h++) {
                for (int s = 0; s < cached_len; s++) {
                    for (int d = 0; d < head_dim; d++) {
                        int src_idx = h * kv_cache->max_seq_len() * head_dim + s * head_dim + d;
                        int dst_idx = s * num_kv_heads * head_dim + h * head_dim + d;
                        k_cache_buf[dst_idx] = fp16_to_fp32(ck[src_idx]);
                        v_cache_buf[dst_idx] = fp16_to_fp32(cv[src_idx]);
                    }
                }
            }

            backend_->unmap_buffer(cached_k.buffer());
            backend_->unmap_buffer(cached_v.buffer());

            k_cached = k_cache_buf.data();
            v_cached = v_cache_buf.data();
        }
    }

    // Compute attention for each head
    int heads_per_kv = num_heads / num_kv_heads;
    std::vector<float> context(batch * num_nodes * num_heads * head_dim);

    int total_work = batch * num_heads * num_nodes;
    size_t num_threads = get_num_threads();

    parallel_for(static_cast<size_t>(total_work), num_threads, [&](size_t work_idx_u) {
        int work_idx = static_cast<int>(work_idx_u);
        int b = work_idx / (num_heads * num_nodes);
        int rem = work_idx % (num_heads * num_nodes);
        int h = rem / num_nodes;
        int q_node = rem % num_nodes;  // Query's tree node index

        int kv_h = h / heads_per_kv;

        // Thread-local scores buffer
        std::vector<float> scores(total_kv);
        const float* q_vec = q_data + (b * num_nodes + q_node) * num_heads * head_dim + h * head_dim;

        // Compute attention scores
        // 1. Attend to cached positions (all tree nodes can attend to full cache)
        for (int k_pos = 0; k_pos < cached_len; k_pos++) {
            const float* k_vec = k_cached + k_pos * num_kv_heads * head_dim + kv_h * head_dim;
            float dot = 0;
            for (int d = 0; d < head_dim; d++) {
                dot += q_vec[d] * k_vec[d];
            }
            scores[k_pos] = dot * scale;
        }

        // 2. Attend to tree nodes (use tree_mask)
        for (int k_node = 0; k_node < num_nodes; k_node++) {
            int k_pos_in_total = cached_len + k_node;

            // Check tree_mask: can q_node attend to k_node?
            if (tree_mask[q_node][k_node]) {
                const float* k_vec = k_data + (b * num_nodes + k_node) * num_kv_heads * head_dim + kv_h * head_dim;
                float dot = 0;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[k_pos_in_total] = dot * scale;
            } else {
                scores[k_pos_in_total] = -std::numeric_limits<float>::infinity();
            }
        }

        // Softmax
        float max_score = *std::max_element(scores.begin(), scores.begin() + total_kv);
        float sum = 0;
        for (int i = 0; i < total_kv; i++) {
            scores[i] = std::exp(scores[i] - max_score);
            sum += scores[i];
        }
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < total_kv; i++) {
            scores[i] *= inv_sum;
        }

        // Weighted sum of values
        float* ctx = context.data() + (b * num_nodes + q_node) * num_heads * head_dim + h * head_dim;
        std::fill(ctx, ctx + head_dim, 0.0f);

        // From cached values
        for (int v_pos = 0; v_pos < cached_len; v_pos++) {
            const float* v_vec = v_cached + v_pos * num_kv_heads * head_dim + kv_h * head_dim;
            float w = scores[v_pos];
            for (int d = 0; d < head_dim; d++) {
                ctx[d] += w * v_vec[d];
            }
        }

        // From tree values
        for (int v_node = 0; v_node < num_nodes; v_node++) {
            int v_pos_in_total = cached_len + v_node;
            const float* v_vec = v_data + (b * num_nodes + v_node) * num_kv_heads * head_dim + kv_h * head_dim;
            float w = scores[v_pos_in_total];
            for (int d = 0; d < head_dim; d++) {
                ctx[d] += w * v_vec[d];
            }
        }
    });  // End parallel_for

    backend_->unmap_buffer(q.buffer());
    backend_->unmap_buffer(k.buffer());
    backend_->unmap_buffer(v.buffer());

    // Output projection
    int attn_out_dim = num_heads * head_dim;
    matmul_transb_fp16(context.data(), wo_data, out_data, num_nodes, hidden_dim, attn_out_dim);

    backend_->unmap_buffer(output.buffer());
    backend_->unmap_buffer(wo->buffer());

    return output;
}

// =============================================================================
// GPU-Accelerated Tree Attention
// =============================================================================

Result<Tensor> TransformerModel::attention_tree_gpu(
    const Tensor& hidden,
    int layer,
    KVCache* kv_cache,
    int start_pos,
    const std::vector<int>& parent_indices)
{
#ifdef GRANITE_HAS_METAL
    auto* mc = get_metal_compute();
    if (!mc || !mc->is_initialized()) {
        // Fall back to CPU path by building tree_mask
        int num_nodes = static_cast<int>(parent_indices.size());
        std::vector<std::vector<bool>> tree_mask(num_nodes, std::vector<bool>(num_nodes, false));
        for (int i = 0; i < num_nodes; i++) {
            tree_mask[i][i] = true;
            int current = parent_indices[i];
            while (current >= 0 && current < num_nodes) {
                tree_mask[i][current] = true;
                current = parent_indices[current];
            }
        }
        return attention_tree(hidden, layer, kv_cache, start_pos, tree_mask);
    }

    std::string prefix = "blk." + std::to_string(layer) + ".";

    // Get attention weights
    const Tensor* wq = get_weight(prefix + "attn_q.weight");
    const Tensor* wk = get_weight(prefix + "attn_k.weight");
    const Tensor* wv = get_weight(prefix + "attn_v.weight");
    const Tensor* wo = get_weight(prefix + "attn_output.weight");

    if (!wq || !wk || !wv || !wo) {
        GRANITE_FAIL(ErrorCode::InvalidState,
                     "Missing attention weights for layer " + std::to_string(layer));
    }

    int batch = static_cast<int>(hidden.size(0));
    int num_nodes = static_cast<int>(hidden.size(1));
    int hidden_dim = model_config_.hidden_dim;
    int num_heads = model_config_.num_heads;
    int num_kv_heads = model_config_.num_kv_heads;
    int head_dim = model_config_.head_dim;
    int kv_dim = num_kv_heads * head_dim;

    // Allocate Q, K, V tensors
    std::vector<int64_t> q_shape = {1, static_cast<int64_t>(num_nodes),
                                    static_cast<int64_t>(num_heads), static_cast<int64_t>(head_dim)};
    std::vector<int64_t> kv_shape = {1, static_cast<int64_t>(num_nodes),
                                     static_cast<int64_t>(num_kv_heads), static_cast<int64_t>(head_dim)};

    auto q_result = Tensor::allocate(q_shape, DataType::FP32, backend_);
    auto k_result = Tensor::allocate(kv_shape, DataType::FP32, backend_);
    auto v_result = Tensor::allocate(kv_shape, DataType::FP32, backend_);

    if (!q_result.ok() || !k_result.ok() || !v_result.ok()) {
        GRANITE_FAIL(ErrorCode::AllocationFailed, "Failed to allocate Q/K/V tensors");
    }

    auto q = std::move(q_result).take();
    auto k = std::move(k_result).take();
    auto v = std::move(v_result).take();

    // QKV projection (CPU for now - could use GPU matvec kernels)
    auto map_h = backend_->map_buffer(hidden.buffer());
    auto map_q = backend_->map_buffer(q.buffer());
    auto map_k = backend_->map_buffer(k.buffer());
    auto map_v = backend_->map_buffer(v.buffer());
    auto map_wq = backend_->map_buffer(wq->buffer());
    auto map_wk = backend_->map_buffer(wk->buffer());
    auto map_wv = backend_->map_buffer(wv->buffer());

    if (!map_h.ok() || !map_q.ok() || !map_k.ok() || !map_v.ok() ||
        !map_wq.ok() || !map_wk.ok() || !map_wv.ok()) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to map attention buffers");
    }

    const float* h_data = static_cast<const float*>(map_h.value());
    float* q_data = static_cast<float*>(map_q.value());
    float* k_data = static_cast<float*>(map_k.value());
    float* v_data = static_cast<float*>(map_v.value());
    const uint16_t* wq_data = static_cast<const uint16_t*>(map_wq.value());
    const uint16_t* wk_data = static_cast<const uint16_t*>(map_wk.value());
    const uint16_t* wv_data = static_cast<const uint16_t*>(map_wv.value());

    int q_out_dim = num_heads * head_dim;
    matmul_transb_fp16(h_data, wq_data, q_data, num_nodes, q_out_dim, hidden_dim);
    matmul_transb_fp16(h_data, wk_data, k_data, num_nodes, kv_dim, hidden_dim);
    matmul_transb_fp16(h_data, wv_data, v_data, num_nodes, kv_dim, hidden_dim);

    backend_->unmap_buffer(hidden.buffer());
    backend_->unmap_buffer(wq->buffer());
    backend_->unmap_buffer(wk->buffer());
    backend_->unmap_buffer(wv->buffer());
    backend_->unmap_buffer(q.buffer());
    backend_->unmap_buffer(k.buffer());
    backend_->unmap_buffer(v.buffer());

    // Apply RoPE
    auto rope_result = rope_cache_.apply(q, k, start_pos, backend_);
    if (!rope_result.ok()) {
        GRANITE_LOG_WARN("RoPE failed: {}", rope_result.error().message());
    }

    // Prepare GPU buffers for tree attention
    // Q needs to be reshaped to [num_heads, num_nodes, head_dim]
    size_t q_gpu_size = num_heads * num_nodes * head_dim * sizeof(float);
    size_t kv_tree_size = num_kv_heads * num_nodes * head_dim * sizeof(uint16_t);  // FP16
    size_t parent_size = num_nodes * sizeof(int);
    size_t output_size = num_heads * num_nodes * head_dim * sizeof(float);

    MTL::Buffer* q_gpu = mc->create_buffer(q_gpu_size, true);
    MTL::Buffer* k_tree_gpu = mc->create_buffer(kv_tree_size, true);
    MTL::Buffer* v_tree_gpu = mc->create_buffer(kv_tree_size, true);
    MTL::Buffer* parent_gpu = mc->create_buffer(parent_size, true);
    MTL::Buffer* output_gpu = mc->create_buffer(output_size, true);

    if (!q_gpu || !k_tree_gpu || !v_tree_gpu || !parent_gpu || !output_gpu) {
        if (q_gpu) q_gpu->release();
        if (k_tree_gpu) k_tree_gpu->release();
        if (v_tree_gpu) v_tree_gpu->release();
        if (parent_gpu) parent_gpu->release();
        if (output_gpu) output_gpu->release();
        GRANITE_FAIL(ErrorCode::AllocationFailed, "Failed to allocate GPU buffers for tree attention");
    }

    // Copy and reshape Q: [1, num_nodes, num_heads, head_dim] -> [num_heads, num_nodes, head_dim]
    map_q = backend_->map_buffer(q.buffer());
    q_data = static_cast<float*>(map_q.value());
    float* q_gpu_ptr = static_cast<float*>(q_gpu->contents());
    for (int h = 0; h < num_heads; h++) {
        for (int n = 0; n < num_nodes; n++) {
            for (int d = 0; d < head_dim; d++) {
                int src_idx = n * num_heads * head_dim + h * head_dim + d;
                int dst_idx = h * num_nodes * head_dim + n * head_dim + d;
                q_gpu_ptr[dst_idx] = q_data[src_idx];
            }
        }
    }
    backend_->unmap_buffer(q.buffer());

    // Copy and convert K/V to FP16: [1, num_nodes, num_kv_heads, head_dim] -> [num_kv_heads, num_nodes, head_dim]
    map_k = backend_->map_buffer(k.buffer());
    map_v = backend_->map_buffer(v.buffer());
    k_data = static_cast<float*>(map_k.value());
    v_data = static_cast<float*>(map_v.value());
    uint16_t* k_gpu_ptr = static_cast<uint16_t*>(k_tree_gpu->contents());
    uint16_t* v_gpu_ptr = static_cast<uint16_t*>(v_tree_gpu->contents());

    for (int kv_h = 0; kv_h < num_kv_heads; kv_h++) {
        for (int n = 0; n < num_nodes; n++) {
            for (int d = 0; d < head_dim; d++) {
                int src_idx = n * num_kv_heads * head_dim + kv_h * head_dim + d;
                int dst_idx = kv_h * num_nodes * head_dim + n * head_dim + d;
                k_gpu_ptr[dst_idx] = fp32_to_fp16(k_data[src_idx]);
                v_gpu_ptr[dst_idx] = fp32_to_fp16(v_data[src_idx]);
            }
        }
    }
    backend_->unmap_buffer(k.buffer());
    backend_->unmap_buffer(v.buffer());

    // Copy parent indices
    int* parent_ptr = static_cast<int*>(parent_gpu->contents());
    std::memcpy(parent_ptr, parent_indices.data(), parent_size);

    // Get KV cache buffers
    int cache_len = kv_cache ? kv_cache->seq_len() : 0;
    MTL::Buffer* k_cache_gpu = nullptr;
    MTL::Buffer* v_cache_gpu = nullptr;

    if (kv_cache && cache_len > 0) {
        auto [cached_k, cached_v] = kv_cache->get(layer);
        k_cache_gpu = static_cast<MTL::Buffer*>(backend_->get_native_buffer(cached_k.buffer()));
        v_cache_gpu = static_cast<MTL::Buffer*>(backend_->get_native_buffer(cached_v.buffer()));
    }

    // Call GPU tree attention kernel
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto attn_result = mc->attention_tree(
        q_gpu, k_cache_gpu, v_cache_gpu, k_tree_gpu, v_tree_gpu, parent_gpu, output_gpu,
        num_heads, num_kv_heads, num_nodes, cache_len, head_dim, scale);

    mc->sync();

    if (!attn_result.ok()) {
        q_gpu->release();
        k_tree_gpu->release();
        v_tree_gpu->release();
        parent_gpu->release();
        output_gpu->release();
        return attn_result.error();
    }

    // Allocate final output tensor
    std::vector<int64_t> output_shape = {1, static_cast<int64_t>(num_nodes),
                                          static_cast<int64_t>(hidden_dim)};
    auto output_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
    if (!output_result.ok()) {
        q_gpu->release();
        k_tree_gpu->release();
        v_tree_gpu->release();
        parent_gpu->release();
        output_gpu->release();
        return output_result.error();
    }
    auto output = std::move(output_result).take();

    // Copy attention output and reshape: [num_heads, num_nodes, head_dim] -> [1, num_nodes, num_heads * head_dim]
    auto map_out = backend_->map_buffer(output.buffer());
    auto map_wo = backend_->map_buffer(wo->buffer());
    float* out_data = static_cast<float*>(map_out.value());
    const uint16_t* wo_data = static_cast<const uint16_t*>(map_wo.value());
    float* attn_out = static_cast<float*>(output_gpu->contents());

    // Reshape attention output for output projection
    std::vector<float> context(num_nodes * num_heads * head_dim);
    for (int n = 0; n < num_nodes; n++) {
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                int src_idx = h * num_nodes * head_dim + n * head_dim + d;
                int dst_idx = n * num_heads * head_dim + h * head_dim + d;
                context[dst_idx] = attn_out[src_idx];
            }
        }
    }

    // Output projection
    int attn_out_dim = num_heads * head_dim;
    matmul_transb_fp16(context.data(), wo_data, out_data, num_nodes, hidden_dim, attn_out_dim);

    backend_->unmap_buffer(output.buffer());
    backend_->unmap_buffer(wo->buffer());

    // Cleanup GPU buffers
    q_gpu->release();
    k_tree_gpu->release();
    v_tree_gpu->release();
    parent_gpu->release();
    output_gpu->release();

    return output;
#else
    // No Metal - fall back to CPU path
    int num_nodes = static_cast<int>(parent_indices.size());
    std::vector<std::vector<bool>> tree_mask(num_nodes, std::vector<bool>(num_nodes, false));
    for (int i = 0; i < num_nodes; i++) {
        tree_mask[i][i] = true;
        int current = parent_indices[i];
        while (current >= 0 && current < num_nodes) {
            tree_mask[i][current] = true;
            current = parent_indices[current];
        }
    }
    return attention_tree(hidden, layer, kv_cache, start_pos, tree_mask);
#endif
}

// GPU-accelerated transformer block for tree speculation
Result<Tensor> TransformerModel::transformer_block_tree_gpu(
    const Tensor& hidden,
    int layer,
    KVCache* kv_cache,
    int start_pos,
    const std::vector<int>& parent_indices)
{
    std::string prefix = "blk." + std::to_string(layer) + ".";

    // Get weights
    const Tensor* attn_norm_weight = get_weight(prefix + "attn_norm.weight");
    const Tensor* ffn_norm_weight = get_weight(prefix + "ffn_norm.weight");

    // Save residual for later
    Tensor residual = hidden;

    // 1. RMSNorm for attention
    auto attn_input = apply_rms_norm(hidden, attn_norm_weight);

    // 2. GPU-accelerated tree attention
    auto attn_result = attention_tree_gpu(attn_input, layer, kv_cache, start_pos, parent_indices);
    if (!attn_result.ok()) {
        return attn_result.error();
    }
    auto attn_output = std::move(attn_result).take();

    // 3. Residual add: hidden = residual + attn_output
    auto post_attn = add_tensors(residual, attn_output);

    // 4. RMSNorm for FFN
    auto ffn_input = apply_rms_norm(post_attn, ffn_norm_weight);

    // 5. Feed forward (SwiGLU)
    auto ffn_result = feed_forward(ffn_input, layer);
    if (!ffn_result.ok()) {
        return ffn_result.error();
    }
    auto ffn_output = std::move(ffn_result).take();

    // 6. Residual add: output = post_attn + ffn_output
    auto output = add_tensors(post_attn, ffn_output);
    return output;
}

// =============================================================================
// SECTION 7: CPU Feed-Forward (SwiGLU)
// =============================================================================

Result<Tensor> TransformerModel::feed_forward(const Tensor& hidden, int layer) {
    std::string prefix = "blk." + std::to_string(layer) + ".";

    // SwiGLU FFN: output = down(silu(gate(x)) * up(x))
    const Tensor* w_gate = get_weight(prefix + "ffn_gate.weight");
    const Tensor* w_up = get_weight(prefix + "ffn_up.weight");
    const Tensor* w_down = get_weight(prefix + "ffn_down.weight");

    if (!w_gate || !w_up || !w_down) {
        GRANITE_FAIL(ErrorCode::InvalidState,
                     "Missing FFN weights for layer " + std::to_string(layer));
    }

    int batch = static_cast<int>(hidden.size(0));
    int seq_len = static_cast<int>(hidden.size(1));
    int hidden_dim = model_config_.hidden_dim;
    int intermediate_dim = model_config_.intermediate_dim;

    // Allocate output
    std::vector<int64_t> output_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(hidden_dim)
    };
    auto output_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
    if (!output_result.ok()) {
        return output_result.error();
    }
    auto output = std::move(output_result).take();

    // Map buffers
    auto map_h = backend_->map_buffer(hidden.buffer());
    auto map_o = backend_->map_buffer(output.buffer());
    auto map_wg = backend_->map_buffer(w_gate->buffer());
    auto map_wu = backend_->map_buffer(w_up->buffer());
    auto map_wd = backend_->map_buffer(w_down->buffer());

    if (!map_h.ok() || !map_o.ok() || !map_wg.ok() || !map_wu.ok() || !map_wd.ok()) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to map FFN buffers");
    }

    const float* h_data = static_cast<const float*>(map_h.value());
    float* o_data = static_cast<float*>(map_o.value());
    const uint16_t* wg_data = static_cast<const uint16_t*>(map_wg.value());
    const uint16_t* wu_data = static_cast<const uint16_t*>(map_wu.value());
    const uint16_t* wd_data = static_cast<const uint16_t*>(map_wd.value());

    // Use optimized BLAS-based matmul
    int total_tokens = batch * seq_len;

    // Allocate intermediate buffers
    std::vector<float> gate_buf(total_tokens * intermediate_dim);
    std::vector<float> up_buf(total_tokens * intermediate_dim);
    std::vector<float> intermediate_buf(total_tokens * intermediate_dim);

    // gate = hidden @ w_gate.T  [total_tokens, intermediate_dim]
    // Weight shape: [intermediate_dim, hidden_dim]
    matmul_transb_fp16(h_data, wg_data, gate_buf.data(),
                       total_tokens, intermediate_dim, hidden_dim);

    // up = hidden @ w_up.T  [total_tokens, intermediate_dim]
    matmul_transb_fp16(h_data, wu_data, up_buf.data(),
                       total_tokens, intermediate_dim, hidden_dim);

    // Apply SiLU to gate and multiply with up
    silu_inplace(gate_buf.data(), total_tokens * intermediate_dim);
    elementwise_mul(gate_buf.data(), up_buf.data(), intermediate_buf.data(),
                    total_tokens * intermediate_dim);

    // output = intermediate @ w_down.T  [total_tokens, hidden_dim]
    // Down weight shape: [hidden_dim, intermediate_dim]
    matmul_transb_fp16(intermediate_buf.data(), wd_data, o_data,
                       total_tokens, hidden_dim, intermediate_dim);

    backend_->unmap_buffer(hidden.buffer());
    backend_->unmap_buffer(output.buffer());
    backend_->unmap_buffer(w_gate->buffer());
    backend_->unmap_buffer(w_up->buffer());
    backend_->unmap_buffer(w_down->buffer());

    return output;
}

// =============================================================================
// SECTION 8: GPU/Metal Implementations
// =============================================================================

#ifdef GRANITE_HAS_METAL

// GPU Feed-Forward using Q4_K quantized weights
Result<Tensor> TransformerModel::feed_forward_gpu(const Tensor& hidden, int layer) {
    std::string prefix = "blk." + std::to_string(layer) + ".";

    // Get raw Q4_K weights for GPU
    const RawWeight* w_gate = get_raw_weight(prefix + "ffn_gate.weight");
    const RawWeight* w_up = get_raw_weight(prefix + "ffn_up.weight");
    const RawWeight* w_down = get_raw_weight(prefix + "ffn_down.weight");

    // Fall back to CPU if raw weights not available
    if (!w_gate || !w_up || !w_down) {
        GRANITE_LOG_DEBUG("GPU weights not available for layer {}, falling back to CPU", layer);
        return feed_forward(hidden, layer);
    }

    // Support Q4_K, Q5_K, Q6_K, Q3_K, Q2_K, Q8_0, Q4_0, IQ4_NL, IQ4_XS, and IQ3_S
    GGMLType qtype = w_gate->quant_type;
    if (qtype != GGMLType::Q4_K && qtype != GGMLType::Q5_K && qtype != GGMLType::Q6_K &&
        qtype != GGMLType::Q3_K && qtype != GGMLType::Q2_K &&
        qtype != GGMLType::Q8_0 && qtype != GGMLType::Q4_0 &&
        qtype != GGMLType::IQ4_NL && qtype != GGMLType::IQ4_XS && qtype != GGMLType::IQ3_S) {
        return feed_forward(hidden, layer);
    }

    auto* gpu = get_metal_compute();
    if (!gpu || !gpu->is_initialized()) {
        return feed_forward(hidden, layer);
    }

    int batch = static_cast<int>(hidden.size(0));
    int seq_len = static_cast<int>(hidden.size(1));
    int hidden_dim = model_config_.hidden_dim;
    int intermediate_dim = model_config_.intermediate_dim;
    int total_tokens = batch * seq_len;

    // Allocate output tensor
    std::vector<int64_t> output_shape = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(hidden_dim)
    };
    auto output_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
    if (!output_result.ok()) {
        return output_result.error();
    }
    auto output = std::move(output_result).take();

    // Get Metal buffers
    auto* h_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(hidden.buffer()));
    auto* o_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(output.buffer()));
    auto* wg_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(w_gate->buffer));
    auto* wu_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(w_up->buffer));
    auto* wd_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(w_down->buffer));

    if (!h_buf || !o_buf || !wg_buf || !wu_buf || !wd_buf) {
        GRANITE_LOG_WARN("Failed to get Metal buffers, falling back to CPU");
        return feed_forward(hidden, layer);
    }

    // Use pooled buffers - decode pool for M=1, prefill pool for M>1
    bool use_decode_pool = (total_tokens == 1) && decode_pool_ && decode_pool_->initialized &&
                           decode_pool_->ffn_gate_buf && decode_pool_->ffn_up_buf;
    bool use_prefill_pool = (total_tokens > 1) && prefill_pool_ && prefill_pool_->initialized &&
                            prefill_pool_->max_tokens >= total_tokens;

    MTL::Buffer* gate_buf;
    MTL::Buffer* up_buf;

    if (use_decode_pool) {
        gate_buf = static_cast<MTL::Buffer*>(decode_pool_->ffn_gate_buf);
        up_buf = static_cast<MTL::Buffer*>(decode_pool_->ffn_up_buf);
    } else if (use_prefill_pool) {
        gate_buf = static_cast<MTL::Buffer*>(prefill_pool_->ffn_gate_buf);
        up_buf = static_cast<MTL::Buffer*>(prefill_pool_->ffn_up_buf);
    } else {
        // Fallback: allocate (should rarely happen if pools are initialized)
        gate_buf = gpu->create_buffer(total_tokens * intermediate_dim * sizeof(float));
        up_buf = gpu->create_buffer(total_tokens * intermediate_dim * sizeof(float));
    }

    bool use_pool = use_decode_pool || use_prefill_pool;

    if (!gate_buf || !up_buf) {
        if (!use_pool) {
            if (gate_buf) gate_buf->release();
            if (up_buf) up_buf->release();
        }
        return feed_forward(hidden, layer);
    }

    // FFN computation on GPU:
    // gate = hidden @ w_gate.T
    // up = hidden @ w_up.T
    // gate = silu(gate)
    // intermediate = gate * up
    // output = intermediate @ w_down.T
    // NOTE: Each weight may have different quantization type!

    // Helper lambdas to dispatch based on actual weight type
    auto dispatch_matvec = [&](MTL::Buffer* in, MTL::Buffer* weight, MTL::Buffer* out,
                               GGMLType wtype, int K, int N) {
        switch (wtype) {
            case GGMLType::Q8_0:  gpu->matvec_q8_0(in, weight, out, K, N); break;
            case GGMLType::Q4_0:  gpu->matvec_q4_0(in, weight, out, K, N); break;
            case GGMLType::IQ4_NL: gpu->matvec_iq4_nl(in, weight, out, K, N); break;
            case GGMLType::IQ4_XS: gpu->matvec_iq4_xs(in, weight, out, K, N); break;
            case GGMLType::IQ3_S: gpu->matvec_iq3_s(in, weight, out, K, N); break;
            case GGMLType::Q6_K: gpu->matvec_q6_k(in, weight, out, K, N); break;
            case GGMLType::Q5_K: gpu->matvec_q5_k(in, weight, out, K, N); break;
            case GGMLType::Q4_K: gpu->matvec_q4k(in, weight, out, K, N); break;
            case GGMLType::Q3_K: gpu->matvec_q3_k(in, weight, out, K, N); break;
            case GGMLType::Q2_K: gpu->matvec_q2_k(in, weight, out, K, N); break;
            default: gpu->matvec_q4k(in, weight, out, K, N); break;
        }
    };

    // For single token (decode), use matvec with each weight's actual type
    if (total_tokens == 1) {
        dispatch_matvec(h_buf, wg_buf, gate_buf, w_gate->quant_type, hidden_dim, intermediate_dim);
        dispatch_matvec(h_buf, wu_buf, up_buf, w_up->quant_type, hidden_dim, intermediate_dim);
        gpu->silu_mul(gate_buf, up_buf, gate_buf, intermediate_dim);
        dispatch_matvec(gate_buf, wd_buf, o_buf, w_down->quant_type, intermediate_dim, hidden_dim);
    } else {
        // Batched: use matmul with each weight's actual type
        auto dispatch_matmul = [&](MTL::Buffer* in, MTL::Buffer* weight, MTL::Buffer* out,
                                   GGMLType wtype, int M, int K, int N) {
            switch (wtype) {
                case GGMLType::Q8_0:  gpu->matmul_q8_0(in, weight, out, M, K, N); break;
                case GGMLType::Q4_0:  gpu->matmul_q4_0(in, weight, out, M, K, N); break;
                case GGMLType::IQ4_NL: gpu->matmul_iq4_nl(in, weight, out, M, K, N); break;
                case GGMLType::IQ4_XS: gpu->matmul_iq4_xs(in, weight, out, M, K, N); break;
                case GGMLType::IQ3_S: gpu->matmul_iq3_s(in, weight, out, M, K, N); break;
                case GGMLType::Q6_K: gpu->matmul_q6_k(in, weight, out, M, K, N); break;
                case GGMLType::Q5_K: gpu->matmul_q5_k(in, weight, out, M, K, N); break;
                case GGMLType::Q4_K: gpu->matmul_q4k(in, weight, out, M, K, N); break;
                case GGMLType::Q3_K: gpu->matmul_q3_k(in, weight, out, M, K, N); break;
                case GGMLType::Q2_K: gpu->matmul_q2_k(in, weight, out, M, K, N); break;
                default: gpu->matmul_q4k(in, weight, out, M, K, N); break;
            }
        };

        dispatch_matmul(h_buf, wg_buf, gate_buf, w_gate->quant_type, total_tokens, hidden_dim, intermediate_dim);
        dispatch_matmul(h_buf, wu_buf, up_buf, w_up->quant_type, total_tokens, hidden_dim, intermediate_dim);
        gpu->silu_mul(gate_buf, up_buf, gate_buf, total_tokens * intermediate_dim);
        dispatch_matmul(gate_buf, wd_buf, o_buf, w_down->quant_type, total_tokens, intermediate_dim, hidden_dim);
    }

    // NOTE: No sync here - let commands batch across layers for better pipelining
    // The sync happens at the end of forward() or when results are needed

    // Clean up intermediate buffers (only if not using pool)
    if (!use_pool) {
        gate_buf->release();
        up_buf->release();
    }

    return output;
}

// Allocate GPU KV cache
Result<void> TransformerModel::allocate_gpu_kv_cache(int max_seq_len) {
    auto* gpu = get_metal_compute();
    if (!gpu || !gpu->is_initialized()) {
        GRANITE_FAIL(ErrorCode::InternalError, "MetalCompute not initialized");
    }

    gpu_kv_cache_ = std::make_unique<GPUKVCache>();
    gpu_kv_cache_->max_seq_len = max_seq_len;
    gpu_kv_cache_->num_kv_heads = model_config_.num_kv_heads;
    gpu_kv_cache_->head_dim = model_config_.head_dim;
    gpu_kv_cache_->current_len = 0;
    gpu_kv_cache_->layers.resize(model_config_.num_layers);

    for (int layer = 0; layer < model_config_.num_layers; layer++) {
        auto [k_cache, v_cache] = gpu->create_kv_cache(
            model_config_.num_kv_heads,
            max_seq_len,
            model_config_.head_dim
        );
        if (!k_cache || !v_cache) {
            GRANITE_FAIL(ErrorCode::OutOfMemory, "Failed to allocate GPU KV cache");
        }
        gpu_kv_cache_->layers[layer].k_cache = k_cache;
        gpu_kv_cache_->layers[layer].v_cache = v_cache;
    }

    // Also initialize decode buffer pool
    auto pool_result = init_decode_pool();
    if (!pool_result.ok()) {
        GRANITE_LOG_WARN("Failed to initialize decode buffer pool: {}", pool_result.error().message());
    }

    return {};
}

// Initialize decode buffer pool for single-token decode
Result<void> TransformerModel::init_decode_pool() {
    if (decode_pool_ && decode_pool_->initialized) {
        return {};  // Already initialized
    }

    decode_pool_ = std::make_unique<DecodeBufferPool>();

    std::vector<int64_t> hidden_shape = {1, 1, static_cast<int64_t>(model_config_.hidden_dim)};
    std::vector<int64_t> logits_shape = {1, 1, static_cast<int64_t>(model_config_.vocab_size)};

    // Allocate buffers
    auto attn_in = Tensor::allocate(hidden_shape, DataType::FP32, backend_);
    auto post_attn = Tensor::allocate(hidden_shape, DataType::FP32, backend_);
    auto ffn_in = Tensor::allocate(hidden_shape, DataType::FP32, backend_);
    auto block_out = Tensor::allocate(hidden_shape, DataType::FP32, backend_);
    auto attn_layer_out = Tensor::allocate(hidden_shape, DataType::FP32, backend_);
    auto norm_out = Tensor::allocate(hidden_shape, DataType::FP32, backend_);
    auto logits = Tensor::allocate(logits_shape, DataType::FP32, backend_);

    if (!attn_in.ok() || !post_attn.ok() || !ffn_in.ok() ||
        !block_out.ok() || !attn_layer_out.ok() || !norm_out.ok() || !logits.ok()) {
        return Error(ErrorCode::OutOfMemory, "Failed to allocate decode buffer pool");
    }

    decode_pool_->attn_input = std::move(attn_in).take();
    decode_pool_->post_attn = std::move(post_attn).take();
    decode_pool_->ffn_input = std::move(ffn_in).take();
    decode_pool_->block_output = std::move(block_out).take();
    decode_pool_->attn_layer_out = std::move(attn_layer_out).take();
    decode_pool_->norm_out = std::move(norm_out).take();
    decode_pool_->logits = std::move(logits).take();

    // Allocate GPU-specific buffers for attention and FFN
    auto* gpu = get_metal_compute();
    if (gpu && gpu->is_initialized()) {
        int q_dim = model_config_.num_heads * model_config_.head_dim;
        int kv_dim = model_config_.num_kv_heads * model_config_.head_dim;
        int intermediate_dim = model_config_.intermediate_dim;

        decode_pool_->q_buf = gpu->create_buffer(q_dim * sizeof(float));
        decode_pool_->k_buf = gpu->create_buffer(kv_dim * sizeof(float));
        decode_pool_->v_buf = gpu->create_buffer(kv_dim * sizeof(float));
        decode_pool_->attn_out_buf = gpu->create_buffer(q_dim * sizeof(float));
        decode_pool_->ffn_gate_buf = gpu->create_buffer(intermediate_dim * sizeof(float));
        decode_pool_->ffn_up_buf = gpu->create_buffer(intermediate_dim * sizeof(float));

        if (!decode_pool_->q_buf || !decode_pool_->k_buf || !decode_pool_->v_buf ||
            !decode_pool_->attn_out_buf || !decode_pool_->ffn_gate_buf || !decode_pool_->ffn_up_buf) {
            GRANITE_LOG_WARN("Failed to allocate some GPU buffers for decode pool");
        } else {
            GRANITE_LOG_DEBUG("GPU decode buffers allocated (q={}, kv={}, ffn={})",
                             q_dim, kv_dim, intermediate_dim);
        }
    }

    decode_pool_->initialized = true;

    GRANITE_LOG_DEBUG("Decode buffer pool initialized");
    return {};
}

// Initialize or resize prefill buffer pool
Result<void> TransformerModel::ensure_prefill_pool(int num_tokens) {
    auto* gpu = get_metal_compute();
    if (!gpu || !gpu->is_initialized()) {
        return {};  // No GPU, no pool needed
    }

    // Check if existing pool is sufficient
    if (prefill_pool_ && prefill_pool_->initialized && prefill_pool_->max_tokens >= num_tokens) {
        return {};  // Pool is large enough
    }

    // Release old pool if exists
    if (prefill_pool_) {
        if (prefill_pool_->q_buf) static_cast<MTL::Buffer*>(prefill_pool_->q_buf)->release();
        if (prefill_pool_->k_buf) static_cast<MTL::Buffer*>(prefill_pool_->k_buf)->release();
        if (prefill_pool_->v_buf) static_cast<MTL::Buffer*>(prefill_pool_->v_buf)->release();
        if (prefill_pool_->attn_out_buf) static_cast<MTL::Buffer*>(prefill_pool_->attn_out_buf)->release();
        if (prefill_pool_->ffn_gate_buf) static_cast<MTL::Buffer*>(prefill_pool_->ffn_gate_buf)->release();
        if (prefill_pool_->ffn_up_buf) static_cast<MTL::Buffer*>(prefill_pool_->ffn_up_buf)->release();
    } else {
        prefill_pool_ = std::make_unique<PrefillBufferPool>();
    }

    // Allocate new buffers with extra headroom (round up to power of 2 for reuse)
    int alloc_tokens = 64;
    while (alloc_tokens < num_tokens) alloc_tokens *= 2;

    int q_dim = model_config_.num_heads * model_config_.head_dim;
    int kv_dim = model_config_.num_kv_heads * model_config_.head_dim;
    int intermediate_dim = model_config_.intermediate_dim;

    prefill_pool_->q_buf = gpu->create_buffer(alloc_tokens * q_dim * sizeof(float));
    prefill_pool_->k_buf = gpu->create_buffer(alloc_tokens * kv_dim * sizeof(float));
    prefill_pool_->v_buf = gpu->create_buffer(alloc_tokens * kv_dim * sizeof(float));
    prefill_pool_->attn_out_buf = gpu->create_buffer(alloc_tokens * q_dim * sizeof(float));
    prefill_pool_->ffn_gate_buf = gpu->create_buffer(alloc_tokens * intermediate_dim * sizeof(float));
    prefill_pool_->ffn_up_buf = gpu->create_buffer(alloc_tokens * intermediate_dim * sizeof(float));

    if (!prefill_pool_->q_buf || !prefill_pool_->k_buf || !prefill_pool_->v_buf ||
        !prefill_pool_->attn_out_buf || !prefill_pool_->ffn_gate_buf || !prefill_pool_->ffn_up_buf) {
        GRANITE_FAIL(ErrorCode::OutOfMemory, "Failed to allocate prefill buffer pool");
    }

    prefill_pool_->max_tokens = alloc_tokens;
    prefill_pool_->initialized = true;

    GRANITE_LOG_DEBUG("Prefill buffer pool initialized with {} token capacity", alloc_tokens);
    return {};
}

// Sync CPU KV cache to GPU KV cache (for transitioning from prefill to decode)
Result<void> TransformerModel::sync_cpu_to_gpu_kv_cache(KVCache* kv_cache) {
    if (!kv_cache || !kv_cache->is_allocated()) {
        GRANITE_FAIL(ErrorCode::InvalidState, "CPU KV cache not allocated");
    }
    if (!gpu_kv_cache_ || !gpu_kv_cache_->is_allocated()) {
        GRANITE_FAIL(ErrorCode::InvalidState, "GPU KV cache not allocated");
    }

    auto* gpu = get_metal_compute();
    if (!gpu || !gpu->is_initialized()) {
        GRANITE_FAIL(ErrorCode::InternalError, "MetalCompute not initialized");
    }

    int cpu_len = kv_cache->seq_len();
    if (cpu_len == 0) {
        gpu_kv_cache_->current_len = 0;
        return {};
    }

    if (cpu_len > gpu_kv_cache_->max_seq_len) {
        GRANITE_FAIL(ErrorCode::InvalidArgument, "CPU cache too large for GPU cache");
    }

    int num_kv_heads = model_config_.num_kv_heads;
    int head_dim = model_config_.head_dim;

    // Copy each layer's cache
    for (int layer = 0; layer < model_config_.num_layers; layer++) {
        auto [k_cpu, v_cpu] = kv_cache->get(layer);

        auto* k_gpu = static_cast<MTL::Buffer*>(gpu_kv_cache_->layers[layer].k_cache);
        auto* v_gpu = static_cast<MTL::Buffer*>(gpu_kv_cache_->layers[layer].v_cache);

        // Map CPU buffers (FP16)
        auto map_k = backend_->map_buffer(k_cpu.buffer());
        auto map_v = backend_->map_buffer(v_cpu.buffer());
        if (!map_k.ok() || !map_v.ok()) {
            GRANITE_FAIL(ErrorCode::InternalError, "Failed to map CPU KV cache");
        }

        const auto* k_fp16 = static_cast<const uint16_t*>(map_k.value());
        const auto* v_fp16 = static_cast<const uint16_t*>(map_v.value());

        // Map GPU buffers (FP16 - same format as CPU cache)
        auto* k_gpu_fp16 = static_cast<uint16_t*>(k_gpu->contents());
        auto* v_gpu_fp16 = static_cast<uint16_t*>(v_gpu->contents());

        // Both CPU and GPU caches are FP16, just copy directly
        // CPU cache shape: [1, num_kv_heads, max_seq_len, head_dim]
        // GPU cache shape: [num_kv_heads, max_seq_len, head_dim]
        int max_seq_cpu = kv_cache->max_seq_len();
        int max_seq_gpu = gpu_kv_cache_->max_seq_len;

        // Optimized path: if max_seq matches, use contiguous memcpy per head
        if (max_seq_cpu == max_seq_gpu) {
            // Contiguous copy per head
            size_t copy_size = cpu_len * head_dim * sizeof(uint16_t);
            for (int h = 0; h < num_kv_heads; h++) {
                size_t offset = h * max_seq_gpu * head_dim;
                std::memcpy(&k_gpu_fp16[offset], &k_fp16[offset], copy_size);
                std::memcpy(&v_gpu_fp16[offset], &v_fp16[offset], copy_size);
            }
        } else {
            // Fallback: element-wise copy when max_seq differs
            for (int h = 0; h < num_kv_heads; h++) {
                for (int s = 0; s < cpu_len; s++) {
                    size_t cpu_base = h * max_seq_cpu * head_dim + s * head_dim;
                    size_t gpu_base = h * max_seq_gpu * head_dim + s * head_dim;
                    std::memcpy(&k_gpu_fp16[gpu_base], &k_fp16[cpu_base], head_dim * sizeof(uint16_t));
                    std::memcpy(&v_gpu_fp16[gpu_base], &v_fp16[cpu_base], head_dim * sizeof(uint16_t));
                }
            }
        }

        backend_->unmap_buffer(k_cpu.buffer());
        backend_->unmap_buffer(v_cpu.buffer());
    }

    gpu_kv_cache_->current_len = cpu_len;
    GRANITE_LOG_DEBUG("Synced CPU KV cache to GPU: {} tokens", cpu_len);

    return {};
}

// Full GPU attention (no CPU sync until final output)
Result<Tensor> TransformerModel::attention_full_gpu(
    const Tensor& hidden,
    int layer,
    int start_pos)
{
    // Debug logging disabled for performance - uncomment to trace attention calls
    // if (layer == 0) {
    //     GRANITE_LOG_DEBUG("ATTN_FULL_GPU enter layer={} sp={}", layer, start_pos);
    // }
    std::string prefix = "blk." + std::to_string(layer) + ".";

    // Get raw quantized weights for GPU projection
    const RawWeight* raw_wq = get_raw_weight(prefix + "attn_q.weight");
    const RawWeight* raw_wk = get_raw_weight(prefix + "attn_k.weight");
    const RawWeight* raw_wv = get_raw_weight(prefix + "attn_v.weight");
    const RawWeight* raw_wo = get_raw_weight(prefix + "attn_output.weight");

    if (!raw_wq || !raw_wk || !raw_wv || !raw_wo) {
        GRANITE_FAIL(ErrorCode::InternalError, "Missing attention weights");
    }

    // Check quantization type
    GGMLType attn_qtype = raw_wq->quant_type;

    auto* gpu = get_metal_compute();
    if (!gpu || !gpu->is_initialized() || !gpu_kv_cache_ || !gpu_kv_cache_->is_allocated()) {
        GRANITE_FAIL(ErrorCode::InternalError, "GPU or KV cache not initialized");
    }

    int hidden_dim = model_config_.hidden_dim;
    int num_heads = model_config_.num_heads;
    int num_kv_heads = model_config_.num_kv_heads;
    int head_dim = model_config_.head_dim;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int max_seq = gpu_kv_cache_->max_seq_len;
    int current_len = gpu_kv_cache_->current_len;
    int total_seq = current_len + 1;  // Including new token

    // Get Metal buffers
    auto* h_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(hidden.buffer()));
    auto* wq_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(raw_wq->buffer));
    auto* wk_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(raw_wk->buffer));
    auto* wv_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(raw_wv->buffer));
    auto* wo_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(raw_wo->buffer));

    // Get KV cache for this layer
    auto& layer_cache = gpu_kv_cache_->layers[layer];
    auto* k_cache_buf = static_cast<MTL::Buffer*>(layer_cache.k_cache);
    auto* v_cache_buf = static_cast<MTL::Buffer*>(layer_cache.v_cache);

    // Use pooled buffers if available, otherwise allocate new ones
    bool use_pool = decode_pool_ && decode_pool_->initialized &&
                    decode_pool_->q_buf && decode_pool_->k_buf &&
                    decode_pool_->v_buf && decode_pool_->attn_out_buf;

    MTL::Buffer* q_buf;
    MTL::Buffer* k_buf;
    MTL::Buffer* v_buf;
    MTL::Buffer* attn_out_buf;

    if (use_pool) {
        q_buf = static_cast<MTL::Buffer*>(decode_pool_->q_buf);
        k_buf = static_cast<MTL::Buffer*>(decode_pool_->k_buf);
        v_buf = static_cast<MTL::Buffer*>(decode_pool_->v_buf);
        attn_out_buf = static_cast<MTL::Buffer*>(decode_pool_->attn_out_buf);
    } else {
        q_buf = gpu->create_buffer(q_dim * sizeof(float));
        k_buf = gpu->create_buffer(kv_dim * sizeof(float));
        v_buf = gpu->create_buffer(kv_dim * sizeof(float));
        attn_out_buf = gpu->create_buffer(q_dim * sizeof(float));
    }

    if (!q_buf || !k_buf || !v_buf || !attn_out_buf) {
        if (!use_pool) {
            if (q_buf) q_buf->release();
            if (k_buf) k_buf->release();
            if (v_buf) v_buf->release();
            if (attn_out_buf) attn_out_buf->release();
        }
        GRANITE_FAIL(ErrorCode::OutOfMemory, "Failed to allocate attention buffers");
    }

    // Use pooled output tensor if available, otherwise allocate
    Tensor output;
    MTL::Buffer* o_buf = nullptr;
    bool use_output_pool = use_pool && decode_pool_->attn_layer_out.buffer().valid();

    if (use_output_pool) {
        output = decode_pool_->attn_layer_out;
        o_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(output.buffer()));
    } else {
        std::vector<int64_t> output_shape = {1, 1, static_cast<int64_t>(hidden_dim)};
        auto output_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
        if (!output_result.ok()) {
            if (!use_pool) {
                q_buf->release();
                k_buf->release();
                v_buf->release();
                attn_out_buf->release();
            }
            return output_result.error();
        }
        output = std::move(output_result).take();
        o_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(output.buffer()));
    }

    // === ALL GPU OPERATIONS - NO SYNC UNTIL END ===

    // Helper lambda to dispatch matvec based on weight's quantization type
    auto dispatch_matvec = [&](MTL::Buffer* in, MTL::Buffer* weight, MTL::Buffer* out,
                               GGMLType qtype, int K, int N) {
        switch (qtype) {
            case GGMLType::Q8_0:  gpu->matvec_q8_0(in, weight, out, K, N); break;
            case GGMLType::Q4_0:  gpu->matvec_q4_0(in, weight, out, K, N); break;
            case GGMLType::IQ4_NL: gpu->matvec_iq4_nl(in, weight, out, K, N); break;
            case GGMLType::IQ4_XS: gpu->matvec_iq4_xs(in, weight, out, K, N); break;
            case GGMLType::IQ3_S: gpu->matvec_iq3_s(in, weight, out, K, N); break;
            case GGMLType::Q6_K:  gpu->matvec_q6_k(in, weight, out, K, N); break;
            case GGMLType::Q5_K:  gpu->matvec_q5_k(in, weight, out, K, N); break;
            case GGMLType::Q4_K:  gpu->matvec_q4k(in, weight, out, K, N); break;
            case GGMLType::Q3_K:  gpu->matvec_q3_k(in, weight, out, K, N); break;
            case GGMLType::Q2_K:  gpu->matvec_q2_k(in, weight, out, K, N); break;
            default: gpu->matvec_q4k(in, weight, out, K, N); break;  // Fallback
        }
    };

    // 1. Q/K/V projections - use each weight's actual quantization type
    dispatch_matvec(h_buf, wq_buf, q_buf, raw_wq->quant_type, hidden_dim, q_dim);
    dispatch_matvec(h_buf, wk_buf, k_buf, raw_wk->quant_type, hidden_dim, kv_dim);
    dispatch_matvec(h_buf, wv_buf, v_buf, raw_wv->quant_type, hidden_dim, kv_dim);

    // 2. Apply RoPE to Q and K
    gpu->rope_multihead(q_buf, k_buf, num_heads, num_kv_heads, 1, head_dim,
                        start_pos, model_config_.rope_theta);

    // 3. Append new K/V to cache
    gpu->kv_cache_append(
        k_cache_buf, v_cache_buf,
        k_buf, v_buf,
        num_kv_heads, head_dim,
        current_len, 1, max_seq
    );

    // 4. Multi-head attention (GPU kernel or CoreML/ANE)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

#ifdef GRANITE_HAS_COREML
    if (use_coreml_) {
        // Use CoreML/MPSGraph attention (can target ANE)
        auto* coreml = get_coreml_attention();
        if (coreml && coreml->is_initialized()) {
            auto coreml_result = coreml->multihead_attention(
                q_buf,
                k_cache_buf,
                v_cache_buf,
                attn_out_buf,
                num_heads,
                num_kv_heads,
                1,          // seq_q = 1 for decode
                total_seq,  // seq_kv = all cached + new
                head_dim,
                scale
            );
            if (!coreml_result.ok()) {
                GRANITE_LOG_WARN("CoreML attention failed, falling back to Metal: {}",
                                coreml_result.error().message());
                // Fall through to Metal path below
                gpu->multihead_attention(
                    q_buf, k_cache_buf, v_cache_buf, attn_out_buf,
                    num_heads, num_kv_heads, 1, total_seq, head_dim, scale
                );
            }
        } else {
            // CoreML not available, use Metal
            gpu->multihead_attention(
                q_buf, k_cache_buf, v_cache_buf, attn_out_buf,
                num_heads, num_kv_heads, 1, total_seq, head_dim, scale
            );
        }
    } else
#endif
    {
        // Standard Metal GPU attention
        gpu->multihead_attention(
            q_buf,
            k_cache_buf,
            v_cache_buf,
            attn_out_buf,
            num_heads,
            num_kv_heads,
            1,          // seq_q = 1 for decode
            total_seq,  // seq_kv = all cached + new
            head_dim,
            scale
        );
    }

    // 5. Output projection - use Wo's actual quantization type
    dispatch_matvec(attn_out_buf, wo_buf, o_buf, raw_wo->quant_type, q_dim, hidden_dim);

    // Only sync at the end - NOT here, let caller sync
    // gpu->sync();  // Removed - caller will sync

    // Update cache length only on last layer
    // No sync needed here - GPU ops are queued and will execute in order.
    // The final sync in get_logits() ensures everything completes before we
    // return from forward(). increment_len just updates a CPU counter.
    if (layer == model_config_.num_layers - 1) {
        gpu_kv_cache_->increment_len(1);
    }

    // Clean up temporary buffers (only if not using pool)
    if (!use_pool) {
        q_buf->release();
        k_buf->release();
        v_buf->release();
        attn_out_buf->release();
    }

    return output;
}

Result<Tensor> TransformerModel::attention_gpu(
    const Tensor& hidden,
    int layer,
    KVCache* kv_cache,
    int start_pos)
{
    int batch = static_cast<int>(hidden.size(0));
    int seq_len = static_cast<int>(hidden.size(1));
    int total_tokens = batch * seq_len;
    bool is_decode = (total_tokens == 1);

    // Check if GPU cache exists and is allocated
    bool has_gpu_cache = gpu_kv_cache_ && gpu_kv_cache_->is_allocated();

    // If decode mode and GPU cache exists
    if (is_decode && has_gpu_cache) {
        int gpu_len = gpu_kv_cache_->seq_len();

        // Debug logging disabled for performance
        // if (layer == 0) {
        //     GRANITE_LOG_DEBUG("DECODE: layer={} start_pos={} gpu_len={} cpu_len={}",
        //         layer, start_pos, gpu_len, kv_cache ? kv_cache->seq_len() : -1);
        // }

        // If GPU cache is already valid (gpu_len == start_pos), use GPU path
        if (gpu_len == start_pos) {
            auto result = attention_full_gpu(hidden, layer, start_pos);
            // After last layer, sync CPU cache length to match GPU cache
            // This keeps forward_single's start_pos calculation correct
            if (result.ok() && layer == model_config_.num_layers - 1 && kv_cache) {
                kv_cache->increment_seq_len(1);
            }
            return result;
        }

        // If GPU cache is behind CPU cache, sync on layer 0
        if (gpu_len < start_pos && kv_cache && kv_cache->seq_len() >= start_pos) {
            if (layer == 0) {
                auto sync_result = sync_cpu_to_gpu_kv_cache(kv_cache);
                if (!sync_result.ok()) {
                    GRANITE_LOG_WARN("Failed to sync CPU->GPU KV cache: {}",
                                     sync_result.error().message());
                    return attention(hidden, layer, kv_cache, start_pos);
                }
            }
            // After sync, GPU cache should be valid
            if (gpu_kv_cache_->seq_len() == start_pos) {
                auto result = attention_full_gpu(hidden, layer, start_pos);
                // After last layer, sync CPU cache length
                if (result.ok() && layer == model_config_.num_layers - 1 && kv_cache) {
                    kv_cache->increment_seq_len(1);
                }
                return result;
            }
        }

        // If GPU cache is ahead, something is wrong - fall back to CPU
        // This shouldn't happen in normal operation
    }

    // GPU PREFILL PATH: For multi-token prefill with GPU cache
    if (!is_decode && has_gpu_cache && start_pos == 0) {
        auto* gpu = get_metal_compute();
        if (gpu && gpu->is_initialized()) {
            std::string prefix = "blk." + std::to_string(layer) + ".";

            // Get raw quantized weights
            const RawWeight* raw_wq = get_raw_weight(prefix + "attn_q.weight");
            const RawWeight* raw_wk = get_raw_weight(prefix + "attn_k.weight");
            const RawWeight* raw_wv = get_raw_weight(prefix + "attn_v.weight");
            const RawWeight* raw_wo = get_raw_weight(prefix + "attn_output.weight");

            // Debug logging disabled for performance
            // if (layer == 0 && raw_wk && raw_wv) {
            //     GRANITE_LOG_DEBUG("WEIGHTS: wk_type={} wv_type={} wk_size={} wv_size={}",
            //         static_cast<int>(raw_wk->quant_type), static_cast<int>(raw_wv->quant_type),
            //         raw_wk->size_bytes, raw_wv->size_bytes);
            // }

            if (raw_wq && raw_wk && raw_wv && raw_wo &&
                (raw_wq->quant_type == GGMLType::Q4_K || raw_wq->quant_type == GGMLType::Q5_K ||
                 raw_wq->quant_type == GGMLType::Q6_K || raw_wq->quant_type == GGMLType::Q3_K ||
                 raw_wq->quant_type == GGMLType::Q2_K ||
                 raw_wq->quant_type == GGMLType::Q8_0 || raw_wq->quant_type == GGMLType::Q4_0 ||
                 raw_wq->quant_type == GGMLType::IQ4_NL || raw_wq->quant_type == GGMLType::IQ4_XS ||
                 raw_wq->quant_type == GGMLType::IQ3_S)) {

                GGMLType prefill_qtype = raw_wq->quant_type;
                int hidden_dim = model_config_.hidden_dim;
                int num_heads = model_config_.num_heads;
                int num_kv_heads = model_config_.num_kv_heads;
                int head_dim = model_config_.head_dim;
                int q_dim = num_heads * head_dim;
                int kv_dim = num_kv_heads * head_dim;
                int max_seq = gpu_kv_cache_->max_seq_len;

                // Get Metal buffers
                auto* h_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(hidden.buffer()));
                auto* wq_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(raw_wq->buffer));
                auto* wk_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(raw_wk->buffer));
                auto* wv_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(raw_wv->buffer));
                auto* wo_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(raw_wo->buffer));

                // Get KV cache for this layer
                auto& layer_cache = gpu_kv_cache_->layers[layer];
                auto* k_cache_buf = static_cast<MTL::Buffer*>(layer_cache.k_cache);
                auto* v_cache_buf = static_cast<MTL::Buffer*>(layer_cache.v_cache);

                // Use prefill buffer pool (allocated once, reused across layers)
                auto pool_result = ensure_prefill_pool(total_tokens);
                if (!pool_result.ok()) {
                    return pool_result.error();
                }

                MTL::Buffer* q_buf = static_cast<MTL::Buffer*>(prefill_pool_->q_buf);
                MTL::Buffer* k_buf = static_cast<MTL::Buffer*>(prefill_pool_->k_buf);
                MTL::Buffer* v_buf = static_cast<MTL::Buffer*>(prefill_pool_->v_buf);
                MTL::Buffer* attn_out_buf = static_cast<MTL::Buffer*>(prefill_pool_->attn_out_buf);

                if (q_buf && k_buf && v_buf && attn_out_buf) {
                    // Helper lambda to dispatch matmul based on weight's quantization type
                    auto dispatch_matmul = [&](MTL::Buffer* in, MTL::Buffer* weight, MTL::Buffer* out,
                                               GGMLType qtype, int M, int K, int N) {
                        switch (qtype) {
                            case GGMLType::Q8_0:  gpu->matmul_q8_0(in, weight, out, M, K, N); break;
                            case GGMLType::Q4_0:  gpu->matmul_q4_0(in, weight, out, M, K, N); break;
                            case GGMLType::IQ4_NL: gpu->matmul_iq4_nl(in, weight, out, M, K, N); break;
                            case GGMLType::IQ4_XS: gpu->matmul_iq4_xs(in, weight, out, M, K, N); break;
                            case GGMLType::IQ3_S: gpu->matmul_iq3_s(in, weight, out, M, K, N); break;
                            case GGMLType::Q6_K:  gpu->matmul_q6_k(in, weight, out, M, K, N); break;
                            case GGMLType::Q5_K:  gpu->matmul_q5_k(in, weight, out, M, K, N); break;
                            case GGMLType::Q4_K:  gpu->matmul_q4k(in, weight, out, M, K, N); break;
                            case GGMLType::Q3_K:  gpu->matmul_q3_k(in, weight, out, M, K, N); break;
                            case GGMLType::Q2_K:  gpu->matmul_q2_k(in, weight, out, M, K, N); break;
                            default: gpu->matmul_q4k(in, weight, out, M, K, N); break;  // Fallback
                        }
                    };

                    // 1. Q/K/V projections - use each weight's actual quantization type
                    dispatch_matmul(h_buf, wq_buf, q_buf, raw_wq->quant_type, total_tokens, hidden_dim, q_dim);
                    dispatch_matmul(h_buf, wk_buf, k_buf, raw_wk->quant_type, total_tokens, hidden_dim, kv_dim);
                    dispatch_matmul(h_buf, wv_buf, v_buf, raw_wv->quant_type, total_tokens, hidden_dim, kv_dim);

                    // 2. Apply RoPE to Q and K (batched)
                    gpu->rope_multihead(q_buf, k_buf, num_heads, num_kv_heads, total_tokens, head_dim,
                                        start_pos, model_config_.rope_theta);

                    // 3. Append K/V to FP16 cache (kv_cache_append uses FP16 by default)
                    gpu->kv_cache_append(
                        k_cache_buf, v_cache_buf,
                        k_buf, v_buf,
                        num_kv_heads, head_dim,
                        0,  // current_len = 0 for prefill start
                        total_tokens, max_seq
                    );

                    // 4. Multi-head attention (prefill kernel handles causal mask)
                    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                    gpu->multihead_attention(
                        q_buf,
                        k_cache_buf,
                        v_cache_buf,
                        attn_out_buf,
                        num_heads,
                        num_kv_heads,
                        total_tokens,  // seq_q
                        total_tokens,  // seq_kv (same as seq_q for prefill)
                        head_dim,
                        scale
                    );

                    // 5. Output projection - use Wo's actual quantization type
                    std::vector<int64_t> output_shape = {batch, seq_len, static_cast<int64_t>(hidden_dim)};
                    auto output_result = Tensor::allocate(output_shape, DataType::FP32, backend_);
                    if (output_result.ok()) {
                        auto output = std::move(output_result).take();
                        auto* o_buf = static_cast<MTL::Buffer*>(backend_->get_native_buffer(output.buffer()));

                        dispatch_matmul(attn_out_buf, wo_buf, o_buf, raw_wo->quant_type, total_tokens, q_dim, hidden_dim);

                        // Update GPU cache length on last layer
                        if (layer == model_config_.num_layers - 1) {
                            gpu_kv_cache_->current_len = total_tokens;
                            if (kv_cache) {
                                kv_cache->set_seq_len(total_tokens);
                            }
                        }

                        // NOTE: Don't sync here - let operations batch across layers
                        // Sync happens in forward() at the end of all layers

                        // Buffers are from pool - don't release

                        return output;
                    }
                }

                // Buffers are from pool - don't release on failure either
            }
        }
    }

    // Fall back to CPU attention for prefill or when GPU cache is not valid
    return attention(hidden, layer, kv_cache, start_pos);
}

#else
// Non-Metal stubs
Result<Tensor> TransformerModel::feed_forward_gpu(const Tensor& hidden, int layer) {
    return feed_forward(hidden, layer);
}

Result<Tensor> TransformerModel::attention_full_gpu(
    const Tensor& hidden,
    int layer,
    int start_pos) {
    return attention(hidden, layer, nullptr, start_pos);
}

Result<Tensor> TransformerModel::attention_gpu(
    const Tensor& hidden,
    int layer,
    KVCache* kv_cache,
    int start_pos)
{
    return attention(hidden, layer, kv_cache, start_pos);
}
#endif

}  // namespace granite
