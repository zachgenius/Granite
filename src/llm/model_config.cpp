#include "llm_internal.h"

namespace granite {

Result<ModelConfig> parse_model_config(const GGUFFile& gguf) {
    ModelConfig config;

    // Get architecture
    auto arch = gguf.get_architecture();
    if (!arch) {
        GRANITE_FAIL(ErrorCode::InvalidFormat, "No architecture in GGUF");
    }
    config.architecture = *arch;

    // Get model name
    if (auto name = gguf.get_metadata_as<std::string>("general.name")) {
        config.name = *name;
    }

    // Architecture-specific parameters
    std::string prefix = config.architecture + ".";

    if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "context_length")) {
        config.max_seq_len = static_cast<int>(*v);
    }
    if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "embedding_length")) {
        config.hidden_dim = static_cast<int>(*v);
    }
    if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "feed_forward_length")) {
        config.intermediate_dim = static_cast<int>(*v);
    }
    if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "block_count")) {
        config.num_layers = static_cast<int>(*v);
    }
    if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "attention.head_count")) {
        config.num_heads = static_cast<int>(*v);
    }
    if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "attention.head_count_kv")) {
        config.num_kv_heads = static_cast<int>(*v);
    } else {
        config.num_kv_heads = config.num_heads;  // Default to MHA
    }

    // RoPE parameters
    if (auto v = gguf.get_metadata_as<float>(prefix + "rope.freq_base")) {
        config.rope_theta = *v;
    }
    if (auto v = gguf.get_metadata_as<float>(prefix + "attention.layer_norm_rms_epsilon")) {
        config.rms_norm_eps = *v;
    }

    // Vocab size from embedding tensor
    if (auto* emb = gguf.find_tensor("token_embd.weight")) {
        config.vocab_size = static_cast<int>(std::max(emb->dimensions[0], emb->dimensions[1]));
    }

    // Try to get head_dim from metadata first (some models like Gemma have head_dim != hidden_dim / num_heads)
    if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "attention.key_length")) {
        config.head_dim = static_cast<int>(*v);
    } else if (auto v = gguf.get_metadata_as<uint32_t>(prefix + "attention.value_length")) {
        config.head_dim = static_cast<int>(*v);
    }

    // If head_dim still not set, compute from Q weight shape
    // Q weight shape is [q_output_dim, hidden_dim] where q_output_dim = num_heads * head_dim
    if (config.head_dim == 0) {
        if (auto* q_weight = gguf.find_tensor("blk.0.attn_q.weight")) {
            // GGUF stores dimensions in reverse order (innermost first)
            // For [4096, 5376], dimensions[0]=5376 (K), dimensions[1]=4096 (N)
            int q_output_dim = static_cast<int>(q_weight->dimensions[1]);
            if (config.num_heads > 0 && q_output_dim > 0) {
                config.head_dim = q_output_dim / config.num_heads;
            }
        }
    }

    // Fallback: compute derived values (head_dim = hidden_dim / num_heads)
    config.compute_derived();

    GRANITE_LOG_INFO("Model config: arch={}, layers={}, hidden={}, heads={}/{}, head_dim={}, vocab={}",
                     config.architecture, config.num_layers, config.hidden_dim,
                     config.num_heads, config.num_kv_heads, config.head_dim, config.vocab_size);

    return config;
}

}  // namespace granite
