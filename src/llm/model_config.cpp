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

    // Compute derived values
    config.compute_derived();

    GRANITE_LOG_INFO("Model config: arch={}, layers={}, hidden={}, heads={}/{}, vocab={}",
                     config.architecture, config.num_layers, config.hidden_dim,
                     config.num_heads, config.num_kv_heads, config.vocab_size);

    return config;
}

}  // namespace granite
