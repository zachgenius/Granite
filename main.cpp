#include <granite/granite.h>
#include <iostream>
#include <string>
#include <chrono>

int main(int argc, char* argv[]) {
    granite::init_logging(spdlog::level::info);

    GRANITE_LOG_INFO("Granite Embedded Inference Framework v{}", granite::version_string());

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>\n";
        std::cerr << "Example: " << argv[0] << " tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf\n";
        return 1;
    }

    std::string model_path = argv[1];
    GRANITE_LOG_INFO("Loading model: {}", model_path);

    // Test GGUF parsing
    GRANITE_LOG_INFO("Step 1: Testing GGUF parsing...");
    auto gguf_result = granite::GGUFFile::open(model_path);
    if (!gguf_result.ok()) {
        GRANITE_LOG_ERROR("Failed to open GGUF: {}", gguf_result.error().message());
        return 1;
    }

    auto& gguf = gguf_result.value();
    GRANITE_LOG_INFO("GGUF file opened: {} tensors, {:.1f} MB",
                     gguf.tensor_count(), gguf.file_size() / (1024.0 * 1024.0));

    // Parse model config
    auto config_result = granite::parse_model_config(gguf);
    if (!config_result.ok()) {
        GRANITE_LOG_ERROR("Failed to parse config: {}", config_result.error().message());
        return 1;
    }
    auto& config = config_result.value();

    // Load tokenizer
    GRANITE_LOG_INFO("Step 2: Loading tokenizer...");
    auto tok_result = granite::Tokenizer::from_gguf(gguf);
    if (!tok_result.ok()) {
        GRANITE_LOG_ERROR("Failed to load tokenizer: {}", tok_result.error().message());
        return 1;
    }
    auto& tokenizer = tok_result.value();
    GRANITE_LOG_INFO("Tokenizer loaded: {} tokens", tokenizer.vocab_size());

    // Test tokenization round-trip
    std::string test_text = "Hello, world!";
    auto tokens = tokenizer.encode(test_text, false);
    std::string decoded = tokenizer.decode(tokens);
    GRANITE_LOG_INFO("Tokenization test: \"{}\" -> {} tokens -> \"{}\"",
                     test_text, tokens.size(), decoded);

    // Create backend and load weights
    GRANITE_LOG_INFO("Step 3: Creating backend and loading weights...");
    auto backend = granite::create_default_backend();
    if (!backend) {
        GRANITE_LOG_ERROR("Failed to create backend");
        return 1;
    }

    auto init_result = backend->initialize();
    if (!init_result.ok()) {
        GRANITE_LOG_ERROR("Failed to initialize backend: {}", init_result.error().message());
        return 1;
    }

    // Load weights
    granite::ModelLoader loader(backend.get());
    auto start_time = std::chrono::high_resolution_clock::now();

    auto weights_result = loader.load_weights(gguf);
    if (!weights_result.ok()) {
        GRANITE_LOG_ERROR("Failed to load weights: {}", weights_result.error().message());
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    auto& weights = weights_result.value();

    GRANITE_LOG_INFO("Loaded {} weight tensors in {} ms", weights.size(), duration.count());

    // Print some weight info
    size_t total_bytes = 0;
    for (const auto& [name, tensor] : weights) {
        total_bytes += tensor.size_bytes();
    }
    GRANITE_LOG_INFO("Total weights size: {:.1f} MB", total_bytes / (1024.0 * 1024.0));

    // Check for key weights
    std::vector<std::string> key_weights = {
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight"
    };

    GRANITE_LOG_INFO("Checking key weights:");
    for (const auto& name : key_weights) {
        if (weights.count(name)) {
            const auto& t = weights.at(name);
            std::string shape_str = "[";
            for (size_t i = 0; i < t.shape().size(); i++) {
                if (i > 0) shape_str += ", ";
                shape_str += std::to_string(t.shape()[i]);
            }
            shape_str += "]";
            GRANITE_LOG_INFO("  {}: {} {}", name,
                           granite::dtype_name(t.dtype()), shape_str);
        } else {
            GRANITE_LOG_WARN("  {}: MISSING", name);
        }
    }

    // Allocate KV cache
    GRANITE_LOG_INFO("Step 4: Allocating KV cache...");
    auto cache_result = granite::KVCache::allocate(config, 512, backend.get());
    if (!cache_result.ok()) {
        GRANITE_LOG_ERROR("Failed to allocate KV cache: {}", cache_result.error().message());
        return 1;
    }
    auto& kv_cache = cache_result.value();
    GRANITE_LOG_INFO("KV cache allocated: {:.1f} MB", kv_cache.memory_bytes() / (1024.0 * 1024.0));

    GRANITE_LOG_INFO("Model loading test completed successfully!");
    GRANITE_LOG_INFO("");
    GRANITE_LOG_INFO("Model summary:");
    GRANITE_LOG_INFO("  Architecture: {}", config.architecture);
    GRANITE_LOG_INFO("  Parameters: ~1.1B");
    GRANITE_LOG_INFO("  Layers: {}", config.num_layers);
    GRANITE_LOG_INFO("  Hidden: {}", config.hidden_dim);
    GRANITE_LOG_INFO("  Heads: {} (KV: {})", config.num_heads, config.num_kv_heads);
    GRANITE_LOG_INFO("  Vocab: {}", config.vocab_size);
    GRANITE_LOG_INFO("  Context: {}", config.max_seq_len);

    backend->shutdown();
    return 0;
}
