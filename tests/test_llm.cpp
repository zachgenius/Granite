#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <granite/granite.h>

#include <cstring>
#include <cmath>

using namespace granite;

// =============================================================================
// RoPE Cache Tests
// =============================================================================

TEST_CASE("RoPECache initialization", "[llm]") {
    RoPECache cache;

    REQUIRE(!cache.is_initialized());

    cache.initialize(2048, 64, 10000.0f);

    REQUIRE(cache.is_initialized());
    REQUIRE(cache.cos_cache().size() == 2048 * 32);  // max_seq * (head_dim / 2)
    REQUIRE(cache.sin_cache().size() == 2048 * 32);
}

TEST_CASE("RoPECache values at position 0", "[llm]") {
    RoPECache cache;
    cache.initialize(1024, 8, 10000.0f);

    const auto& cos = cache.cos_cache();
    const auto& sin = cache.sin_cache();

    // At position 0, all angles are 0
    // cos(0) = 1, sin(0) = 0
    for (int i = 0; i < 4; i++) {  // head_dim / 2 = 4
        REQUIRE_THAT(cos[i], Catch::Matchers::WithinAbs(1.0f, 1e-5f));
        REQUIRE_THAT(sin[i], Catch::Matchers::WithinAbs(0.0f, 1e-5f));
    }
}

TEST_CASE("RoPECache values at position 1", "[llm]") {
    RoPECache cache;
    int head_dim = 8;
    float theta = 10000.0f;
    cache.initialize(1024, head_dim, theta);

    const auto& cos = cache.cos_cache();
    const auto& sin = cache.sin_cache();

    int half_dim = head_dim / 2;

    // At position 1, angles are based on frequency
    for (int i = 0; i < half_dim; i++) {
        float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim);
        float angle = 1.0f * freq;  // position 1

        float expected_cos = std::cos(angle);
        float expected_sin = std::sin(angle);

        REQUIRE_THAT(cos[1 * half_dim + i], Catch::Matchers::WithinAbs(expected_cos, 1e-5f));
        REQUIRE_THAT(sin[1 * half_dim + i], Catch::Matchers::WithinAbs(expected_sin, 1e-5f));
    }
}

// =============================================================================
// ModelConfig Tests
// =============================================================================

TEST_CASE("ModelConfig compute_derived", "[llm]") {
    ModelConfig config;
    config.hidden_dim = 4096;
    config.num_heads = 32;
    config.head_dim = 0;  // Not set

    config.compute_derived();

    REQUIRE(config.head_dim == 128);  // 4096 / 32
}

TEST_CASE("ModelConfig defaults", "[llm]") {
    ModelConfig config;

    REQUIRE(config.vocab_size == 0);
    REQUIRE(config.hidden_dim == 0);
    REQUIRE(config.num_layers == 0);
    REQUIRE(config.num_heads == 0);
    REQUIRE(config.max_seq_len == 2048);
    REQUIRE(config.rope_theta == 10000.0f);
    REQUIRE(config.rms_norm_eps == 1e-5f);
    REQUIRE(config.tie_word_embeddings == false);
}

// =============================================================================
// GenerationConfig Tests
// =============================================================================

TEST_CASE("GenerationConfig defaults", "[llm]") {
    GenerationConfig config;

    REQUIRE_THAT(config.temperature, Catch::Matchers::WithinAbs(0.7f, 1e-5f));
    REQUIRE(config.top_k == 40);
    REQUIRE_THAT(config.top_p, Catch::Matchers::WithinAbs(0.9f, 1e-5f));
    REQUIRE_THAT(config.repetition_penalty, Catch::Matchers::WithinAbs(1.0f, 1e-5f));
    REQUIRE(config.max_tokens == 256);
    REQUIRE(config.do_sample == true);
    REQUIRE(config.stop_tokens.empty());
}

// =============================================================================
// KVCache Tests
// =============================================================================

TEST_CASE("KVCache allocation", "[llm]") {
    auto backend = create_default_backend();
    REQUIRE(backend != nullptr);

    auto init_result = backend->initialize();
    REQUIRE(init_result.ok());

    ModelConfig config;
    config.num_layers = 2;
    config.num_kv_heads = 4;
    config.head_dim = 64;
    config.num_heads = 8;

    auto cache_result = KVCache::allocate(config, 512, backend.get());
    REQUIRE(cache_result.ok());

    auto cache = std::move(cache_result).take();

    REQUIRE(cache.is_allocated());
    REQUIRE(cache.max_seq_len() == 512);
    REQUIRE(cache.seq_len() == 0);

    // Memory should be: 2 layers * 2 (K+V) * 4 heads * 512 seq * 64 dim * 2 bytes
    size_t expected = 2 * 2 * 4 * 512 * 64 * 2;
    REQUIRE(cache.memory_bytes() == expected);
}

TEST_CASE("KVCache clear", "[llm]") {
    auto backend = create_default_backend();
    REQUIRE(backend != nullptr);

    auto init_result = backend->initialize();
    REQUIRE(init_result.ok());

    ModelConfig config;
    config.num_layers = 1;
    config.num_kv_heads = 2;
    config.head_dim = 32;
    config.num_heads = 2;

    auto cache_result = KVCache::allocate(config, 128, backend.get());
    REQUIRE(cache_result.ok());

    auto cache = std::move(cache_result).take();

    // Clear should reset seq_len to 0
    cache.clear();
    REQUIRE(cache.seq_len() == 0);
}

// =============================================================================
// Tokenizer Tests (without GGUF file)
// =============================================================================

TEST_CASE("Tokenizer not loaded by default", "[llm]") {
    Tokenizer tok;

    REQUIRE(!tok.is_loaded());
    REQUIRE(tok.vocab_size() == 0);
}

TEST_CASE("Tokenizer special token defaults", "[llm]") {
    Tokenizer tok;

    // Default special tokens
    REQUIRE(tok.bos_token() == 1);
    REQUIRE(tok.eos_token() == 2);
    REQUIRE(tok.pad_token() == 0);
}
