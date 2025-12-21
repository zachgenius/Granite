// SpeculativeRunner - Speculative decoding with draft and target models
//
// Speculative decoding workflow:
// 1. Draft model generates K candidate tokens autoregressively
// 2. Target model verifies all K candidates in a single batch forward
// 3. Accept tokens until mismatch (greedy verification)
// 4. Adjust K based on acceptance rate (adaptive speculation)
// 5. Rollback KV caches to accepted position and continue

#include "llm_internal.h"

namespace granite {

// Thread-local RNG for sampling (shared with runner.cpp)
extern thread_local std::mt19937 g_rng;

Result<std::unique_ptr<SpeculativeRunner>> SpeculativeRunner::load(
    const std::string& target_path,
    const std::string& draft_path)
{
    return load(target_path, draft_path, Config::Balanced());
}

Result<std::unique_ptr<SpeculativeRunner>> SpeculativeRunner::load(
    const std::string& target_path,
    const std::string& draft_path,
    const Config& config)
{
    auto runner = std::make_unique<SpeculativeRunner>();

    // Create backends for both models
    runner->target_backend_ = create_default_backend();
    runner->draft_backend_ = create_default_backend();

    if (!runner->target_backend_ || !runner->draft_backend_) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to create backends");
    }

    auto target_init = runner->target_backend_->initialize();
    if (!target_init.ok()) {
        return target_init.error();
    }

    auto draft_init = runner->draft_backend_->initialize();
    if (!draft_init.ok()) {
        return draft_init.error();
    }

    // Load target model
    GRANITE_LOG_INFO("Loading target model: {}", target_path);
    auto target_result = TransformerModel::load(target_path, runner->target_backend_.get(), config);
    if (!target_result.ok()) {
        return target_result.error();
    }
    runner->target_model_ = std::move(target_result).take();

    // Load draft model
    GRANITE_LOG_INFO("Loading draft model: {}", draft_path);
    auto draft_result = TransformerModel::load(draft_path, runner->draft_backend_.get(), config);
    if (!draft_result.ok()) {
        return draft_result.error();
    }
    runner->draft_model_ = std::move(draft_result).take();

    // Load tokenizer from target model (assume same vocabulary)
    auto gguf_result = GGUFFile::open(target_path);
    if (!gguf_result.ok()) {
        return gguf_result.error();
    }

    auto tok_result = Tokenizer::from_gguf(gguf_result.value());
    if (!tok_result.ok()) {
        GRANITE_LOG_WARN("Failed to load tokenizer: {}", tok_result.error().message());
    } else {
        runner->tokenizer_ = std::move(tok_result).take();
    }

    // Allocate KV caches for both models
    auto target_kv_result = KVCache::allocate(
        runner->target_model_.config(),
        runner->target_model_.config().max_seq_len,
        runner->target_backend_.get());
    if (!target_kv_result.ok()) {
        return target_kv_result.error();
    }
    runner->target_kv_cache_ = std::move(target_kv_result).take();

    auto draft_kv_result = KVCache::allocate(
        runner->draft_model_.config(),
        runner->draft_model_.config().max_seq_len,
        runner->draft_backend_.get());
    if (!draft_kv_result.ok()) {
        return draft_kv_result.error();
    }
    runner->draft_kv_cache_ = std::move(draft_kv_result).take();

    GRANITE_LOG_INFO("Speculative runner initialized:");
    GRANITE_LOG_INFO("  Target: {} layers, {} hidden",
                     runner->target_model_.config().num_layers,
                     runner->target_model_.config().hidden_dim);
    GRANITE_LOG_INFO("  Draft:  {} layers, {} hidden",
                     runner->draft_model_.config().num_layers,
                     runner->draft_model_.config().hidden_dim);

    return runner;
}

Result<std::string> SpeculativeRunner::generate(
    const std::string& prompt,
    const GenerationConfig& config,
    const SpeculativeConfig& spec_config)
{
    std::string result;
    auto status = generate_streaming(prompt, config, [&](const std::string& token) {
        result += token;
        return true;
    }, nullptr, spec_config);

    if (!status.ok()) {
        return status.error();
    }
    return result;
}

Result<void> SpeculativeRunner::generate_streaming(
    const std::string& prompt,
    const GenerationConfig& config,
    TokenCallback callback,
    ProgressCallback progress_callback,
    const SpeculativeConfig& spec_config)
{
    cancelled_ = false;
    stats_ = Stats{};  // Reset stats

    if (!tokenizer_.is_loaded()) {
        GRANITE_FAIL(ErrorCode::InvalidState, "Tokenizer not loaded");
    }

    if (cancelled_) {
        return {};
    }

    // Tokenize prompt
    auto prompt_tokens = tokenizer_.encode(prompt, true);

    // Clear KV caches for new generation
    target_kv_cache_.clear();
    draft_kv_cache_.clear();

    // Create token tensor for prefill
    std::vector<int64_t> prompt_shape = {1, static_cast<int64_t>(prompt_tokens.size())};

    auto target_ids_result = Tensor::allocate(prompt_shape, DataType::INT32, target_backend_.get());
    auto draft_ids_result = Tensor::allocate(prompt_shape, DataType::INT32, draft_backend_.get());

    if (!target_ids_result.ok() || !draft_ids_result.ok()) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to allocate prompt tensors");
    }

    auto target_ids = std::move(target_ids_result).take();
    auto draft_ids = std::move(draft_ids_result).take();

    // Copy tokens to both tensors
    auto map_target = target_backend_->map_buffer(target_ids.buffer());
    auto map_draft = draft_backend_->map_buffer(draft_ids.buffer());

    if (map_target.ok() && map_draft.ok()) {
        auto* target_ptr = static_cast<int32_t*>(map_target.value());
        auto* draft_ptr = static_cast<int32_t*>(map_draft.value());
        std::copy(prompt_tokens.begin(), prompt_tokens.end(), target_ptr);
        std::copy(prompt_tokens.begin(), prompt_tokens.end(), draft_ptr);
        target_backend_->unmap_buffer(target_ids.buffer());
        draft_backend_->unmap_buffer(draft_ids.buffer());
    }

    // Prefill both models
    auto target_logits_result = target_model_.forward(target_ids, &target_kv_cache_, 0);
    if (!target_logits_result.ok()) {
        return target_logits_result.error();
    }
    auto target_logits = std::move(target_logits_result).take();

    auto draft_logits_result = draft_model_.forward(draft_ids, &draft_kv_cache_, 0);
    if (!draft_logits_result.ok()) {
        return draft_logits_result.error();
    }
    auto draft_logits = std::move(draft_logits_result).take();

    if (progress_callback) {
        GenerationProgress progress;
        progress.prompt_tokens = static_cast<int>(prompt_tokens.size());
        progress.generated_tokens = 0;
        progress.max_tokens = config.max_tokens;
        progress.is_prefill = true;
        progress_callback(progress);
    }

    // Sample first token from target model
    int32_t last_token = argmax(target_logits);

    // Check for EOS
    if (last_token == tokenizer_.eos_token()) {
        return {};
    }

    // Output first token
    std::string token_str = tokenizer_.decode_token(last_token);
    if (!callback(token_str)) {
        return {};
    }
    if (progress_callback) {
        GenerationProgress progress;
        progress.prompt_tokens = static_cast<int>(prompt_tokens.size());
        progress.generated_tokens = 1;
        progress.max_tokens = config.max_tokens;
        progress.is_prefill = false;
        progress_callback(progress);
    }

    // Speculative decoding loop
    int k = spec_config.initial_k;
    float acceptance_ema = spec_config.target_acceptance;
    int generated = 1;

    while (generated < config.max_tokens && !cancelled_) {
        // 1. Draft generates K candidate tokens
        auto candidates = draft_tokens(k, last_token);
        stats_.total_draft_tokens += static_cast<int>(candidates.size());

        // 2. Target verifies all candidates in batch
        auto accepted = verify_tokens(candidates, last_token);
        stats_.total_target_forwards++;
        stats_.total_accepted_tokens += static_cast<int>(accepted.size());

        // 3. Output accepted tokens
        bool should_stop = false;
        for (size_t i = 0; i < accepted.size(); i++) {
            int32_t token = accepted[i];

            // Check for EOS
            if (token == tokenizer_.eos_token()) {
                should_stop = true;
                break;
            }

            // Check stop tokens
            if (std::find(config.stop_tokens.begin(), config.stop_tokens.end(),
                          token) != config.stop_tokens.end()) {
                should_stop = true;
                break;
            }

            token_str = tokenizer_.decode_token(token);
            if (!callback(token_str)) {
                should_stop = true;
                break;
            }

            generated++;
            if (progress_callback) {
                GenerationProgress progress;
                progress.prompt_tokens = static_cast<int>(prompt_tokens.size());
                progress.generated_tokens = generated;
                progress.max_tokens = config.max_tokens;
                progress.is_prefill = false;
                progress_callback(progress);
            }
            if (generated >= config.max_tokens) {
                should_stop = true;
                break;
            }
        }

        if (should_stop || accepted.empty()) {
            break;
        }

        // Update last token
        last_token = accepted.back();

        // 4. Sync KV caches to accepted position
        sync_kv_caches(static_cast<int>(accepted.size()), static_cast<int>(candidates.size()));

        // 5. Adapt K based on acceptance rate
        float rate = candidates.empty() ? 0.0f :
            static_cast<float>(accepted.size()) / static_cast<float>(candidates.size());
        acceptance_ema = 0.9f * acceptance_ema + 0.1f * rate;
        k = adapt_k(k, acceptance_ema, spec_config);
    }

    GRANITE_LOG_INFO("Speculative decoding stats: {} drafted, {} accepted ({:.1f}%), {} target forwards",
                     stats_.total_draft_tokens, stats_.total_accepted_tokens,
                     stats_.acceptance_rate() * 100.0f, stats_.total_target_forwards);

    return {};
}

std::vector<int32_t> SpeculativeRunner::draft_tokens(int k, int32_t last_token) {
    std::vector<int32_t> candidates;
    candidates.reserve(k);

    int32_t token = last_token;
    int draft_start_pos = draft_kv_cache_.seq_len();

    for (int i = 0; i < k; i++) {
        // Generate next token from draft model
        auto logits_result = draft_model_.forward_single(token, draft_kv_cache_);
        if (!logits_result.ok()) {
            break;
        }

        // Greedy sampling from draft
        int32_t next = argmax(logits_result.value());
        candidates.push_back(next);
        token = next;

        // Early stop on EOS
        if (next == tokenizer_.eos_token()) {
            break;
        }
    }

    return candidates;
}

std::vector<int32_t> SpeculativeRunner::verify_tokens(
    const std::vector<int32_t>& candidates,
    int32_t last_accepted_token)
{
    if (candidates.empty()) {
        return {};
    }

    // First, add the last accepted token to target KV cache
    auto first_logits_result = target_model_.forward_single(last_accepted_token, target_kv_cache_);
    if (!first_logits_result.ok()) {
        return {};
    }

    // For verification: we need to check if target would have generated each candidate
    // Process candidates through target and check each position
    std::vector<int32_t> accepted;

    // Get target logits for all candidate positions in one batch
    int target_start_pos = target_kv_cache_.seq_len();
    auto batch_logits_result = target_model_.forward_batch(candidates, &target_kv_cache_, target_start_pos);
    if (!batch_logits_result.ok()) {
        // Fallback: at least return target's choice for first position
        int32_t target_choice = argmax(first_logits_result.value());
        accepted.push_back(target_choice);
        return accepted;
    }

    auto batch_logits = std::move(batch_logits_result).take();

    // Verify each candidate against target's preference
    // batch_logits shape: [1, k, vocab_size]
    auto map_result = target_backend_->map_buffer(batch_logits.buffer());
    if (!map_result.ok()) {
        return accepted;
    }

    const float* logits_data = static_cast<const float*>(map_result.value());
    int vocab_size = static_cast<int>(batch_logits.size(2));
    int num_candidates = static_cast<int>(candidates.size());

    // First verify against the logits from first_logits_result (position before candidates)
    int32_t first_target_choice = argmax(first_logits_result.value());
    if (candidates[0] == first_target_choice) {
        accepted.push_back(candidates[0]);
    } else {
        // Mismatch at first position - accept target's choice instead
        accepted.push_back(first_target_choice);
        target_backend_->unmap_buffer(batch_logits.buffer());
        return accepted;
    }

    // Now verify remaining candidates using batch_logits
    // batch_logits[i] gives logits for predicting candidates[i+1] (shifted by 1)
    for (int i = 0; i < num_candidates - 1; i++) {
        const float* pos_logits = logits_data + i * vocab_size;

        // Find argmax for this position
        int32_t target_choice = 0;
        float max_val = pos_logits[0];
        for (int v = 1; v < vocab_size; v++) {
            if (pos_logits[v] > max_val) {
                max_val = pos_logits[v];
                target_choice = v;
            }
        }

        if (candidates[i + 1] == target_choice) {
            // Match - accept draft's token
            accepted.push_back(candidates[i + 1]);
        } else {
            // Mismatch - accept target's choice and stop
            accepted.push_back(target_choice);
            break;
        }
    }

    // If all candidates matched, we need one more token from target
    // (the prediction for position after last candidate)
    if (accepted.size() == candidates.size()) {
        const float* last_logits = logits_data + (num_candidates - 1) * vocab_size;
        int32_t bonus_token = 0;
        float max_val = last_logits[0];
        for (int v = 1; v < vocab_size; v++) {
            if (last_logits[v] > max_val) {
                max_val = last_logits[v];
                bonus_token = v;
            }
        }
        accepted.push_back(bonus_token);
    }

    target_backend_->unmap_buffer(batch_logits.buffer());
    return accepted;
}

void SpeculativeRunner::sync_kv_caches(int accepted_count, int drafted_count) {
    // Target KV cache: we added 1 (for last_accepted_token) + drafted_count tokens
    // We want to keep only accepted_count tokens from verification
    // The verify_tokens function already added all candidates to target cache
    // We need to roll back to: original_len + accepted_count

    // Actually, looking at verify_tokens:
    // - forward_single adds 1 token (last_accepted_token)
    // - forward_batch adds drafted_count tokens
    // Total added: 1 + drafted_count
    // We want to keep: accepted_count (which includes the bonus token if all accepted)

    int target_original_len = target_kv_cache_.seq_len() - 1 - drafted_count;
    int target_new_len = target_original_len + accepted_count;
    target_kv_cache_.truncate(target_new_len);

    // Draft KV cache: we added drafted_count tokens in draft_tokens()
    // We need to truncate to match target
    int draft_original_len = draft_kv_cache_.seq_len() - drafted_count;
    int draft_new_len = draft_original_len + (accepted_count > drafted_count ? drafted_count : accepted_count - 1);
    // If we accepted more than drafted (bonus token), draft cache should be at drafted position
    // If we accepted less, draft cache should be at accepted position minus 1 (since bonus isn't from draft)
    if (accepted_count <= drafted_count) {
        draft_new_len = draft_original_len + accepted_count - 1;  // -1 because last accepted might be from target
    } else {
        draft_new_len = draft_original_len + drafted_count;  // All draft tokens used
    }
    draft_kv_cache_.truncate(draft_new_len);
}

int SpeculativeRunner::adapt_k(int current_k, float acceptance_rate, const SpeculativeConfig& config) {
    // Increase K if acceptance rate is high
    if (acceptance_rate > 0.9f && current_k < config.max_k) {
        return current_k + 1;
    }
    // Decrease K if acceptance rate is low
    if (acceptance_rate < 0.5f && current_k > config.min_k) {
        return current_k - 1;
    }
    return current_k;
}

void SpeculativeRunner::cancel() {
    cancelled_ = true;
}

void SpeculativeRunner::reset() {
    target_kv_cache_.clear();
    draft_kv_cache_.clear();
    stats_ = Stats{};
}

int32_t SpeculativeRunner::sample(const Tensor& logits, const GenerationConfig& config) {
    // For speculative decoding, we always use greedy (argmax)
    // Sampling with temperature would complicate verification
    return argmax(logits);
}

int32_t SpeculativeRunner::argmax(const Tensor& logits) {
    auto map_result = target_backend_->map_buffer(logits.buffer());
    if (!map_result.ok()) {
        return 0;
    }

    const float* data = static_cast<const float*>(map_result.value());
    int seq_len = static_cast<int>(logits.size(1));
    int vocab_size = static_cast<int>(logits.size(2));

    // Get logits for last position
    const float* last_logits = data + (seq_len - 1) * vocab_size;

    int32_t best_idx = 0;
    float best_val = last_logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (last_logits[i] > best_val) {
            best_val = last_logits[i];
            best_idx = i;
        }
    }

    target_backend_->unmap_buffer(logits.buffer());
    return best_idx;
}

// =============================================================================
// Tree-based Speculative Decoding
// =============================================================================

void SpeculativeRunner::build_tree(
    SpeculationTree& tree,
    int32_t root_token,
    int width,
    int depth)
{
    tree.clear();

    // Add root node
    int root_idx = tree.add_node(root_token, -1);

    // BFS to build tree layer by layer
    std::vector<int> current_level = {root_idx};
    int current_depth = 0;
    int draft_start_pos = draft_kv_cache_.seq_len();

    while (current_depth < depth && !current_level.empty()) {
        std::vector<int> next_level;

        for (int parent_idx : current_level) {
            int32_t parent_token = tree.get_node(parent_idx)->token;

            // Get logits from draft model for this token
            auto logits_result = draft_model_.forward_single(parent_token, draft_kv_cache_);
            if (!logits_result.ok()) {
                continue;
            }

            // Get top-k tokens from draft logits
            auto top_k = get_top_k(logits_result.value(), width);

            // Add children to tree
            for (const auto& [token, log_prob] : top_k) {
                int child_idx = tree.add_node(token, parent_idx, log_prob);
                next_level.push_back(child_idx);
            }
        }

        current_level = std::move(next_level);
        current_depth++;

        // Limit tree size to avoid explosion
        if (tree.size() > 64) {
            break;
        }
    }

}

std::vector<int32_t> SpeculativeRunner::verify_tree(
    SpeculationTree& tree,
    int32_t last_accepted_token)
{
    if (tree.size() == 0) {
        return {};
    }

    // First, add the last accepted token to target KV cache
    auto first_logits_result = target_model_.forward_single(last_accepted_token, target_kv_cache_);
    if (!first_logits_result.ok()) {
        return {};
    }

    // Get target's prediction for first tree position
    int32_t first_target_choice = argmax(first_logits_result.value());

    // Get flattened tree tokens and parent indices for tree attention
    auto tree_tokens = tree.flatten_tokens();
    auto parent_indices = tree.get_parent_indices();

    // Verify root token first
    if (tree.get_node(0)->token != first_target_choice) {
        // Root doesn't match - return target's choice
        return {first_target_choice};
    }

    // Forward all tree nodes through target with tree attention
    int target_start_pos = target_kv_cache_.seq_len();
    auto tree_logits_result = target_model_.forward_tree(
        tree_tokens, parent_indices, &target_kv_cache_, target_start_pos);

    if (!tree_logits_result.ok()) {
        // Fallback to just the first token
        return {first_target_choice};
    }

    auto tree_logits = std::move(tree_logits_result).take();

    // Get target's choices for each tree position
    auto map_result = target_backend_->map_buffer(tree_logits.buffer());
    if (!map_result.ok()) {
        return {first_target_choice};
    }

    const float* logits_data = static_cast<const float*>(map_result.value());
    int vocab_size = static_cast<int>(tree_logits.size(2));
    int num_nodes = static_cast<int>(tree_tokens.size());

    // Get argmax for each position
    std::vector<int32_t> target_choices(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        const float* pos_logits = logits_data + i * vocab_size;
        int32_t best_idx = 0;
        float best_val = pos_logits[0];
        for (int v = 1; v < vocab_size; v++) {
            if (pos_logits[v] > best_val) {
                best_val = pos_logits[v];
                best_idx = v;
            }
        }
        target_choices[i] = best_idx;
    }

    target_backend_->unmap_buffer(tree_logits.buffer());

    // Find longest accepted path
    auto accepted_path = tree.find_accepted_path(target_choices);

    // The first token was already verified, include the root
    std::vector<int32_t> result;
    result.push_back(first_target_choice);  // Root token

    // Add tokens from the accepted path (skipping root if it matches)
    for (size_t i = 0; i < accepted_path.size(); i++) {
        if (i == 0 && accepted_path[i] == first_target_choice) {
            continue;  // Skip duplicate root
        }
        result.push_back(accepted_path[i]);
    }

    // If we accepted everything, add a bonus token
    if (result.size() > static_cast<size_t>(tree.max_depth())) {
        // Find the last accepted position and get target's next choice
        // This is already included via find_accepted_path's mismatch handling
    }

    return result;
}

std::vector<std::pair<int32_t, float>> SpeculativeRunner::get_top_k(
    const Tensor& logits,
    int k)
{
    auto map_result = draft_backend_->map_buffer(logits.buffer());
    if (!map_result.ok()) {
        return {};
    }

    const float* data = static_cast<const float*>(map_result.value());
    int seq_len = static_cast<int>(logits.size(1));
    int vocab_size = static_cast<int>(logits.size(2));

    // Get logits for last position
    const float* last_logits = data + (seq_len - 1) * vocab_size;

    // Find top-k tokens
    std::vector<std::pair<float, int32_t>> scores;
    scores.reserve(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        scores.emplace_back(last_logits[i], i);
    }

    // Partial sort to get top-k
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    draft_backend_->unmap_buffer(logits.buffer());

    // Return top-k as (token, log_prob) pairs
    std::vector<std::pair<int32_t, float>> result;
    result.reserve(k);
    for (int i = 0; i < k && i < static_cast<int>(scores.size()); i++) {
        result.emplace_back(scores[i].second, scores[i].first);
    }

    return result;
}

Result<void> SpeculativeRunner::generate_streaming_tree(
    const std::string& prompt,
    const GenerationConfig& config,
    TokenCallback callback,
    const SpeculativeConfig& spec_config)
{
    cancelled_ = false;
    stats_ = Stats{};

    if (!tokenizer_.is_loaded()) {
        GRANITE_FAIL(ErrorCode::InvalidState, "Tokenizer not loaded");
    }

    // Tokenize prompt
    auto prompt_tokens = tokenizer_.encode(prompt, true);

    // Clear KV caches
    target_kv_cache_.clear();
    draft_kv_cache_.clear();

    // Create token tensor for prefill
    std::vector<int64_t> prompt_shape = {1, static_cast<int64_t>(prompt_tokens.size())};

    auto target_ids_result = Tensor::allocate(prompt_shape, DataType::INT32, target_backend_.get());
    auto draft_ids_result = Tensor::allocate(prompt_shape, DataType::INT32, draft_backend_.get());

    if (!target_ids_result.ok() || !draft_ids_result.ok()) {
        GRANITE_FAIL(ErrorCode::InternalError, "Failed to allocate prompt tensors");
    }

    auto target_ids = std::move(target_ids_result).take();
    auto draft_ids = std::move(draft_ids_result).take();

    // Copy tokens
    auto map_target = target_backend_->map_buffer(target_ids.buffer());
    auto map_draft = draft_backend_->map_buffer(draft_ids.buffer());

    if (map_target.ok() && map_draft.ok()) {
        auto* target_ptr = static_cast<int32_t*>(map_target.value());
        auto* draft_ptr = static_cast<int32_t*>(map_draft.value());
        std::copy(prompt_tokens.begin(), prompt_tokens.end(), target_ptr);
        std::copy(prompt_tokens.begin(), prompt_tokens.end(), draft_ptr);
        target_backend_->unmap_buffer(target_ids.buffer());
        draft_backend_->unmap_buffer(draft_ids.buffer());
    }

    // Prefill both models
    auto target_logits_result = target_model_.forward(target_ids, &target_kv_cache_, 0);
    if (!target_logits_result.ok()) {
        return target_logits_result.error();
    }
    auto target_logits = std::move(target_logits_result).take();

    auto draft_logits_result = draft_model_.forward(draft_ids, &draft_kv_cache_, 0);
    if (!draft_logits_result.ok()) {
        return draft_logits_result.error();
    }

    // Sample first token
    int32_t last_token = argmax(target_logits);

    if (last_token == tokenizer_.eos_token()) {
        return {};
    }

    std::string token_str = tokenizer_.decode_token(last_token);
    if (!callback(token_str)) {
        return {};
    }

    // Tree speculative decoding loop
    int width = spec_config.tree_width;
    int depth = spec_config.tree_depth;
    int generated = 1;

    SpeculationTree tree;

    while (generated < config.max_tokens && !cancelled_) {
        // 1. Build speculation tree from draft model
        build_tree(tree, last_token, width, depth);
        stats_.total_draft_tokens += tree.size();

        // 2. Verify tree with target model
        auto accepted = verify_tree(tree, last_token);
        stats_.total_target_forwards++;
        stats_.total_accepted_tokens += static_cast<int>(accepted.size());

        // 3. Output accepted tokens
        bool should_stop = false;
        for (size_t i = 0; i < accepted.size(); i++) {
            int32_t token = accepted[i];

            if (token == tokenizer_.eos_token()) {
                should_stop = true;
                break;
            }

            if (std::find(config.stop_tokens.begin(), config.stop_tokens.end(),
                          token) != config.stop_tokens.end()) {
                should_stop = true;
                break;
            }

            token_str = tokenizer_.decode_token(token);
            if (!callback(token_str)) {
                should_stop = true;
                break;
            }

            generated++;
            if (generated >= config.max_tokens) {
                should_stop = true;
                break;
            }
        }

        if (should_stop || accepted.empty()) {
            break;
        }

        // Update last token
        last_token = accepted.back();

        // 4. Sync KV caches
        // For tree verification, we need to truncate to accepted path length
        int tree_size = tree.size();
        int accepted_depth = static_cast<int>(accepted.size());

        // Target: added 1 + tree_size tokens, keep accepted_depth
        int target_original = target_kv_cache_.seq_len() - 1 - tree_size;
        target_kv_cache_.truncate(target_original + accepted_depth);

        // Draft: need to rebuild from accepted position
        // For simplicity, truncate to match target
        int draft_original = draft_kv_cache_.seq_len() - tree_size;
        draft_kv_cache_.truncate(draft_original + accepted_depth - 1);

        tree.clear();
    }

    GRANITE_LOG_INFO("Tree speculative decoding stats: {} drafted, {} accepted ({:.1f}%), {} target forwards",
                     stats_.total_draft_tokens, stats_.total_accepted_tokens,
                     stats_.acceptance_rate() * 100.0f, stats_.total_target_forwards);

    return {};
}

}  // namespace granite
