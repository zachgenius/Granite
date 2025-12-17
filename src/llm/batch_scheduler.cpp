// BatchScheduler - Continuous batching for serving multiple concurrent requests
//
// Key concepts:
// - Prefill queue: New requests that need prompt processing
// - Decode queue: Active requests generating tokens
// - Each request has its own KV cache slot from the pool
// - Batches are assembled to maximize GPU utilization

#include "llm_internal.h"

namespace granite {

Result<void> BatchScheduler::initialize(
    TransformerModel* model,
    Tokenizer* tokenizer,
    int num_cache_slots,
    int max_batch_tokens)
{
    model_ = model;
    tokenizer_ = tokenizer;
    max_batch_tokens_ = max_batch_tokens;

    // Use the model's backend for KV cache pool
    // This ensures buffer mapping works correctly
    if (!model || !model->backend()) {
        GRANITE_FAIL(ErrorCode::InvalidArgument, "Model or backend is null");
    }

    // Allocate KV cache pool using model's backend
    kv_pool_ = std::make_unique<KVCachePool>();
    auto pool_result = kv_pool_->allocate(
        num_cache_slots,
        model->config(),
        model->config().max_seq_len,
        model->backend());

    if (!pool_result.ok()) {
        return pool_result.error();
    }

    GRANITE_LOG_INFO("BatchScheduler initialized: {} cache slots, max {} batch tokens",
                     num_cache_slots, max_batch_tokens);

    return {};
}

int BatchScheduler::submit(
    const std::string& prompt,
    const GenerationConfig& config,
    TokenCallback callback)
{
    auto request = std::make_unique<GenerationRequest>();
    request->request_id = next_request_id_++;
    request->prompt = prompt;
    request->config = config;
    request->callback = callback;
    request->state = GenerationRequest::State::PENDING;
    request->start_time = std::chrono::steady_clock::now();

    // Tokenize prompt
    if (tokenizer_ && tokenizer_->is_loaded()) {
        request->prompt_tokens = tokenizer_->encode(prompt, true);
    }

    int id = request->request_id;
    prefill_queue_.push_back(std::move(request));

    GRANITE_LOG_DEBUG("Submitted request {}: {} tokens", id, prefill_queue_.back()->prompt_tokens.size());

    return id;
}

int BatchScheduler::step() {
    int completed_count = 0;

    // Priority: Process prefill requests first (they're waiting)
    if (!prefill_queue_.empty()) {
        auto batch = assemble_prefill_batch();
        if (!batch.tokens.empty()) {
            process_batch(batch);
        }
    }

    // Then process decode requests
    if (!decode_queue_.empty()) {
        auto batch = assemble_decode_batch();
        if (!batch.tokens.empty()) {
            process_batch(batch);
        }
    }

    // Count completed
    for (const auto& req : completed_) {
        if (req.state == GenerationRequest::State::COMPLETED) {
            completed_count++;
        }
    }

    return completed_count;
}

BatchScheduler::Batch BatchScheduler::assemble_prefill_batch() {
    Batch batch;
    batch.is_prefill = true;

    // For now, process one prefill at a time (simpler implementation)
    // Could batch multiple short prefills later

    if (prefill_queue_.empty()) {
        return batch;
    }

    // Get first request from prefill queue
    auto& request = prefill_queue_.front();

    // Try to acquire a KV cache slot
    int slot = kv_pool_->acquire_slot();
    if (slot < 0) {
        // No free slots - wait
        return batch;
    }

    request->kv_cache_slot = slot;
    request->state = GenerationRequest::State::PREFILLING;

    // Calculate how many tokens to prefill this step
    int remaining = static_cast<int>(request->prompt_tokens.size()) - request->prefill_pos;
    int to_process = std::min(remaining, max_batch_tokens_);

    // Add tokens to batch
    for (int i = 0; i < to_process; i++) {
        int pos = request->prefill_pos + i;
        batch.tokens.push_back(request->prompt_tokens[pos]);
        batch.positions.push_back(pos);
        batch.kv_slots.push_back(slot);
        batch.request_ids.push_back(request->request_id);
    }

    // Update prefill progress
    request->prefill_pos += to_process;

    // If prefill complete, move to decode queue
    if (request->prefill_pos >= static_cast<int>(request->prompt_tokens.size())) {
        request->state = GenerationRequest::State::DECODING;
        request->first_token_time = std::chrono::steady_clock::now();
        decode_queue_.push_back(std::move(prefill_queue_.front()));
        prefill_queue_.pop_front();
    }

    return batch;
}

BatchScheduler::Batch BatchScheduler::assemble_decode_batch() {
    Batch batch;
    batch.is_prefill = false;

    // Batch decode tokens from active requests
    // Each request contributes one token per step

    int tokens_added = 0;
    for (auto& request : decode_queue_) {
        if (tokens_added >= max_batch_tokens_) {
            break;
        }

        if (request->state != GenerationRequest::State::DECODING) {
            continue;
        }

        // Get last generated token (or use a special start token)
        int32_t last_token;
        if (request->generated_tokens.empty()) {
            // First decode step - use last prompt token
            last_token = request->prompt_tokens.back();
        } else {
            last_token = request->generated_tokens.back();
        }

        int pos = static_cast<int>(request->prompt_tokens.size()) +
                  static_cast<int>(request->generated_tokens.size());

        batch.tokens.push_back(last_token);
        batch.positions.push_back(pos);
        batch.kv_slots.push_back(request->kv_cache_slot);
        batch.request_ids.push_back(request->request_id);
        tokens_added++;
    }

    return batch;
}

void BatchScheduler::process_batch(const Batch& batch) {
    if (batch.tokens.empty() || !model_) {
        return;
    }

    // For continuous batching, we need to process tokens from different
    // requests with their respective KV caches. This is a simplified
    // version that processes requests sequentially.

    // Group tokens by request
    std::unordered_map<int, std::vector<size_t>> request_token_indices;
    for (size_t i = 0; i < batch.tokens.size(); i++) {
        request_token_indices[batch.request_ids[i]].push_back(i);
    }

    // Process each request's tokens
    for (auto& [req_id, indices] : request_token_indices) {
        // Find the request
        GenerationRequest* request = nullptr;
        for (auto& r : prefill_queue_) {
            if (r->request_id == req_id) {
                request = r.get();
                break;
            }
        }
        if (!request) {
            for (auto& r : decode_queue_) {
                if (r->request_id == req_id) {
                    request = r.get();
                    break;
                }
            }
        }
        if (!request) continue;

        // Get KV cache for this request
        if (request->kv_cache_slot < 0) continue;
        auto& kv_cache = kv_pool_->get_cache(request->kv_cache_slot);

        if (batch.is_prefill) {
            // Prefill: process multiple tokens
            std::vector<int32_t> tokens;
            for (size_t idx : indices) {
                tokens.push_back(batch.tokens[idx]);
            }

            int start_pos = kv_cache.seq_len();
            auto logits_result = model_->forward_batch(tokens, &kv_cache, start_pos);

            if (!logits_result.ok()) {
                request->state = GenerationRequest::State::FAILED;
                continue;
            }

            // For prefill, we don't output tokens yet
        } else {
            // Decode: single token per request
            int32_t token = batch.tokens[indices[0]];
            auto logits_result = model_->forward_single(token, kv_cache);

            if (!logits_result.ok()) {
                request->state = GenerationRequest::State::FAILED;
                continue;
            }

            // Sample next token using model's backend for buffer mapping
            auto& logits = logits_result.value();
            auto map_result = model_->backend()->map_buffer(logits.buffer());
            if (!map_result.ok()) continue;

            const float* data = static_cast<const float*>(map_result.value());
            int vocab_size = static_cast<int>(logits.size(2));

            // Greedy sampling
            int32_t next_token = 0;
            float max_val = data[0];
            for (int v = 1; v < vocab_size; v++) {
                if (data[v] > max_val) {
                    max_val = data[v];
                    next_token = v;
                }
            }

            model_->backend()->unmap_buffer(logits.buffer());

            // Add generated token
            request->generated_tokens.push_back(next_token);

            // Call callback if provided
            if (request->callback && tokenizer_) {
                std::string token_str = tokenizer_->decode_token(next_token);
                bool continue_gen = request->callback(token_str);
                if (!continue_gen) {
                    request->state = GenerationRequest::State::COMPLETED;
                }
            }

            // Check for completion
            if (tokenizer_ && next_token == tokenizer_->eos_token()) {
                request->state = GenerationRequest::State::COMPLETED;
            }
            if (static_cast<int>(request->generated_tokens.size()) >= request->config.max_tokens) {
                request->state = GenerationRequest::State::COMPLETED;
            }
        }
    }

    // Move completed requests
    auto it = decode_queue_.begin();
    while (it != decode_queue_.end()) {
        if ((*it)->state == GenerationRequest::State::COMPLETED ||
            (*it)->state == GenerationRequest::State::FAILED) {
            // Release KV cache slot
            kv_pool_->release_slot((*it)->kv_cache_slot);
            completed_.push_back(std::move(**it));
            it = decode_queue_.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<GenerationRequest> BatchScheduler::take_completed() {
    std::vector<GenerationRequest> result = std::move(completed_);
    completed_.clear();
    return result;
}

void BatchScheduler::cancel(int request_id) {
    // Check prefill queue
    for (auto it = prefill_queue_.begin(); it != prefill_queue_.end(); ++it) {
        if ((*it)->request_id == request_id) {
            if ((*it)->kv_cache_slot >= 0) {
                kv_pool_->release_slot((*it)->kv_cache_slot);
            }
            prefill_queue_.erase(it);
            return;
        }
    }

    // Check decode queue
    for (auto it = decode_queue_.begin(); it != decode_queue_.end(); ++it) {
        if ((*it)->request_id == request_id) {
            if ((*it)->kv_cache_slot >= 0) {
                kv_pool_->release_slot((*it)->kv_cache_slot);
            }
            decode_queue_.erase(it);
            return;
        }
    }
}

void BatchScheduler::cancel_all() {
    // Release all KV cache slots
    for (auto& req : prefill_queue_) {
        if (req->kv_cache_slot >= 0) {
            kv_pool_->release_slot(req->kv_cache_slot);
        }
    }
    for (auto& req : decode_queue_) {
        if (req->kv_cache_slot >= 0) {
            kv_pool_->release_slot(req->kv_cache_slot);
        }
    }

    prefill_queue_.clear();
    decode_queue_.clear();
}

bool BatchScheduler::has_pending() const {
    return !prefill_queue_.empty() || !decode_queue_.empty();
}

}  // namespace granite
