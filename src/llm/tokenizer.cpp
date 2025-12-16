#include "llm_internal.h"

namespace granite {

Result<Tokenizer> Tokenizer::from_gguf(const GGUFFile& gguf) {
    Tokenizer tok;

    // Load vocabulary
    auto tokens = gguf.get_metadata_as<std::vector<std::string>>("tokenizer.ggml.tokens");
    if (!tokens) {
        GRANITE_FAIL(ErrorCode::InvalidFormat, "No tokenizer vocabulary in GGUF");
    }
    tok.vocab_ = *tokens;

    // Build token to ID map
    for (size_t i = 0; i < tok.vocab_.size(); i++) {
        tok.token_to_id_[tok.vocab_[i]] = static_cast<int32_t>(i);
    }

    // Load special tokens
    if (auto bos = gguf.get_metadata_as<uint32_t>("tokenizer.ggml.bos_token_id")) {
        tok.bos_token_ = static_cast<int32_t>(*bos);
    }
    if (auto eos = gguf.get_metadata_as<uint32_t>("tokenizer.ggml.eos_token_id")) {
        tok.eos_token_ = static_cast<int32_t>(*eos);
    }
    if (auto pad = gguf.get_metadata_as<uint32_t>("tokenizer.ggml.padding_token_id")) {
        tok.pad_token_ = static_cast<int32_t>(*pad);
    }
    if (auto unk = gguf.get_metadata_as<uint32_t>("tokenizer.ggml.unknown_token_id")) {
        tok.unk_token_ = static_cast<int32_t>(*unk);
    }

    // Load merges (BPE)
    auto merges = gguf.get_metadata_as<std::vector<std::string>>("tokenizer.ggml.merges");
    if (merges) {
        for (size_t i = 0; i < merges->size(); i++) {
            const auto& merge = (*merges)[i];
            size_t space_pos = merge.find(' ');
            if (space_pos != std::string::npos) {
                std::string first = merge.substr(0, space_pos);
                std::string second = merge.substr(space_pos + 1);
                tok.merges_.emplace_back(first, second);
                tok.merge_ranks_[{first, second}] = static_cast<int>(i);
            }
        }
    }

    GRANITE_LOG_INFO("Loaded tokenizer: vocab_size={}, bos={}, eos={}",
                     tok.vocab_.size(), tok.bos_token_, tok.eos_token_);

    return tok;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> tokens;

    if (add_bos) {
        tokens.push_back(bos_token_);
    }

    if (text.empty()) {
        return tokens;
    }

    // Simple character-level tokenization as fallback
    std::vector<std::string> chars;
    for (size_t i = 0; i < text.size(); ) {
        // Handle UTF-8 (simplified)
        unsigned char c = static_cast<unsigned char>(text[i]);
        int char_len = 1;
        if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        chars.push_back(text.substr(i, char_len));
        i += char_len;
    }

    // Apply BPE merges
    while (chars.size() > 1) {
        int best_idx = -1;
        int best_rank = INT_MAX;

        for (size_t i = 0; i < chars.size() - 1; i++) {
            auto it = merge_ranks_.find({chars[i], chars[i + 1]});
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx < 0) break;

        // Merge
        chars[best_idx] = chars[best_idx] + chars[best_idx + 1];
        chars.erase(chars.begin() + best_idx + 1);
    }

    // Convert to token IDs
    for (const auto& tok_str : chars) {
        auto it = token_to_id_.find(tok_str);
        if (it != token_to_id_.end()) {
            tokens.push_back(it->second);
        } else {
            // Try to find as byte tokens
            for (unsigned char c : tok_str) {
                std::string byte_tok = "<0x" + fmt::format("{:02X}", c) + ">";
                auto byte_it = token_to_id_.find(byte_tok);
                if (byte_it != token_to_id_.end()) {
                    tokens.push_back(byte_it->second);
                } else {
                    tokens.push_back(unk_token_);
                }
            }
        }
    }

    return tokens;
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string result;
    for (int32_t id : tokens) {
        result += decode_token(id);
    }
    return result;
}

std::string Tokenizer::decode_token(int32_t token) const {
    if (token == bos_token_ || token == eos_token_ || token == pad_token_) {
        return "";
    }
    if (token >= 0 && token < static_cast<int32_t>(vocab_.size())) {
        const std::string& tok = vocab_[token];
        std::string result = tok;

        // Handle byte tokens like "<0x20>" -> space
        if (result.size() == 6 && result.substr(0, 3) == "<0x" && result[5] == '>') {
            std::string hex = result.substr(3, 2);
            int byte_val = std::stoi(hex, nullptr, 16);
            return std::string(1, static_cast<char>(byte_val));
        }

        // Handle special tokens like "▁" (sentencepiece space)
        size_t pos = 0;
        while ((pos = result.find("\xe2\x96\x81", pos)) != std::string::npos) {
            result.replace(pos, 3, " ");
            pos += 1;
        }
        return result;
    }
    return "";
}

}  // namespace granite
