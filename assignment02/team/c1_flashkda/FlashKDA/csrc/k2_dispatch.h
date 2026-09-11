#pragma once

#include <cstdint>
#include <string_view>

namespace flash_kda {

constexpr int64_t kK2ChunkSize = 16;
constexpr int64_t kV1AAutoMinChunks = 128;

enum class K2Mode { Baseline, SM100V0, V1A, Auto, Invalid };
enum class K2Implementation { Baseline, SM100V0, V1A, Unsupported };

struct K2DispatchConfig {
    bool v0_compiled = false;
    bool v1a_compiled = false;
    int compute_major = 0;
    int compute_minor = 0;
    bool is_varlen = false;
    bool has_state_in = false;
    bool has_state_out = false;
    bool state_fp32 = false;
    int64_t total_tokens = 0;
    int64_t sequences = 0;
};

constexpr K2Mode parse_k2_mode(std::string_view value) {
    return value == "baseline" ? K2Mode::Baseline
         : value == "sm100_v0" ? K2Mode::SM100V0
         : value == "v1a" ? K2Mode::V1A
         : value == "auto" ? K2Mode::Auto
         : K2Mode::Invalid;
}

constexpr bool is_sm100_or_sm103(K2DispatchConfig const& config) {
    return config.compute_major == 10 &&
           (config.compute_minor == 0 || config.compute_minor == 3);
}

constexpr bool v1a_supported(K2DispatchConfig const& config) {
    return config.v1a_compiled && config.compute_major == 10 &&
           config.compute_minor == 3 && !config.is_varlen &&
           config.has_state_in && config.has_state_out &&
           !config.state_fp32 && config.sequences > 0;
}

constexpr int64_t chunks_per_sequence(K2DispatchConfig const& config) {
    if (config.is_varlen || config.sequences <= 0) return 0;
    const int64_t sequence_length = config.total_tokens / config.sequences;
    return sequence_length / kK2ChunkSize;
}

constexpr K2Implementation select_k2_implementation(
    K2Mode mode, K2DispatchConfig const& config) {
    switch (mode) {
    case K2Mode::Baseline:
        return K2Implementation::Baseline;
    case K2Mode::SM100V0:
        return config.v0_compiled && is_sm100_or_sm103(config)
            ? K2Implementation::SM100V0 : K2Implementation::Unsupported;
    case K2Mode::V1A:
        return v1a_supported(config)
            ? K2Implementation::V1A : K2Implementation::Unsupported;
    case K2Mode::Auto:
        return v1a_supported(config) &&
               chunks_per_sequence(config) >= kV1AAutoMinChunks
            ? K2Implementation::V1A : K2Implementation::Baseline;
    default:
        return K2Implementation::Unsupported;
    }
}

}  // namespace flash_kda
