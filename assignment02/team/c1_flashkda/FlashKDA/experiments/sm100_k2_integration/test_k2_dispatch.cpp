#include "k2_dispatch.h"

#include <cassert>
#include <cstdio>

using flash_kda::K2DispatchConfig;
using flash_kda::K2Implementation;
using flash_kda::K2Mode;
using flash_kda::parse_k2_mode;
using flash_kda::select_k2_implementation;

K2DispatchConfig supported(int64_t batch, int64_t sequence_length) {
    K2DispatchConfig config;
    config.v0_compiled = true;
    config.v1a_compiled = true;
    config.compute_major = 10;
    config.compute_minor = 3;
    config.has_state_in = true;
    config.has_state_out = true;
    config.total_tokens = batch * sequence_length;
    config.sequences = batch;
    return config;
}

int main() {
    // The caller substitutes "baseline" when the environment variable is unset.
    assert(parse_k2_mode("baseline") == K2Mode::Baseline);
    assert(parse_k2_mode("auto") == K2Mode::Auto);
    assert(parse_k2_mode("invalid") == K2Mode::Invalid);

    assert(select_k2_implementation(K2Mode::Auto, supported(1, 8192)) ==
           K2Implementation::V1A);                         // 512 chunks
    assert(select_k2_implementation(K2Mode::Auto, supported(4, 2048)) ==
           K2Implementation::V1A);                         // 128 chunks
    assert(select_k2_implementation(K2Mode::Auto, supported(8, 1024)) ==
           K2Implementation::Baseline);                    // 64 chunks
    assert(select_k2_implementation(K2Mode::Auto, supported(4, 127 * 16)) ==
           K2Implementation::Baseline);
    assert(select_k2_implementation(K2Mode::Auto, supported(4, 128 * 16)) ==
           K2Implementation::V1A);

    auto varlen = supported(1, 8192);
    varlen.is_varlen = true;
    assert(select_k2_implementation(K2Mode::Auto, varlen) ==
           K2Implementation::Baseline);

    assert(select_k2_implementation(K2Mode::Baseline, supported(1, 8192)) ==
           K2Implementation::Baseline);
    assert(select_k2_implementation(K2Mode::V1A, supported(8, 1024)) ==
           K2Implementation::V1A);                         // force ignores threshold
    assert(select_k2_implementation(K2Mode::SM100V0, supported(8, 1024)) ==
           K2Implementation::SM100V0);

    auto no_state = supported(1, 8192);
    no_state.has_state_in = false;
    assert(select_k2_implementation(K2Mode::Auto, no_state) ==
           K2Implementation::Baseline);
    auto fp32_state = supported(1, 8192);
    fp32_state.state_fp32 = true;
    assert(select_k2_implementation(K2Mode::Auto, fp32_state) ==
           K2Implementation::Baseline);
    auto sm100 = supported(1, 8192);
    sm100.compute_minor = 0;
    assert(select_k2_implementation(K2Mode::Auto, sm100) ==
           K2Implementation::Baseline);
    assert(select_k2_implementation(K2Mode::V1A, sm100) ==
           K2Implementation::Unsupported);

    auto not_compiled = supported(1, 8192);
    not_compiled.v1a_compiled = false;
    assert(select_k2_implementation(K2Mode::Auto, not_compiled) ==
           K2Implementation::Baseline);
    assert(select_k2_implementation(K2Mode::V1A, not_compiled) ==
           K2Implementation::Unsupported);
    std::puts("K2 dispatch tests passed");
}
