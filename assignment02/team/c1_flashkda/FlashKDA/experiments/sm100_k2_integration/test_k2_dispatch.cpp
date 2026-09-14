#include "k2_dispatch.h"

#include <cassert>
#include <cstdio>
#include <initializer_list>

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
    config.heads = 32;  // Existing tests exercise the H!=64 conservative policy.
    return config;
}

int main() {
    // The caller substitutes "baseline" when the environment variable is unset.
    assert(parse_k2_mode("baseline") == K2Mode::Baseline);
    assert(parse_k2_mode("auto") == K2Mode::Auto);
    assert(parse_k2_mode("v1ae") == K2Mode::V1AE);
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
    auto egress = supported(4, 16);
    egress.heads = 64;
    egress.v1ae_compiled = true;
    assert(select_k2_implementation(K2Mode::V1AE, egress) == K2Implementation::V1AE);
    // V1aE compilation is independent of the frozen V1a specialization.
    egress.v1a_compiled = false;
    assert(select_k2_implementation(K2Mode::V1AE, egress) == K2Implementation::V1AE);
    auto unavailable = egress;
    unavailable.v1ae_compiled = false;
    assert(select_k2_implementation(K2Mode::V1AE, unavailable) == K2Implementation::Unsupported);
    for (int condition = 0; condition < 6; ++condition) {
        auto unsupported = egress;
        if (condition == 0) unsupported.is_varlen = true;
        if (condition == 1) unsupported.state_fp32 = true;
        if (condition == 2) unsupported.has_state_in = false;
        if (condition == 3) unsupported.has_state_out = false;
        if (condition == 4) unsupported.compute_minor = 0;
        if (condition == 5) unsupported.sequences = 0;
        assert(select_k2_implementation(K2Mode::V1AE, unsupported) == K2Implementation::Unsupported);
        assert(select_k2_implementation(K2Mode::Auto, unsupported) == K2Implementation::Baseline);
    }
    for (int heads : {1, 32, 65, 128}) {
        for (int chunks : {127, 128}) {
            auto fallback = supported(4, chunks * 16);
            fallback.heads = heads;
            fallback.v1ae_compiled = true;
            assert(select_k2_implementation(K2Mode::Auto, fallback) ==
                   (chunks >= 128 ? K2Implementation::V1A : K2Implementation::Baseline));
            assert(select_k2_implementation(K2Mode::Baseline, fallback) == K2Implementation::Baseline);
            assert(select_k2_implementation(K2Mode::SM100V0, fallback) == K2Implementation::SM100V0);
            assert(select_k2_implementation(K2Mode::V1A, fallback) == K2Implementation::V1A);
        }
    }
    using Impl = K2Implementation;
    struct Boundary { int batch, chunks; Impl expected; };
    const Boundary boundaries[] = {
        {1,79,Impl::Baseline}, {1,80,Impl::V1AE}, {1,159,Impl::V1AE}, {1,160,Impl::V1A},
        {2,31,Impl::Baseline}, {2,32,Impl::V1AE}, {2,95,Impl::V1AE}, {2,96,Impl::V1A},
        {3,23,Impl::Baseline}, {3,24,Impl::V1AE}, {3,319,Impl::V1AE}, {3,320,Impl::V1A},
        {4,31,Impl::Baseline}, {4,32,Impl::V1AE}, {4,255,Impl::V1AE}, {4,256,Impl::V1A},
        {5,23,Impl::Baseline}, {5,24,Impl::V1AE}, {5,319,Impl::V1AE}, {5,320,Impl::V1AE}, {5,321,Impl::V1A},
        {6,23,Impl::Baseline}, {6,24,Impl::V1AE}, {6,319,Impl::V1AE}, {6,320,Impl::V1AE}, {6,321,Impl::V1A},
        {7,23,Impl::Baseline}, {7,24,Impl::V1AE}, {7,319,Impl::V1AE}, {7,320,Impl::V1AE}, {7,321,Impl::V1A},
        {8,23,Impl::Baseline}, {8,24,Impl::V1AE}, {8,319,Impl::V1AE}, {8,320,Impl::Baseline},
        {8,512,Impl::Baseline}, {9,24,Impl::Baseline}, {9,128,Impl::Baseline}, {9,512,Impl::Baseline},
        {16,512,Impl::Baseline}
    };
    for (const auto& test : boundaries) {
        auto config = supported(test.batch, test.chunks * 16);
        config.heads = 64;
        config.v1ae_compiled = true;
        assert(flash_kda::chunks_per_sequence(config) == test.chunks);
        assert(select_k2_implementation(K2Mode::Auto, config) == test.expected);
        // Explicit requests ignore the empirical auto intervals and B>8 limit.
        assert(select_k2_implementation(K2Mode::V1AE, config) == Impl::V1AE);
        assert(select_k2_implementation(K2Mode::V1A, config) == Impl::V1A);
        assert(select_k2_implementation(K2Mode::Baseline, config) == Impl::Baseline);
        assert(select_k2_implementation(K2Mode::SM100V0, config) == Impl::SM100V0);
        for (int v1a_built : {0, 1}) for (int v1ae_built : {0, 1}) {
            auto build = config;
            build.v1a_compiled = v1a_built;
            build.v1ae_compiled = v1ae_built;
            auto expected = test.expected;
            if ((expected == Impl::V1A && !v1a_built) ||
                (expected == Impl::V1AE && !v1ae_built)) expected = Impl::Baseline;
            assert(select_k2_implementation(K2Mode::Auto, build) == expected);
        }
        for (int condition = 0; condition < 7; ++condition) {
            auto unsupported = config;
            if (condition == 0) unsupported.is_varlen = true;
            if (condition == 1) unsupported.state_fp32 = true;
            if (condition == 2) unsupported.has_state_in = false;
            if (condition == 3) unsupported.has_state_out = false;
            if (condition == 4) unsupported.compute_minor = 0;
            if (condition == 5) unsupported.compute_major = 9;
            if (condition == 6) unsupported.sequences = 0;
            assert(select_k2_implementation(K2Mode::Auto, unsupported) == Impl::Baseline);
        }
        // H alone selects the empirical map; do not generalize grid_ctas.
        for (int heads : {32, 65, 128}) {
            config.heads = heads;
            assert(select_k2_implementation(K2Mode::Auto, config) ==
                   (test.chunks >= 128 ? Impl::V1A : Impl::Baseline));
        }
    }
    std::puts("K2 dispatch tests passed");
}
