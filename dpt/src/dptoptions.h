#pragma once

#include "commondef.h"

#include <chrono>
#include <string>

struct DptOptions {
    std::string integrator = "mcmc";                 // MC or MCMC
    bool bidirectional = true;
    int spp = 256;
    int numInitSamples = 300000;
    int minDepth = -1;
    int maxDepth = 8;
    int directSpp = 256;

    bool h2mc = false;                               // Hessian-based H2MC kernel
    Float perturbStdDev = Float(0.01);               // H2MC small step sigma
    Float roughnessThreshold = Float(0.05);          // roughness
    Float largeStepProbability = Float(0.05);        // Large step probability
    Float largeStepProbScale = Float(1.0);           // Scale up large step probability in MALA second phase
    bool mala = false;                               // MALA-based kernel 
    Float malaGN = Float(100.0);                     // MALA truncated gradient magnitude
    Float malaStepsize = Float(0.005);               // MALA stepsize
    Float malaStdDev = Float(0.005);                 // MALA shrink prior to prevent noisy gradient issue
    bool sampleFromGlobalCache = false;              // Sampling from the cache for global jumps

    int numChains = 128;
    int seedOffset = 0;
    int reportIntervalSpp = 0;
    Float discreteStdDev = Float(0.01);
    Float uniformMixingProbability = Float(0.1);      
    bool useLightCoordinateSampling = false;         // turned off by default 
    bool largeStepMultiplexed = false;               // turned off by default

    // begin [sholl]
    int direct_depth_min,
        direct_depth_max = -1;
    Float expected_lifetime = 1,
        sigma = .01;
    bool rao_blackwellize = true,
        compute_mse = false;
    std::size_t realization_count = 1;
    std::string reference{};

    std::size_t chain_count = 1000;

    std::chrono::duration<Float, std::nano> frame_period = std::chrono::seconds(1),
        rendering_time = std::chrono::seconds(1);
    // end [sholl]
};

inline std::string GetLibPath() {
    return std::string(getenv("DPT_LIBPATH"));
}
