#pragma once 
#include "mutation.h"

struct LargeStep
    : public Mutation
{
    LargeStep() = default; // [sholl]
    LargeStep(std::shared_ptr<PiecewiseConstant1D> lengthDist)
        : lengthDist(lengthDist)
    {}

    Float Mutate(const MLTState &mltState,
                 const Float normalization,
                 MarkovState &currentState,
                 MarkovState &proposalState,
                 RNG &rng,
                 Chain *chain = NULL) override;

    std::shared_ptr<PiecewiseConstant1D> lengthDist;
    std::vector<SubpathContrib> spContribs;
    std::vector<Float> contribCdf;
    Float lastScoreSum = Float(1.0);
    Float lastScore = Float(1.0);
};

// [sholl] moved Mutate implementation to cpp
