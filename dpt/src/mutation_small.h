#pragma once 
#include "mutation.h"

struct SmallStep
    : public Mutation
{
    Float Mutate(const MLTState &mltState,
                 const Float normalization,
                 MarkovState &currentState,
                 MarkovState &proposalState,
                 RNG &rng,
                 Chain *chain = NULL) override;

    std::vector<SubpathContrib> spContribs;
    Vector offset;
};

// [sholl] moved Mutate implementation to cpp
