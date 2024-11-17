#pragma once 

#include "mutation.h"
#include "mutation_small.h"

using DervFuncMap = std::unordered_map<std::pair<int, int>, PathFuncDerv>;
struct H2MCSmallStep
    : public Mutation
{
    // begin [sholl]
    H2MCSmallStep() = default;
    H2MCSmallStep(Scene const* const scene, int const maxDervDepth, Float const sigma) {
        initialize(scene, maxDervDepth, sigma);
    }

    void initialize(Scene const* const scene, int const maxDervDepth, Float const sigma)
    {
        h2mcParam = sigma;
        sceneParams.resize(GetSceneSerializedSize());
        Serialize(scene, &sceneParams[0]);
        ssubPath.primary.resize(GetPrimaryParamSize(maxDervDepth, maxDervDepth));
        ssubPath.vertParams.resize(GetVertParamSize(maxDervDepth, maxDervDepth));
    }
    // end [sholl]

    Float Mutate(const MLTState &mltState,
                 const Float normalization,
                 MarkovState &currentState,
                 MarkovState &proposalState,
                 RNG &rng, 
                 Chain *chain = NULL) override;

    std::vector<SubpathContrib> spContribs;
    H2MCParam h2mcParam;
    AlignedStdVector sceneParams;
    SerializedSubpath ssubPath;
    SmallStep isotropicSmallStep;

    AlignedStdVector vGrad;
    AlignedStdVector vHess;
};

// [sholl] moved implementation Mutate to cpp
