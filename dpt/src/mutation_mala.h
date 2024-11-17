#pragma once 

#include "mutation.h"
#include "mutation_small.h"
#include "h2mc.h"
#include "mala.h"
#include "fastmath.h"

using DervFuncMap = std::unordered_map<std::pair<int, int>, PathFuncDerv>;
struct MALASmallStep
    : public Mutation
{
    // begin [sholl]
    MALASmallStep() = default;
    MALASmallStep(Scene const* const scene, int const maxDervDepth) {
        initialize(scene, maxDervDepth);
    }

    void initialize(Scene const* const scene, int const maxDervDepth)
    {
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

    SmallStep isotropicSmallStep;
    std::vector<SubpathContrib> spContribs;
    AlignedStdVector sceneParams;
    SerializedSubpath ssubPath;
    AlignedStdVector vGrad;
};

// [sholl] moved implementation Mutate to cpp
