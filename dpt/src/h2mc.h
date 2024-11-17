#pragma once

#include <numbers> // [sholl] Replaced M_PI with std::numbers::pi_v<Float>

#include "commondef.h"
#include "gaussian.h"
#include "alignedallocator.h"

#include <vector>

struct H2MCParam {
    H2MCParam(const Float sigma = 0.01, const Float L = std::numbers::pi_v<Float> / 2) : sigma(sigma), L(L) {
        posScaleFactor = Float(0.5) * (exp(L) - exp(-L)) * Float(0.5) * (exp(L) - exp(-L));
        posOffsetFactor = Float(0.5) * (exp(L) + exp(-L) - Float(1.0));
        negScaleFactor = sin(L) * sin(L);
        negOffsetFactor = -(cos(L) - Float(1.0));
    }

    Float sigma;
    Float posScaleFactor;
    Float posOffsetFactor;
    Float negScaleFactor;
    Float negOffsetFactor;
    Float L;
};

void ComputeGaussian(const H2MCParam &param,
                     const Float sc,
                     const AlignedStdVector &vGrad,
                     const AlignedStdVector &vHess,
                     Gaussian &gaussian);
