// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#ifndef HPP_SHOLL_UTILITY_INCLUDED
#define HPP_SHOLL_UTILITY_INCLUDED


#include <algorithm>

#include "../mutation.h"
#include "../scene.h"

#include "../mutation_large.h"
#include "../mutation_small.h"
#include "../mutation_mala.h"
#include "../mutation_h2mc.h"


namespace sholl
{
	template<class>
	inline bool constexpr always_false_v = false;

	template<class Mutation>
	Mutation make_small_step(Scene const* const scene, int const depth_max, Float const sigma)
	{
		if constexpr (std::is_same_v<Mutation, SmallStep>)
			return Mutation{};
		else if constexpr (std::is_same_v<Mutation, MALASmallStep>)
			return Mutation{ scene, depth_max };
		else if constexpr (std::is_same_v<Mutation, H2MCSmallStep>)
			return Mutation{ scene, depth_max, sigma };
		else
			static_assert(always_false_v<Mutation>, "invalid small step mutation");
	}


	template<class Mutation>
	void initialize(Mutation& mutation, Scene const* const scene, int const depth_max, Float const sigma)
	{
		if constexpr (std::is_same_v<Mutation, MALASmallStep>)
			mutation.initialize(scene, depth_max);
		else if constexpr (std::is_same_v<Mutation, H2MCSmallStep>)
			mutation.initialize(scene, depth_max, sigma);
	}

	void reset(Chain& chain)
	{
		chain.pss.resize(0);
		chain.last_pss.resize(0);
		chain.v1.resize(0);
		chain.v2.resize(0);
		chain.g.resize(0);
		chain.M.resize(0);
		chain.curr_new_v1.resize(0);
		chain.curr_new_v2.resize(0);
		chain.curr_new_g.resize(0);
		chain.prop_new_v1.resize(0);
		chain.prop_new_v2.resize(0);
		chain.prop_new_g.resize(0);

		chain.buffered = false;
		Clear(chain.path);
		chain.queried = false;
		chain.t = 0;
	}

	void reset(MarkovState& state)
	{
		state.gaussianInitialized = false;
		state.pss.resize(0);
		state.toSplat.resize(0);
		state.valid = false;
	}

	template<class Mutation>
	void reset(Mutation& mutation)
	{
		if constexpr (std::is_same_v<Mutation, LargeStep>)
		{
			mutation.contribCdf.resize(0);
			mutation.lastScore = 1;
			mutation.lastScoreSum = 1;
			mutation.spContribs.resize(0);
		}
		else if constexpr (std::is_same_v<Mutation, SmallStep>)
		{
			mutation.offset.resize(0);
			mutation.spContribs.resize(0);
		}
		else if constexpr (std::is_same_v<Mutation, MALASmallStep>)
		{
			reset(mutation.isotropicSmallStep);
			mutation.spContribs.resize(0);
			mutation.vGrad.resize(0);
		}
		else if constexpr (std::is_same_v<Mutation, H2MCSmallStep>)
		{
			reset(mutation.isotropicSmallStep);
			mutation.spContribs.resize(0);
			mutation.vGrad.resize(0);
			mutation.vHess.resize(0);
		}
	}

	void update_current_state(MarkovState& current_state, MarkovState& proposed_state)
	{
		// TODO:
		std::swap(current_state.gaussian.covL, proposed_state.gaussian.covL);
		std::swap(current_state.gaussian.covL_d, proposed_state.gaussian.covL_d);
		std::swap(current_state.gaussian.invCov, proposed_state.gaussian.invCov);
		std::swap(current_state.gaussian.invCov_d, proposed_state.gaussian.invCov_d);
		current_state.gaussian.isDiagonal = proposed_state.gaussian.isDiagonal;
		current_state.gaussian.logDet = proposed_state.gaussian.logDet;
		std::swap(current_state.gaussian.mean, proposed_state.gaussian.mean);

		current_state.gaussianInitialized = proposed_state.gaussianInitialized;

		current_state.path.time = proposed_state.path.time;
		current_state.path.camVertex = proposed_state.path.camVertex;
		std::swap(current_state.path.camSurfaceVertex, proposed_state.path.camSurfaceVertex);
		current_state.path.lgtVertex = proposed_state.path.lgtVertex;
		std::swap(current_state.path.lgtSurfaceVertex, proposed_state.path.lgtSurfaceVertex);
		current_state.path.envLightInst = proposed_state.path.envLightInst;
		current_state.path.lensVertexPos = proposed_state.path.lensVertexPos;
		current_state.path.isSubpath = proposed_state.path.isSubpath;
		current_state.path.camDepth = proposed_state.path.camDepth;
		current_state.path.lgtDepth = proposed_state.path.lgtDepth;

		std::swap(current_state.pss, proposed_state.pss);

		current_state.scoreSum = proposed_state.scoreSum;

		current_state.spContrib = proposed_state.spContrib;

		std::swap(current_state.toSplat, proposed_state.toSplat);
	}
} // namespace sholl


#endif // !HPP_SHOLL_UTILITY_INCLUDED
