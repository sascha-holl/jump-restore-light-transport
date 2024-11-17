// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#ifndef HPP_SHOLL_SAMPLER_LOCAL_EXPLORATION_SAMPLER_INCLUDED
#define HPP_SHOLL_SAMPLER_LOCAL_EXPLORATION_SAMPLER_INCLUDED


#include "sampler.hpp"

#include <sholl/random/local_exploration_sampler.hpp>


namespace pbrt
{
	class local_exploration_sampler
		: public sampler<local_exploration_sampler>
	{
	public:
		local_exploration_sampler(
			int const sample_count_per_pixel,
			std::mt19937& g,
			Float const large_step_probability,
			Float const sigma)
			: sampler{ sample_count_per_pixel, g },
			  m_sampler(large_step_probability, sigma)
		{}

		bool begin_iteration() {
			return m_sampler.begin_iteration(m_g);
		}

		void begin_stream(std::size_t const stream_index, std::size_t const off = 0) {
			return m_sampler.begin_stream(stream_index, off);
		}

		bool end_iteration(
			Float const target_density_at_current_sample,
			Float const target_density_at_proposed_sample,
			Float& alpha)
		{
			return m_sampler.end_iteration(target_density_at_current_sample, target_density_at_proposed_sample, m_g, alpha);
		}

		Float Get1D() { return m_sampler(m_g); }

	private:
		sholl::local_exploration_sampler<Float, 3> m_sampler;
	}; // class metropolis_restore_sampler
} // namespace pbrt


#endif // !HPP_SHOLL_SAMPLER_LOCAL_EXPLORATION_SAMPLER_INCLUDED
