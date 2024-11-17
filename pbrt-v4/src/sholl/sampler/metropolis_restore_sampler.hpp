// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#ifndef HPP_SHOLL_SAMPLER_METROPOLIS_RESTORE_SAMPLER_INCLUDED
#define HPP_SHOLL_SAMPLER_METROPOLIS_RESTORE_SAMPLER_INCLUDED


#include "sampler.hpp"

#include <sholl/random/metropolis_restore_sampler.hpp>


namespace pbrt
{
	class metropolis_restore_sampler
		: public sampler<metropolis_restore_sampler>
	{
	public:
		metropolis_restore_sampler(
			int const sample_count_per_pixel,
			std::mt19937& g,
			std::array<std::vector<sholl::restorable<Float>>, 3>& regeneration_distribution_sample,
			Float const sigma)
			: sampler{ sample_count_per_pixel, g },
			m_sampler(regeneration_distribution_sample, sigma)
		{}
		metropolis_restore_sampler(
			int const sample_count_per_pixel,
			std::mt19937& g,
			std::array<std::vector<sholl::restorable<Float>>, 3>& regeneration_distribution_sample,
			Float const regeneration_distribution_normalization_constant,
			Float const target_distribution_normalization_constant,
			Float const expected_lifetime,
			Float const sigma)
			: sampler{ sample_count_per_pixel, g },
			m_sampler(regeneration_distribution_sample, regeneration_distribution_normalization_constant, target_distribution_normalization_constant, expected_lifetime, sigma)
		{}

		void initialize(Float const regeneration_distribution_normalization_constant,
			Float const target_distribution_normalization_constant,
			Float const expected_lifetime) {
			m_sampler.initialize(regeneration_distribution_normalization_constant, target_distribution_normalization_constant, expected_lifetime);
		}

		bool try_begin_iteration(
			Float const regeneration_density_at_current_sample,
			Float const target_density_at_current_sample,
			Float& tau,
			Float& lambda)
		{
			return m_sampler.try_begin_iteration(regeneration_density_at_current_sample, target_density_at_current_sample, m_g, tau, lambda);
		}

		void begin_stream(std::size_t const stream_index, std::size_t const off = 0) {
			m_sampler.begin_stream(stream_index, off);
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
		sholl::metropolis_restore_sampler<Float, 3> m_sampler;
	}; // class metropolis_restore_sampler
} // namespace pbrt


#endif // !HPP_SHOLL_SAMPLER_METROPOLIS_RESTORE_SAMPLER_INCLUDED
