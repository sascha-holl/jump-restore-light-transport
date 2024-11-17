// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#ifndef HPP_SHOLL_RANDOM_METROPOLIS_RESTORE_SAMPLER_INCLUDED
#define HPP_SHOLL_RANDOM_METROPOLIS_RESTORE_SAMPLER_INCLUDED


#include <cassert>
#include <concepts>
#include <random>


namespace sholl
{
	template<std::floating_point RealType = double>
	class metropolis_restore_sampler
	{
	public:
		using result_type = RealType;

		metropolis_restore_sampler(
			result_type const regeneration_distribution_normalization_constant,
			result_type const target_distribution_normalization_constant,
			result_type const expected_lifetime)
			: m_c0(target_distribution_normalization_constant / (regeneration_distribution_normalization_constant * expected_lifetime)),
			  m_tau1_distribution{ 1 }
		{
			assert(regeneration_distribution_normalization_constant > 0);
			assert(target_distribution_normalization_constant > 0);
			assert(expected_lifetime > 0);
		}

		template<class Generator>
		[[nodiscard]] bool try_begin_iteration(
			result_type const regeneration_density_at_current_sample,
			result_type const target_density_at_current_sample,
			Generator& g,
			result_type& tau,
			result_type& lambda)
		{
			assert(!m_iteration_started);
#ifdef _DEBUG
			m_iteration_started = true;
#endif // _DEBUG

			std::exponential_distribution<double> tau2_distribution{ m_c0 * regeneration_density_at_current_sample / target_density_at_current_sample };
			lambda = static_cast<result_type>(m_tau1_distribution.lambda() + tau2_distribution.lambda());

			result_type const tau1 = static_cast<result_type>(m_tau1_distribution(g)),
				tau2 = static_cast<result_type>(tau2_distribution(g));

			if (tau1 < tau2) [[likely]]
			{
				tau = tau1;
				return true;
			}

			tau = tau2;
			return false;
		}

		template<class Generator>
		bool end_iteration(result_type const& alpha, Generator& g)
		{
			assert(m_iteration_started);
#ifdef _DEBUG
			m_iteration_started = false;
#endif // _DEBUG

			assert(alpha >= 0 && alpha <= 1);
			if (m_u(g) < alpha)
				return true;
			return false;
		}

	private:
		result_type const m_c0;
		std::exponential_distribution<double> m_tau1_distribution;

		std::uniform_real_distribution<result_type> m_u;

#ifdef _DEBUG
		bool m_iteration_started = false;
#endif // _DEBUG
	}; // class metropolis_restore_sampler
} // namespace sholl


#endif // !HPP_SHOLL_RANDOM_METROPOLIS_RESTORE_SAMPLER_INCLUDED
