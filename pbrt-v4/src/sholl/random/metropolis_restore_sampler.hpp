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


#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <execution>
#include <random>
#include <utility>
#include <vector>

#include <sholl/random/wrapped_normal_distribution.hpp>
#include <sholl/utility/restorable.hpp>


namespace sholl
{
	template<std::floating_point RealType = double, std::size_t StreamCount = 1>
	class metropolis_restore_sampler
	{
	public:
		using result_type = RealType;

		metropolis_restore_sampler(
			std::array<std::vector<restorable<pbrt::Float>>, StreamCount>& regeneration_distribution_sample,
			result_type const sigma)
			: m_x(regeneration_distribution_sample),
			  m_w{ sigma },
			  m_c0(0),
			  m_tau1_distribution{ 1 }
		{
			assert(sigma > 0);
		}
		metropolis_restore_sampler(
			std::array<std::vector<restorable<pbrt::Float>>, StreamCount>& regeneration_distribution_sample,
			result_type const regeneration_distribution_normalization_constant,
			result_type const target_distribution_normalization_constant,
			result_type const expected_lifetime,
			result_type const sigma)
			: m_x(regeneration_distribution_sample),
			  m_c0(target_distribution_normalization_constant / (regeneration_distribution_normalization_constant * expected_lifetime)),
			  m_w{ sigma },
			  m_tau1_distribution{ 1 }
		{
			assert(regeneration_distribution_normalization_constant > 0);
			assert(target_distribution_normalization_constant > 0);
			assert(expected_lifetime > 0);
			assert(sigma > 0);
		}

		void initialize(result_type const regeneration_distribution_normalization_constant,
			result_type const target_distribution_normalization_constant,
			result_type const expected_lifetime)
		{
			assert(regeneration_distribution_normalization_constant > 0);
			assert(target_distribution_normalization_constant > 0);
			assert(expected_lifetime > 0);

			const_cast<result_type&>(m_c0) = target_distribution_normalization_constant / (regeneration_distribution_normalization_constant * expected_lifetime);
		}

		template<class Generator>
		[[nodiscard]] bool try_begin_iteration(
			result_type const regeneration_density_at_current_sample,
			result_type const target_density_at_current_sample,
			Generator& g,
			result_type& tau,
			result_type& lambda)
		{
#ifdef _DEBUG
			assert(!m_iteration_started);
			m_iteration_started = true;
#endif // _DEBUG

			std::exponential_distribution<double> tau2_distribution{ m_c0 * regeneration_density_at_current_sample / target_density_at_current_sample };
			lambda = m_tau1_distribution.lambda() + tau2_distribution.lambda();

			result_type const tau1 = static_cast<result_type>(m_tau1_distribution(g)),
				tau2 = static_cast<result_type>(tau2_distribution(g));

			if (tau1 < tau2) [[likely]]
			{
				++m_current_iteration;
				tau = tau1;
				return true;
			}

#ifdef _DEBUG
			m_iteration_started = false;
			m_stream_started = false;
#endif // _DEBUG

			tau = tau2;
			return false;
		}

		void begin_stream(std::size_t const stream_index, std::size_t const off = 0)
		{
#ifdef _DEBUG
			assert(stream_index < StreamCount);
			m_stream_started = true;
#endif // _DEBUG

			m_stream_index = stream_index;
			m_sample_index[stream_index] = off;
		}

		template<class Generator>
		result_type const& operator()(Generator& g)
		{
			assert(m_stream_started);

			if (m_sample_index[m_stream_index] >= m_x[m_stream_index].size())
				m_x[m_stream_index].emplace_back(m_u(g));
			auto& x = m_x[m_stream_index][m_sample_index[m_stream_index]];

			x.backup();
			x.value = m_w(x.value, g, m_current_iteration - x.last_modification_iteration);
			x.last_modification_iteration = m_current_iteration;

			++m_sample_index[m_stream_index];
			return x.value;
		}

		template<class Generator>
		bool end_iteration(
			result_type const target_density_at_current_sample,
			result_type const target_density_at_proposed_sample,
			Generator& g,
			result_type& alpha)
		{
#ifdef _DEBUG
			assert(m_iteration_started);
			m_iteration_started = false;
			m_stream_started = false;
#endif // _DEBUG

			assert(target_density_at_current_sample > 0);
			alpha = std::min<result_type>(
				1, target_density_at_proposed_sample / target_density_at_current_sample);

			if (m_u(g) < alpha)
				return true;

			for (std::size_t stream_index = 0; stream_index <= m_stream_index; ++stream_index)
				std::for_each_n(std::execution::par_unseq, m_x[stream_index].begin(), m_sample_index[stream_index], [](auto& x) { x.restore(); });
			--m_current_iteration;
			return false;
		}

	private:
		result_type const m_c0;
		std::exponential_distribution<double> m_tau1_distribution;

		std::size_t m_current_iteration{},
			m_stream_index = 0;

		std::array<std::size_t, StreamCount> m_sample_index{};
		std::array<std::vector<restorable<pbrt::Float>>, StreamCount>& m_x;

		std::uniform_real_distribution<result_type> m_u;
		wrapped_normal_distribution<result_type> m_w;

#ifdef _DEBUG
		bool m_iteration_started = false,
			m_stream_started = false;
#endif // _DEBUG
	}; // class metropolis_restore_sampler
} // namespace sholl


#endif // !HPP_SHOLL_RANDOM_METROPOLIS_RESTORE_SAMPLER_INCLUDED
