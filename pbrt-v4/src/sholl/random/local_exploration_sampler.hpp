// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#ifndef HPP_SAMPLER_LOCAL_EXPLORATION_SAMPLER_INCLUDED
#define HPP_SAMPLER_LOCAL_EXPLORATION_SAMPLER_INCLUDED


#include <algorithm>
#include <array>
#include <cassert>
#include <execution>
#include <random>
#include <utility>
#include <vector>

#include <sholl/random/wrapped_normal_distribution.hpp>
#include <sholl/utility/restorable.hpp>


namespace sholl
{
	template<typename RealType = double, std::size_t StreamCount = 1>
	class local_exploration_sampler
	{
	public:
		using result_type = RealType;

		local_exploration_sampler(
			result_type const large_step_probability,
			result_type const sigma)
			: m_large_step_probability(large_step_probability),
			  m_w{ sigma }
		{
			assert(large_step_probability >= 0 && large_step_probability <= 1);
			assert(sigma > 0);
		}

		template<class Generator>
		bool begin_iteration(Generator& g)
		{
			assert(!m_iteration_started);
#ifdef _DEBUG
			m_iteration_started = true;
#endif // _DEBUG

			++m_current_iteration;
			return m_large_step = m_u(g) < m_large_step_probability;
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

			if (m_current_iteration > 0)
			{
				if (!m_large_step)
				{
					if (x.last_modification_iteration < m_last_large_step_iteration)
					{
						x.value = m_u(g);
						x.last_modification_iteration = m_last_large_step_iteration;
					}

					x.backup();
					x.value = m_w(x.value, g, m_current_iteration - x.last_modification_iteration);
				}
				else
				{
					x.backup();
					x.value = m_u(g);
				}

				x.last_modification_iteration = m_current_iteration;
			}

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
			assert(m_iteration_started);
#ifdef _DEBUG
			m_iteration_started = false;
			m_stream_started = false;
#endif // _DEBUG

			alpha = target_density_at_current_sample > 0 ? 
				std::min<result_type>(1, target_density_at_proposed_sample / target_density_at_current_sample) : 1;

			if (m_u(g) < alpha)
			{
				if (m_large_step)
					m_last_large_step_iteration = m_current_iteration;
				return true;
			}

			for (std::size_t stream_index = 0; stream_index <= m_stream_index; ++stream_index)
				std::for_each_n(std::execution::par_unseq, m_x[stream_index].begin(), m_sample_index[stream_index], [](auto& x) { x.restore(); });
			--m_current_iteration;
			return false;
		}

	private:
		result_type const m_large_step_probability;
		bool m_large_step = true;

		std::size_t m_current_iteration{},
			m_last_large_step_iteration{},
			m_stream_index = 0;

		std::array<std::size_t, StreamCount> m_sample_index{};
		std::array<std::vector<restorable<pbrt::Float>>, StreamCount> m_x;

		std::uniform_real_distribution<result_type> m_u;
		wrapped_normal_distribution<result_type> m_w;

#ifdef _DEBUG
		bool m_iteration_started = false,
			m_stream_started = false;
#endif // _DEBUG
	}; // class local_exploration_sampler
} // namespace sholl


#endif // !HPP_SAMPLER_LOCAL_EXPLORATION_SAMPLER_INCLUDED
