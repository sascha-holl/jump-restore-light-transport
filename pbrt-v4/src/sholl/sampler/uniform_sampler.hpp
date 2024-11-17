// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#ifndef HPP_SHOLL_SAMPLER_UNIFORM_SAMPLER_INCLUDED
#define HPP_SHOLL_SAMPLER_UNIFORM_SAMPLER_INCLUDED


#include <array>
#include <vector>

#include "sampler.hpp"

#include <sholl/utility/restorable.hpp>


namespace pbrt
{
	class uniform_sampler
		: public sampler<uniform_sampler>
	{
	public:
		uniform_sampler(int const sample_count_per_pixel, std::mt19937& g, std::array<std::vector<sholl::restorable<pbrt::Float>>, 3>& x)
			: sampler{ sample_count_per_pixel, g },
			  m_x(x)
		{}

		void begin_stream(std::size_t const stream_index)
		{
			assert(stream_index < 3);
			m_stream_index = stream_index;
		}

		Float Get1D()
		{
			m_x[m_stream_index].emplace_back(m_u(m_g));
			return m_x[m_stream_index].back().value;
		}

	private:
		std::size_t m_stream_index;

		std::uniform_real_distribution<Float> m_u;
		std::array<std::vector<sholl::restorable<pbrt::Float>>, 3>& m_x;
	}; // class uniform_sampler
} // namespace pbrt


#endif // !HPP_SHOLL_SAMPLER_UNIFORM_SAMPLER_INCLUDED
