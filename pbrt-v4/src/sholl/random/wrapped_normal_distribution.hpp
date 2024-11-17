// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#ifndef HPP_SHOLL_RANDOM_WRAPPED_NORMAL_DISTRIBUTION_INCLUDED
#define HPP_SHOLL_RANDOM_WRAPPED_NORMAL_DISTRIBUTION_INCLUDED


#include <cassert>
#include <random>


namespace sholl
{
	template<typename RealType = double>
	class wrapped_normal_distribution
	{
	public:
		using result_type = RealType;

		explicit wrapped_normal_distribution(result_type const sigma)
			: m_sigma(sigma)
		{
			assert(sigma > 0);
		}

		template<class Generator>
		result_type operator()(result_type x, Generator& g) const
		{
			x = std::normal_distribution<result_type>{ x, m_sigma }(g);
			return x - std::floor(x);
		}

		template<class Generator>
		result_type operator()(result_type x, Generator& g, std::size_t n) const
		{
			if (n > 0) [[likely]]
			{
				x = std::normal_distribution<result_type>{ x, static_cast<result_type>(std::sqrt(n)) * m_sigma }(g);
				return x - std::floor(x);
			}
			return x;
		}

		constexpr result_type min() const noexcept { return 0; }
		constexpr result_type max() const noexcept { return 1; }

		constexpr result_type const& stddev() const noexcept { return m_sigma; }

	private:
		result_type const m_sigma;
	}; // class wrapped_normal_distribution
} // namespace sholl


#endif // !HPP_SHOLL_RANDOM_WRAPPED_NORMAL_DISTRIBUTION_INCLUDED
