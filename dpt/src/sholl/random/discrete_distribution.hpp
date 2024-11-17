// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#ifndef HPP_SHOLL_RANDOM_DISCRETE_DISTRIBUTION_INCLUDED
#define HPP_SHOLL_RANDOM_DISCRETE_DISTRIBUTION_INCLUDED


#include <algorithm>
#include <cassert>
#include <execution>
#include <iterator>
#include <numeric>
#include <vector>


namespace sholl
{
	template<typename IntType = int, typename RealType = double>
	class discrete_distribution
	{
	public:
		using result_type = IntType;

		struct param_type
		{
			template<class InputIt>
			param_type(InputIt first, InputIt last)
				: pmf(first, last)
			{
				init();
			}

			void init()
			{
				if (!pmf.empty())
				{
					assert(std::all_of(std::execution::par_unseq, pmf.begin(), pmf.end(),
						[](auto const pmf) { return pmf >= 0; }));
					auto const c = std::reduce(std::execution::par_unseq, pmf.begin(), pmf.end());
					assert(c > 0);
					std::transform(std::execution::par_unseq, pmf.begin(), pmf.end(), pmf.begin(),
						[&](auto const pmf) { return pmf / c; });
				}
				else
					pmf.push_back(1);

				cdf.reserve(pmf.size());
				cdf.push_back(pmf[0]);
				for (std::size_t i = 1; i < pmf.size(); ++i)
					cdf.push_back(cdf.back() + pmf[i]);
			}

			std::vector<RealType> pmf,
				cdf;
		}; // struct param_type

		template<class InputIt>
		discrete_distribution(InputIt first, InputIt last)
			: m_param(first, last)
		{}

		result_type operator()(RealType const u) const {
			//return static_cast<result_type>(std::distance(m_param.cdf.begin(), std::upper_bound(m_param.cdf.begin(), m_param.cdf.end(), u)));
			return static_cast<result_type>(std::lower_bound(m_param.cdf.begin(), std::prev(m_param.cdf.end()), u) - m_param.cdf.begin());
		}

		param_type const& param() const noexcept {
			return m_param;
		}

	private:
		param_type m_param;
	}; // class discrete_distribution
} // namespace sholl


#endif // !HPP_SHOLL_RANDOM_DISCRETE_DISTRIBUTION_INCLUDED
