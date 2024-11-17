// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#ifndef HPP_SHOLL_SAMPLER_SAMPLER_INCLUDED
#define HPP_SHOLL_SAMPLER_SAMPLER_INCLUDED


#include <random>

#include <pbrt/base/sampler.h>


namespace pbrt
{
	template<class T>
	class sampler
	{
	public:
		sampler(int const sample_count_per_pixel, std::mt19937& g)
			: m_sample_count_per_pixel(sample_count_per_pixel),
			  m_g(g)
		{}

		Float Get1D() {
			return static_cast<T*>(this)->Get1D();
		}

		Point2f Get2D() { return { Get1D(), Get1D() }; }
		Point2f GetPixel2D() { return Get2D(); }

		int SamplesPerPixel() const noexcept {
			return m_sample_count_per_pixel;
		}

		Sampler Clone(Allocator = {})
		{
			LOG_FATAL("sampler::Clone() is not implemented");
			return {};
		}

		void StartPixelSample(Point2i, int, int = 0) {
			LOG_FATAL("sampler::StartPixelSample() is not implemented");
		}

		std::string ToString() const
		{
			LOG_FATAL("sampler::ToString() is not implemented");
			return {};
		}

	protected:
		int const m_sample_count_per_pixel;
		std::mt19937& m_g;
	}; // class sampler
} // namespace pbrt


#endif // !HPP_SHOLL_SAMPLER_SAMPLER_INCLUDED
