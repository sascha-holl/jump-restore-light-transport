// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#include "metropolis_restore_integrator.hpp"


namespace pbrt
{
	int GenerateCameraSubpath(const Integrator& integrator, const RayDifferential& ray,
		SampledWavelengths& lambda, Sampler sampler,
		ScratchBuffer& scratchBuffer, int maxDepth, Camera camera,
		Vertex* path, bool regularize);
	SampledSpectrum ConnectBDPT(const Integrator& integrator, SampledWavelengths& lambda,
		Vertex* lightVertices, Vertex* cameraVertices, int s, int t,
		LightSampler lightSampler, Camera camera, Sampler sampler,
		pstd::optional<Point2f>* pRaster,
		Float* misWeightPtr, bool apply_multiple_importance_sampling_weighting);

	SampledSpectrum metropolis_restore_integrator::sample_path(std::size_t const s, std::size_t const t, metropolis_restore_sampler& sampler,
		ScratchBuffer& scratch_buffer, Point2f& raster_point, SampledWavelengths& lambda) const
	{
		raster_point = m_film_sample_space_bounding_box.Lerp(sampler.GetPixel2D());
		lambda = m_camera.GetFilm().SampleWavelengths(sampler.Get1D());

		CameraSample camera_sample{};
		camera_sample.pFilm = raster_point;
		RayDifferential const r1{ m_camera.GenerateRay(camera_sample, lambda)->ray };

		thread_local std::vector<Vertex> x{ m_k_max + 1 };
		x.resize(t);

		if (GenerateCameraSubpath(*this, r1, lambda, &sampler, scratch_buffer, t, m_camera, x.data(), false) == t)
		{
			thread_local std::vector<Vertex> y{ m_k_max + 1 };
			y.resize(s);

			sampler.begin_stream(1);
			if (GenerateLightSubpath(*this, lambda, &sampler, m_camera, scratch_buffer, s, x[0].time(), m_light_sampler, y.data(), false) == s)
			{
				sampler.begin_stream(2);
				pstd::optional<Point2f> raster_point_new;
				auto const contribution = ConnectBDPT(*this, lambda, y.data(), x.data(), s, t, m_light_sampler, m_camera, &sampler, &raster_point_new);

				if (raster_point_new)
					raster_point = *raster_point_new;

				return contribution;
			}
		}
		return {};
	}

	SampledSpectrum metropolis_restore_integrator::sample_path(std::size_t const depth, metropolis_restore_sampler& sampler,
		ScratchBuffer& scratch_buffer, Point2f& raster_point, SampledWavelengths& lambda) const
	{
		std::size_t s, t,
			strategy_count;
		Float const u = sampler.Get1D();
		if (depth > 0) [[likely]]
		{
			strategy_count = depth + 2;
			s = std::min<std::size_t>(u * strategy_count, strategy_count - 1);
			t = strategy_count - s;
		}
		else [[unlikely]]
		{
			strategy_count = 1;
			s = 0;
			t = 2;
		}

		auto const contribution = sample_path(s, t, sampler, scratch_buffer, raster_point, lambda);
		return strategy_count * contribution;
	}

	SampledSpectrum metropolis_restore_integrator::sample_path(std::size_t const s, std::size_t const t, uniform_sampler& sampler,
		ScratchBuffer& scratch_buffer, Point2f& raster_point, SampledWavelengths& lambda) const
	{
		raster_point = m_film_sample_space_bounding_box.Lerp(sampler.GetPixel2D());
		lambda = m_camera.GetFilm().SampleWavelengths(sampler.Get1D());

		CameraSample camera_sample{};
		camera_sample.pFilm = raster_point;
		RayDifferential const r1{ m_camera.GenerateRay(camera_sample, lambda)->ray };

		thread_local std::vector<Vertex> x{ m_k_max + 1 };
		x.resize(t);

		if (GenerateCameraSubpath(*this, r1, lambda, &sampler, scratch_buffer, t, m_camera, x.data(), false) == t)
		{
			thread_local std::vector<Vertex> y{ m_k_max + 1 };
			y.resize(s);

			sampler.begin_stream(1);
			if (GenerateLightSubpath(*this, lambda, &sampler, m_camera, scratch_buffer, s, x[0].time(), m_light_sampler, y.data(), false) == s)
			{
				sampler.begin_stream(2);
				pstd::optional<Point2f> raster_point_new;
				auto const contribution = ConnectBDPT(*this, lambda, y.data(), x.data(), s, t, m_light_sampler, m_camera, &sampler, &raster_point_new);

				if (raster_point_new)
					raster_point = *raster_point_new;

				return contribution;
			}
		}
		return {};
	}

	SampledSpectrum metropolis_restore_integrator::sample_path(std::size_t const depth, uniform_sampler& sampler,
		ScratchBuffer& scratch_buffer, Point2f& raster_point, SampledWavelengths& lambda) const
	{
		std::size_t s, t,
			strategy_count;
		Float const u = sampler.Get1D();
		if (depth > 0) [[likely]]
		{
			strategy_count = depth + 2;
			s = std::min<std::size_t>(u * strategy_count, strategy_count - 1);
			t = strategy_count - s;
		}
		else [[unlikely]]
		{
			strategy_count = 1;
			s = 0;
			t = 2;
		}

		auto const contribution = sample_path(s, t, sampler, scratch_buffer, raster_point, lambda);
		return strategy_count * contribution;
	}
} // namespace pbrt
