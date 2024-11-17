// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#ifndef HPP_SHOLL_INTEGRATOR_METROPOLIS_RESTORE_INTEGRATOR_INCLUDED
#define HPP_SHOLL_INTEGRATOR_METROPOLIS_RESTORE_INTEGRATOR_INCLUDED


#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <execution>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <boost/iterator/counting_iterator.hpp>
#include <Eigen/Eigen>

#include <pbrt/cpu/integrators.h>

#include <sholl/atomic/replace_if.hpp>
#include <sholl/random/discrete_distribution.hpp>
#include <sholl/sampler/metropolis_restore_sampler.hpp>
#include <sholl/sampler/uniform_sampler.hpp>


namespace pbrt
{
	class metropolis_restore_integrator
		: public Integrator
	{
	public:
		static std::unique_ptr<metropolis_restore_integrator> Create(ParameterDictionary const& parameter_dictionary,
			Primitive aggregate, std::vector<Light> const& light, Camera camera, FileLoc const*)
		{
			if (!camera.Is<PerspectiveCamera>())
				ErrorExit("Only the \"perspective\" camera is currently supported with the \"restore\" integrator.");

			int const k_max = parameter_dictionary.GetOneInt("k_max", 10),
				bootstrap_sample_count = parameter_dictionary.GetOneInt("bootstrap_sample_count", 10000),
				sample_count_per_pixel = parameter_dictionary.GetOneInt("sample_count_per_pixel", 1),
				realization_count = parameter_dictionary.GetOneInt("realization_count", 1),
				fps = parameter_dictionary.GetOneInt("fps", 1);
			Float const expected_lifetime = parameter_dictionary.GetOneFloat("expected_lifetime", 1),
				sigma = parameter_dictionary.GetOneFloat("sigma", .01),
				computation_time = parameter_dictionary.GetOneFloat("computation_time", 0);
			bool const rao_blackwellize = parameter_dictionary.GetOneBool("rao_blackwellize", true),
				compute_mse = parameter_dictionary.GetOneBool("compute_mse", false);

			if (k_max < 1)
				ErrorExit("k_max < 1");
			if (bootstrap_sample_count < 1)
				ErrorExit("bootstrap_sample_count < 1");
			if (sample_count_per_pixel < 1)
				ErrorExit("sample_count_per_pixel < 1");
			if (fps < 1)
				ErrorExit("fps < 1");
			if (expected_lifetime < 0)
				ErrorExit("expected_lifetime < 0");
			if (sigma <= 0)
				ErrorExit("sigma <= 0");
			if (computation_time < 0)
				ErrorExit("computation_time < 0");
			if (realization_count < 1)
				ErrorExit("realization_count < 1");

			return std::make_unique<metropolis_restore_integrator>(aggregate, light, camera,
				static_cast<std::size_t>(k_max),
				static_cast<std::size_t>(bootstrap_sample_count),
				sample_count_per_pixel,
				expected_lifetime,
				sigma,
				rao_blackwellize,
				compute_mse,
				realization_count,
				std::chrono::duration<Float>(computation_time),
				std::chrono::duration<Float>(Float{ 1 } / fps)
			);
		}

		metropolis_restore_integrator(Primitive aggregate, std::vector<Light> const& light, Camera camera,
			std::size_t const k_max, std::size_t const bootstrap_sample_count, int const sample_count_per_pixel, Float const expected_lifetime, Float const sigma, bool const rao_blackwellize,
			bool const compute_mse, std::size_t const realization_count,
			std::chrono::duration<Float, std::nano> const& computation_time, std::chrono::duration<Float, std::nano> const& frame_period
		)
			: Integrator(aggregate, light),
			  m_camera(camera),
			  m_film(camera.GetFilm()),
			  m_film_sample_space_bounding_box(camera.GetFilm().SampleBounds()),
			  m_light_sampler(new PowerLightSampler(light, Allocator())),
			  m_k_max(k_max),
			  m_bootstrap_sample_count(bootstrap_sample_count),
			  m_film_sample_space_width(static_cast<std::size_t>(m_film_sample_space_bounding_box.pMax.x - m_film_sample_space_bounding_box.pMin.x)),
			  m_film_sample_space_height(static_cast<std::size_t>(m_film_sample_space_bounding_box.pMax.y - m_film_sample_space_bounding_box.pMin.y)),
			  m_pixel_count(static_cast<std::size_t>(m_film_sample_space_bounding_box.Area())),
			  m_bootstrap_sample_count_per_pixel{ static_cast<int>(std::ceil(static_cast<Float>(k_max * m_bootstrap_sample_count) / m_pixel_count)) },
			  m_sample_count_per_pixel(sample_count_per_pixel),
			  m_sample_count(m_pixel_count * sample_count_per_pixel),
			  m_expected_lifetime(expected_lifetime),
			  m_sigma(sigma),
			  m_rao_blackwellize(rao_blackwellize),
			  m_compute_mse(compute_mse),
			  m_realization_count(realization_count),
			  m_multiplex_weight_correction{ (static_cast<Float>(k_max * k_max + 3 * k_max) / 2 - 1) / k_max },
#ifdef _DEBUG
			  m_thread_count(1),
#else
			  m_thread_count(std::thread::hardware_concurrency()),
#endif // _DEBUG || SHOLL_OUTPUT_HISTOGRAM
			  m_thread_pool(m_thread_count - 1),
			  m_thread_data(m_thread_count, m_pixel_count),
			  m_computation_time(computation_time),
			  m_frame_period(frame_period)
		{}

		void Render() override
		{
			std::optional<Image> reference_image = Image{ PixelFormat::Float, static_cast<Point2i>(m_film_sample_space_bounding_box.Diagonal()), { "R", "G", "B" } };

			if (!m_compute_mse)
			{
				if (m_rao_blackwellize)
					render<true>(m_film.GetFilename(), reference_image);
				else
					render<false>(m_film.GetFilename(), reference_image);
			}
			else
			{
				if (!reference_image)
					ErrorExit("cannot compute mse without reference image");

				if (m_rao_blackwellize)
					compute_equal_time_mse<true>(m_film.GetFilename(), *reference_image);
				else
					compute_equal_time_mse<false>(m_film.GetFilename(), *reference_image);
			}
		}

		std::string ToString() const override { return {}; }

	private:
		struct state_base
		{
			Point2f raster_point;
			SampledWavelengths lambda;

			SampledSpectrum contribution;
			Float target_density;
		}; // struct state_base

		struct rao_blackwellization_state
			: public state_base
		{
			rao_blackwellization_state() = default;
			rao_blackwellization_state(state_base const& other)
				: state_base(other)
			{}

			Float weight = 1,
				xi = 1;
		}; // struct rao_blackwellization_state

		template<bool RaoBlackwellize>
		using state = std::conditional_t<RaoBlackwellize, rao_blackwellization_state, state_base>;

		struct bootstrap_result
		{
			std::array<std::size_t, 3> d_max;
			Float c;
			sholl::discrete_distribution<std::size_t> path_depth_distribution;
		};

		struct render_result
		{
			std::size_t generated_sample_count,
				generated_tour_count;
			double tau;
			std::chrono::duration<Float, std::nano> elapsed_time;
		};

		template<bool RaoBlackwellize>
		void render(std::string const& filename, std::optional<Image> const& reference_image = std::nullopt)
		{
			bool const equal_time_comparison = m_computation_time > std::chrono::duration<Float, std::nano>::zero();
			if (equal_time_comparison)
				std::cout << "equal time comparison for metropolis restore\n";
			else
				std::cout << "equal sample count comparison for metropolis restore\n";

			std::ofstream log(filename + ".log", std::ios::binary);

			bootstrap_result const bootstrap_result = bootstrap();
			if (bootstrap_result.c == 0)
			{
				std::cout << "black image\n";
				return;
			}

			std::cout << "rendering\n" << std::flush;

			init_thead_data(bootstrap_result);

			Float min_mse = std::numeric_limits<Float>::infinity(),
				max_mse = -min_mse,
				average_mse{},
				average_generated_sample_count{},
				average_generated_tour_count{},
				average_tau{};
			std::chrono::duration<Float, std::nano> average_rendering_time{};
			for (std::size_t realization = 0; realization < m_realization_count; ++realization)
			{
				std::vector<RGB> unscaled_image(m_pixel_count);

				render_result render_result;
				if (equal_time_comparison)
					render_result = render_for<RaoBlackwellize>(bootstrap_result, m_computation_time, unscaled_image);
				else
					render_result = render<RaoBlackwellize>(bootstrap_result, m_sample_count, unscaled_image);

				average_generated_sample_count += render_result.generated_sample_count;
				average_generated_tour_count += render_result.generated_tour_count;
				average_tau += render_result.tau;
				average_rendering_time += render_result.elapsed_time;

				Image image{ PixelFormat::Float, static_cast<Point2i>(m_film_sample_space_bounding_box.Diagonal()), { "R", "G", "B" } };

				Float const s = bootstrap_result.c * m_pixel_count / render_result.tau;
				std::for_each_n(std::execution::par_unseq, boost::counting_iterator<std::size_t>(0), m_pixel_count, [&](std::size_t const i)
				{
					image.SetChannels(Point2i{ static_cast<int>(i % m_film_sample_space_width), static_cast<int>(i / m_film_sample_space_width) }, {
						s * unscaled_image[i][0],
						s * unscaled_image[i][1],
						s * unscaled_image[i][2]
					});
				});

				Float const mse = image.MSE(image.AllChannelsDesc(), *reference_image).Average();
				average_mse += mse;

				bool write_image = false;
				if (mse < min_mse)
				{
					min_mse = mse;
					write_image = true;
				}
				if (mse > max_mse)
					max_mse = mse;

				if (write_image)
				{
					ImageMetadata image_metadata;
					m_camera.InitMetadata(&image_metadata);

					using namespace std::chrono_literals;
					image_metadata.renderTimeSeconds = static_cast<Float>(render_result.elapsed_time.count()) / std::chrono::duration<Float, std::nano>{ 1s }.count();
					image_metadata.samplesPerPixel = static_cast<int>(std::floor(static_cast<Float>(render_result.generated_sample_count)) / m_pixel_count);
					image_metadata.pixelBounds = static_cast<Bounds2i>(m_film_sample_space_bounding_box);
					image_metadata.fullResolution = m_film.FullResolution();

					image.Write(filename, image_metadata);
				}
			}

			average_mse /= m_realization_count;
			average_generated_sample_count /= m_realization_count;
			average_generated_tour_count /= m_realization_count;
			average_tau /= m_realization_count;
			average_rendering_time /= m_realization_count;

			//std::cout << "average generated_tour_count: " << average_generated_tour_count << std::endl
			//	<< "average generated_tour_count: " << average_generated_sample_count / average_generated_tour_count << std::endl;

			using namespace std::chrono_literals;
			std::cout << "average_generated_sample_count_per_pixel = " << average_generated_sample_count / m_pixel_count << std::endl
				<< "min_mse = " << min_mse << std::endl
				<< "max_mse = " << max_mse << std::endl
				<< "average_mse = " << average_mse << std::endl
				<< "average_rendering_time = " << static_cast<Float>(average_rendering_time.count()) / std::chrono::duration<Float, std::nano>{ 1s }.count() << std::endl;
			log << "average_generated_sample_count_per_pixel = " << average_generated_sample_count / m_pixel_count << std::endl
				<< "min_mse = " << min_mse << std::endl
				<< "max_mse = " << max_mse << std::endl
				<< "average_mse = " << average_mse << std::endl
				<< "average_rendering_time = " << static_cast<Float>(average_rendering_time.count()) / std::chrono::duration<Float, std::nano>{ 1s }.count() << std::endl;
		}

		template<bool RaoBlackwellize>
		void compute_mse(std::string const& filename, Image const& reference_image)
		{
			std::cout << "equal sample count comparison for metropolis restore\n";

			std::ofstream log(filename + ".log", std::ios::binary);
			std::ofstream mse_out(filename + ".mse", std::ios::binary);
			mse_out << std::fixed << std::setprecision(4);

			bootstrap_result const bootstrap_result = bootstrap();
			if (bootstrap_result.c == 0)
			{
				std::cout << "black image\n";
				return;
			}

			init_thead_data(bootstrap_result);

			Float const scale = m_pixel_count * bootstrap_result.c,
				scale_over_realization_count = scale / m_realization_count;

			std::vector<std::size_t> generated_sample_count(m_realization_count);
			std::vector<double> tau(m_realization_count);
			std::vector<std::chrono::duration<Float, std::nano>> elapsed_time(m_realization_count);
			std::vector<std::vector<RGB>> unscaled_image(m_realization_count, std::vector<RGB>(m_pixel_count));

			std::size_t const sample_count_per_pixel_k = static_cast<std::size_t>(std::ceil(std::log2(static_cast<Float>(m_sample_count_per_pixel))));
			for (std::size_t k = 0; k <= sample_count_per_pixel_k; ++k)
			{
				std::vector<Image> image(m_realization_count, Image{ PixelFormat::Float, static_cast<Point2i>(m_film_sample_space_bounding_box.Diagonal()), { "R", "G", "B" } });
				std::vector<RGB> unscaled_image_empirical_mean(m_pixel_count);

				std::size_t const sample_count_per_pixel = static_cast<std::size_t>(1 << k);
				//mse_out << sample_count_per_pixel << '\t';

				std::size_t const sample_count = m_pixel_count * sample_count_per_pixel;

				Float generated_sample_count_per_pixel_average{},
					rendering_time_average{},
					mse_average{},
					min_mse = std::numeric_limits<Float>::infinity();

				for (std::size_t realization = 0; realization < m_realization_count; ++realization)
				{
					std::stringstream ss;
					ss << "rendering 2^" << k << "spp (realization " << realization << ")\n" << std::flush;
					log << ss.str();
					std::cout << ss.str();

					render_result const render_result = render<RaoBlackwellize>(bootstrap_result.c, bootstrap_result.path_depth_distribution, sample_count - generated_sample_count[realization], unscaled_image[realization]);

					generated_sample_count[realization] += render_result.generated_sample_count;
					tau[realization] += render_result.tau;
					elapsed_time[realization] += render_result.elapsed_time;

					for (std::size_t i = 0; i < m_pixel_count; ++i)
					{
						Float const
							r = unscaled_image[realization][i][0] / tau[realization],
							g = unscaled_image[realization][i][1] / tau[realization],
							b = unscaled_image[realization][i][2] / tau[realization];

						image[realization].SetChannels(Point2i{ static_cast<int>(i % m_film_sample_space_width), static_cast<int>(i / m_film_sample_space_width) }, {
							scale * r,
							scale * g,
							scale * b
						});

						unscaled_image_empirical_mean[i][0] += r;
						unscaled_image_empirical_mean[i][1] += g;
						unscaled_image_empirical_mean[i][2] += b;
					}

					Float const mse = image[realization].MSE(image[realization].AllChannelsDesc(), reference_image).Average(),
						rendering_time = std::chrono::duration_cast<std::chrono::seconds>(elapsed_time[realization]).count();

					if (mse < min_mse)
					{
						min_mse = mse;

						ImageMetadata image_metadata;
						m_camera.InitMetadata(&image_metadata);

						image_metadata.renderTimeSeconds = rendering_time;
						image_metadata.samplesPerPixel = sample_count_per_pixel;
						image_metadata.pixelBounds = static_cast<Bounds2i>(m_film_sample_space_bounding_box);
						image_metadata.fullResolution = m_film.FullResolution();

						ss.str(std::string{});
						ss << filename << '_' << k << ".exr";
						image[realization].Write(ss.str(), image_metadata);
					}

					generated_sample_count_per_pixel_average += static_cast<Float>(generated_sample_count[realization]) / m_pixel_count;
					rendering_time_average += rendering_time;
					mse_average += mse;
				}

				generated_sample_count_per_pixel_average /= m_realization_count;
				rendering_time_average /= m_realization_count;
				mse_average /= m_realization_count;

				Image image_empirical_mean{ PixelFormat::Float, static_cast<Point2i>(m_film_sample_space_bounding_box.Diagonal()), { "R", "G", "B" } };
				for (std::size_t i = 0; i < m_pixel_count; ++i)
				{
					image_empirical_mean.SetChannels(Point2i{ static_cast<int>(i % m_film_sample_space_width), static_cast<int>(i / m_film_sample_space_width) }, {
						scale_over_realization_count * unscaled_image_empirical_mean[i][0],
						scale_over_realization_count * unscaled_image_empirical_mean[i][1],
						scale_over_realization_count * unscaled_image_empirical_mean[i][2]
					});
				}

				Float empirical_variance{};
				for (std::size_t realization = 0; realization < m_realization_count; ++realization)
					empirical_variance += image[realization].MSE(image[realization].AllChannelsDesc(), image_empirical_mean).Average();
				empirical_variance /= (m_realization_count - 1);

				mse_out << generated_sample_count_per_pixel_average << '\t'
					<< rendering_time_average << '\t' 
					<< mse_average << '\t'
					<< empirical_variance << '\n' << std::flush;
			}
		}

		template<bool RaoBlackwellize>
		void compute_equal_time_mse(std::string const& filename, Image const& reference_image)
		{
			std::cout << "equal time comparison for metropolis restore\n";

			std::ofstream log(filename + ".log", std::ios::binary);
			std::ofstream mse_out(filename + ".mse", std::ios::binary);
			mse_out << std::fixed << std::setprecision(4);

			bootstrap_result const bootstrap_result = bootstrap();
			if (bootstrap_result.c == 0)
			{
				std::cout << "black image\n";
				return;
			}

			init_thead_data(bootstrap_result);

			Float const scale = bootstrap_result.c * m_pixel_count,
				scale_over_realization_count = scale / m_realization_count;

			std::vector<std::size_t> generated_sample_count(m_realization_count);
			std::vector<double> tau(m_realization_count);
			std::vector<std::chrono::duration<Float, std::nano>> computation_time(m_realization_count);
			std::vector<std::vector<RGB>> unscaled_image(m_realization_count, std::vector<RGB>(m_pixel_count));

			for (std::size_t frame = 1;; ++frame)
			{
				if (computation_time[0] >= m_computation_time)
					break;

				std::stringstream ss;
				ss << "rendering frame " << frame << std::endl;
				log << ss.str();
				std::cout << ss.str();

				std::vector<Image> image(m_realization_count, Image{ PixelFormat::Float, static_cast<Point2i>(m_film_sample_space_bounding_box.Diagonal()), { "R", "G", "B" } });
				std::vector<RGB> unscaled_image_empirical_mean(m_pixel_count);

				std::chrono::duration<Float, std::nano> rendering_time_average{};
				Float generated_sample_count_average{},
					mse_average{},
					min_mse = std::numeric_limits<Float>::infinity(),
					max_mse = -min_mse;

				for (std::size_t realization = 0; realization < m_realization_count; ++realization)
				{
					render_result const render_result = render_for<RaoBlackwellize>(bootstrap_result, frame * m_frame_period - computation_time[realization], unscaled_image[realization]);

					generated_sample_count[realization] += render_result.generated_sample_count;
					tau[realization] += render_result.tau;
					computation_time[realization] += render_result.elapsed_time;

					for (std::size_t i = 0; i < m_pixel_count; ++i)
					{
						Float const
							r = unscaled_image[realization][i][0] / tau[realization],
							g = unscaled_image[realization][i][1] / tau[realization],
							b = unscaled_image[realization][i][2] / tau[realization];

						image[realization].SetChannels(Point2i{ static_cast<int>(i % m_film_sample_space_width), static_cast<int>(i / m_film_sample_space_width) }, {
							scale * r,
							scale * g,
							scale * b
						});

						unscaled_image_empirical_mean[i][0] += r;
						unscaled_image_empirical_mean[i][1] += g;
						unscaled_image_empirical_mean[i][2] += b;
					}

					Float const mse = image[realization].MSE(image[realization].AllChannelsDesc(), reference_image).Average();

					bool write_image = false;
					if (mse < min_mse)
					{
						min_mse = mse;
						write_image = true;
					}
					if (mse > max_mse)
						max_mse = mse;

					if (write_image)
					{
						ImageMetadata image_metadata;
						m_camera.InitMetadata(&image_metadata);

						using namespace std::chrono_literals;
						image_metadata.renderTimeSeconds = static_cast<Float>(computation_time[realization].count()) / std::chrono::duration<Float, std::nano>{ 1s }.count();
						image_metadata.samplesPerPixel = static_cast<int>(std::floor(static_cast<Float>(generated_sample_count[realization])) / m_pixel_count);
						image_metadata.pixelBounds = static_cast<Bounds2i>(m_film_sample_space_bounding_box);
						image_metadata.fullResolution = m_film.FullResolution();

						ss.str(std::string{});
						ss << filename << '_' << frame << ".exr";
						image[realization].Write(ss.str(), image_metadata);
					}

					generated_sample_count_average += generated_sample_count[realization];
					rendering_time_average += computation_time[realization];
					mse_average += mse;
				}

				generated_sample_count_average /= m_realization_count;
				rendering_time_average /= m_realization_count;
				mse_average /= m_realization_count;

				Image image_empirical_mean{ PixelFormat::Float, static_cast<Point2i>(m_film_sample_space_bounding_box.Diagonal()), { "R", "G", "B" } };
				for (std::size_t i = 0; i < m_pixel_count; ++i)
				{
					image_empirical_mean.SetChannels(Point2i{ static_cast<int>(i % m_film_sample_space_width), static_cast<int>(i / m_film_sample_space_width) }, {
						scale_over_realization_count * unscaled_image_empirical_mean[i][0],
						scale_over_realization_count * unscaled_image_empirical_mean[i][1],
						scale_over_realization_count * unscaled_image_empirical_mean[i][2]
					});
				}

				Float const empirical_variance = std::accumulate(/*std::execution::par_unseq,*/ image.begin(), image.end(), Float{}, [&](auto const& acc, auto const& image) {
					return acc + image.MSE(image.AllChannelsDesc(), image_empirical_mean).Average();
				}) / (m_realization_count - 1);

				using namespace std::chrono_literals;
				mse_out << generated_sample_count_average / m_pixel_count << '\t'
					<< static_cast<Float>(rendering_time_average.count()) / std::chrono::duration<Float, std::nano>{ 1s }.count() << '\t'
					<< mse_average << '\t'
					<< empirical_variance
					<< std::endl << std::flush;
			}
		}

		bootstrap_result bootstrap() const
		{
			std::cout << "bootstrapping\n" << std::flush;

			ThreadLocal<ScratchBuffer> thread_scratch_buffer([]() { return ScratchBuffer{}; });
			
			std::array<std::atomic_size_t, 3> d_max;
			std::vector<std::vector<Float>> bootstrap_weight_per_path_length(m_k_max, std::vector<Float>(m_bootstrap_sample_count));

			auto const bootrapping_begin = std::chrono::high_resolution_clock::now();
			ParallelFor(0, static_cast<std::int64_t>(m_bootstrap_sample_count), [&](std::int64_t const first, std::int64_t const last)
            {
				ScratchBuffer& scratch_buffer = thread_scratch_buffer.Get();

				std::array<std::size_t, 3> thread_d_max{};
				thread_local std::array<std::vector<sholl::restorable<Float>>, 3> regeneration_distribution_sample;

				for (std::int64_t i = first; i < last; ++i)
                {
					for (std::size_t depth = 0; depth < m_k_max; ++depth)
					{
						regeneration_distribution_sample[0].resize(0);
						regeneration_distribution_sample[1].resize(0);
						regeneration_distribution_sample[2].resize(0);

						std::seed_seq seed{ static_cast<std::size_t>(i), depth };
						std::mt19937 g{ seed };
						uniform_sampler sampler{ m_bootstrap_sample_count_per_pixel, g, regeneration_distribution_sample };
						sampler.begin_stream(0);

						Point2f raster_point;
						SampledWavelengths lambda;
						bootstrap_weight_per_path_length[depth][i] =
							sample_path(depth, sampler, scratch_buffer, raster_point, lambda).y(lambda);

						scratch_buffer.Reset();

						thread_d_max[0] = std::max<std::size_t>(thread_d_max[0], regeneration_distribution_sample[0].size());
						thread_d_max[1] = std::max<std::size_t>(thread_d_max[1], regeneration_distribution_sample[1].size());
						thread_d_max[2] = std::max<std::size_t>(thread_d_max[2], regeneration_distribution_sample[2].size());
					}
                }

				sholl::replace_if(d_max[0], thread_d_max[0], std::less<>{});
				sholl::replace_if(d_max[1], thread_d_max[1], std::less<>{});
				sholl::replace_if(d_max[2], thread_d_max[2], std::less<>{});
            });
			auto const bootrapping_end = std::chrono::high_resolution_clock::now();

			std::cout << "bootrapping finished after "
				<< std::chrono::duration_cast<std::chrono::seconds>(bootrapping_end - bootrapping_begin).count() << "s\n";

			Float c{};
			std::vector<Float> c_per_path_length(m_k_max);
			for (std::size_t depth = 0; depth < m_k_max; ++depth)
			{
				c += c_per_path_length[depth] = std::reduce(std::execution::par_unseq, bootstrap_weight_per_path_length[depth].begin(), bootstrap_weight_per_path_length[depth].end(), Float{}) / m_bootstrap_sample_count;
				std::cout << "c[" << depth << "] = " << c_per_path_length[depth] << '\n';
			}
			std::cout << "c = " << c << '\n' << std::flush;

			return {
				{ d_max[0] + 1, d_max[1], d_max[2] },
				c,
				{ c_per_path_length.begin(), c_per_path_length.end() }
			};
		}

		template<bool RaoBlackwellize>
		render_result render(bootstrap_result const& bootstrap_result,
			std::size_t const sample_count, std::vector<RGB>& unscaled_image)
		{
			ThreadLocal<ScratchBuffer> thread_scratch_buffer([]() { return ScratchBuffer{}; });

			auto const work = [&](std::size_t const thread, std::size_t const thread_sample_count)
			{
				ScratchBuffer& scratch_buffer = thread_scratch_buffer.Get();

				std::fill(std::execution::par_unseq, m_thread_data[thread].unscaled_image.begin(), m_thread_data[thread].unscaled_image.end(), RGB{});
				m_thread_data[thread].generated_sample_count = 0;
				m_thread_data[thread].generated_tour_count = 1;
				m_thread_data[thread].tau = 0;

				for (;; ++m_thread_data[thread].generated_tour_count)
				{
					m_thread_data[thread].primary_sample_space_data[0].resize(0);
					m_thread_data[thread].primary_sample_space_data[1].resize(0);
					m_thread_data[thread].primary_sample_space_data[2].resize(0);
					uniform_sampler regeneration_sampler{ m_sample_count_per_pixel, m_thread_data[thread].g, m_thread_data[thread].primary_sample_space_data };

					regeneration_sampler.begin_stream(0);
					std::size_t const depth = bootstrap_result.path_depth_distribution(regeneration_sampler.Get1D());
					Float const u = regeneration_sampler.Get1D();

					std::size_t s, t,
						strategy_count;
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

					state<RaoBlackwellize> current_state;
					current_state.contribution = sample_path(s, t, regeneration_sampler, scratch_buffer, current_state.raster_point, current_state.lambda) * m_multiplex_weight_correction;
					current_state.target_density = current_state.contribution.y(current_state.lambda);
					scratch_buffer.Reset();
					++m_thread_data[thread].generated_sample_count;

					if (current_state.target_density > 0)
					{
						Float const regeneration_density_at_current_state = bootstrap_result.path_depth_distribution.param().pmf[depth] / strategy_count;

						metropolis_restore_sampler restore_sampler{ m_sample_count_per_pixel, m_thread_data[thread].g, m_thread_data[thread].primary_sample_space_data, 1, bootstrap_result.c, m_expected_lifetime, m_sigma };

						Float tau,
							lambda;
						while (restore_sampler.try_begin_iteration(regeneration_density_at_current_state, current_state.target_density, tau, lambda))
						{
							if constexpr (!RaoBlackwellize)
							{
								m_thread_data[thread].tau += tau;
								if (current_state.target_density > 0) [[likely]]
									add_splat(m_thread_data[thread].unscaled_image, current_state.raster_point, tau * m_film.ToOutputRGB(current_state.contribution / current_state.target_density, current_state.lambda));
							}

							restore_sampler.begin_stream(0, 2); // skip depth and strategy coordinate

							state_base proposed_state;
							proposed_state.contribution = sample_path(s, t, restore_sampler, scratch_buffer, proposed_state.raster_point, proposed_state.lambda) * m_multiplex_weight_correction;
							proposed_state.target_density = proposed_state.contribution.y(proposed_state.lambda);
							scratch_buffer.Reset();
							++m_thread_data[thread].generated_sample_count;

							Float alpha;
							bool const accept = restore_sampler.end_iteration(current_state.target_density, proposed_state.target_density, alpha);

							if (accept)
							{
								if constexpr (RaoBlackwellize)
								{
									Float const w = current_state.xi / lambda;
									m_thread_data[thread].tau += w;
									if (current_state.target_density > 0) [[likely]]
										add_splat(m_thread_data[thread].unscaled_image, current_state.raster_point, w * m_film.ToOutputRGB(current_state.contribution / current_state.target_density, current_state.lambda));
								}

								current_state = state<RaoBlackwellize>{ proposed_state };
							}
							else
							{
								if constexpr (RaoBlackwellize)
									current_state.xi += current_state.weight *= 1 - alpha;
							}
						}

						if constexpr (!RaoBlackwellize)
						{
							m_thread_data[thread].tau += tau;
							if (current_state.target_density > 0)
								add_splat(m_thread_data[thread].unscaled_image, current_state.raster_point, tau * m_film.ToOutputRGB(current_state.contribution / current_state.target_density, current_state.lambda));
						}
						else
						{
							Float const w = current_state.xi / lambda;
							m_thread_data[thread].tau += w;
							if (current_state.target_density > 0)
								add_splat(m_thread_data[thread].unscaled_image, current_state.raster_point, w * m_film.ToOutputRGB(current_state.contribution / current_state.target_density, current_state.lambda));
						}
					}

					if (m_thread_data[thread].generated_sample_count >= thread_sample_count) [[unlikely]]
						break;
				}
			};

			std::size_t const thread_sample_count = sample_count / m_thread_count,
				last_thread_sample_count = sample_count - m_thread_pool.size() * thread_sample_count;

			auto const begin = std::chrono::high_resolution_clock::now();
			for (std::size_t thread = 0; thread < m_thread_pool.size(); ++thread)
				m_thread_pool[thread] = std::thread{ work, thread, thread_sample_count };
			work(m_thread_pool.size(), last_thread_sample_count);
			for (auto& thread : m_thread_pool)
				thread.join();

			std::size_t generated_sample_count{},
				generated_tour_count{};
			Float tau{};
			for (std::size_t thread = 0; thread < m_thread_count; ++thread)
			{
				generated_sample_count += m_thread_data[thread].generated_sample_count;
				generated_tour_count += m_thread_data[thread].generated_tour_count;
				tau += m_thread_data[thread].tau;
				std::transform(std::execution::par_unseq, unscaled_image.begin(), unscaled_image.end(),
					m_thread_data[thread].unscaled_image.begin(), unscaled_image.begin(), std::plus<>{});
			}
			auto const end = std::chrono::high_resolution_clock::now();

			return { generated_sample_count, generated_tour_count, tau, end - begin };
		}

		template<bool RaoBlackwellize>
		render_result render_for(bootstrap_result const& bootstrap_result,
			std::chrono::duration<Float, std::nano> const& computation_time, std::vector<RGB>& unscaled_image)
		{
			ThreadLocal<ScratchBuffer> thread_scratch_buffer([]() { return ScratchBuffer{}; });

			auto const work = [&](std::size_t const thread, std::chrono::duration<Float, std::nano> const& computation_time)
			{
				ScratchBuffer& scratch_buffer = thread_scratch_buffer.Get();

				std::fill(std::execution::par_unseq, m_thread_data[thread].unscaled_image.begin(), m_thread_data[thread].unscaled_image.end(), RGB{});
				m_thread_data[thread].generated_sample_count = 0;
				m_thread_data[thread].tau = 0;

				state<RaoBlackwellize> current_state;

				uniform_sampler regeneration_sampler{ m_sample_count_per_pixel, m_thread_data[thread].g, m_thread_data[thread].primary_sample_space_data };
				metropolis_restore_sampler restore_sampler{ m_sample_count_per_pixel, m_thread_data[thread].g, m_thread_data[thread].primary_sample_space_data, 1, bootstrap_result.c, m_expected_lifetime, m_sigma };

				for (auto const begin = std::chrono::high_resolution_clock::now();;)
				{
					m_thread_data[thread].primary_sample_space_data[0].resize(0);
					m_thread_data[thread].primary_sample_space_data[1].resize(0);
					m_thread_data[thread].primary_sample_space_data[2].resize(0);

					regeneration_sampler.begin_stream(0);
					std::size_t const depth = bootstrap_result.path_depth_distribution(regeneration_sampler.Get1D());
					Float const u = regeneration_sampler.Get1D();

					std::size_t s, t,
						strategy_count;
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

					current_state.contribution = sample_path(s, t, regeneration_sampler, scratch_buffer, current_state.raster_point, current_state.lambda) * m_multiplex_weight_correction;
					current_state.target_density = current_state.contribution.y(current_state.lambda);
					scratch_buffer.Reset();
					++m_thread_data[thread].generated_sample_count;

					if (current_state.target_density > 0)
					{
						Float const regeneration_density_at_current_state = bootstrap_result.path_depth_distribution.param().pmf[depth] / strategy_count;

						Float tau,
							lambda;
						while (restore_sampler.try_begin_iteration(regeneration_density_at_current_state, current_state.target_density, tau, lambda))
						{
							if constexpr (!RaoBlackwellize)
							{
								m_thread_data[thread].tau += tau;
								if (current_state.target_density > 0) [[likely]]
									add_splat(m_thread_data[thread].unscaled_image, current_state.raster_point, tau * m_film.ToOutputRGB(current_state.contribution / current_state.target_density, current_state.lambda));
							}

							restore_sampler.begin_stream(0, 2); // skip depth and strategy coordinate

							state_base proposed_state;
							proposed_state.contribution = sample_path(s, t, restore_sampler, scratch_buffer, proposed_state.raster_point, proposed_state.lambda) * m_multiplex_weight_correction;
							proposed_state.target_density = proposed_state.contribution.y(proposed_state.lambda);
							scratch_buffer.Reset();
							++m_thread_data[thread].generated_sample_count;

							Float alpha;
							if (restore_sampler.end_iteration(current_state.target_density, proposed_state.target_density, alpha))
							{
								if constexpr (RaoBlackwellize)
								{
									Float const w = current_state.xi / lambda;
									m_thread_data[thread].tau += w;
									if (current_state.target_density > 0) [[likely]]
										add_splat(m_thread_data[thread].unscaled_image, current_state.raster_point, w * m_film.ToOutputRGB(current_state.contribution / current_state.target_density, current_state.lambda));
								}

								current_state = state<RaoBlackwellize>{ proposed_state };
							}
							else
							{
								if constexpr (RaoBlackwellize)
									current_state.xi += current_state.weight *= 1 - alpha;
							}
						}

						if constexpr (!RaoBlackwellize)
						{
							m_thread_data[thread].tau += tau;
							if (current_state.target_density > 0)
								add_splat(m_thread_data[thread].unscaled_image, current_state.raster_point, tau * m_film.ToOutputRGB(current_state.contribution / current_state.target_density, current_state.lambda));
						}
						else
						{
							Float const w = current_state.xi / lambda;
							m_thread_data[thread].tau += w;
							if (current_state.target_density > 0)
								add_splat(m_thread_data[thread].unscaled_image, current_state.raster_point, w * m_film.ToOutputRGB(current_state.contribution / current_state.target_density, current_state.lambda));
						}

					}

					if (std::chrono::high_resolution_clock::now() - begin >= computation_time)
						return;
				}
			};

			for (std::size_t thread = 0; thread < m_thread_pool.size(); ++thread)
				m_thread_pool[thread] = std::thread{ work, thread, computation_time };
			work(m_thread_pool.size(), computation_time);
			for (auto& thread : m_thread_pool)
				thread.join();

			std::size_t generated_sample_count{};
			double tau{};
			for (std::size_t thread = 0; thread < m_thread_count; ++thread)
			{
				generated_sample_count += m_thread_data[thread].generated_sample_count;
				tau += m_thread_data[thread].tau;
				std::transform(std::execution::par_unseq, unscaled_image.begin(), unscaled_image.end(),
					m_thread_data[thread].unscaled_image.begin(), unscaled_image.begin(), std::plus<>{});
			}

			return {
				generated_sample_count,
				0,
				tau,
				computation_time
			};
		}

		void add_splat(std::vector<RGB>& image, Point2f const& raster_point, RGB const& rgb) const
		{
			std::size_t const j = std::min(static_cast<std::size_t>(raster_point[0]), m_film_sample_space_width - 1),
				i = std::min(static_cast<std::size_t>(raster_point[1]), m_film_sample_space_height - 1);

			std::size_t const pixel_index = i * m_film_sample_space_width + j;
			image[pixel_index][0] += rgb.r;
			image[pixel_index][1] += rgb.g;
			image[pixel_index][2] += rgb.b;
		}

		std::optional<Image> get_reference_image() const
		{
			auto image_and_metadata = Image::Read(Options->mseReferenceImage);

			Bounds2i const pixel_bounding_box = image_and_metadata.metadata.pixelBounds ?
				*image_and_metadata.metadata.pixelBounds : Bounds2i{ Point2i{}, image_and_metadata.image.Resolution() };
			if (!Inside(static_cast<Bounds2i>(m_film_sample_space_bounding_box), pixel_bounding_box))
			{
				ErrorExit("Output image pixel bound box %s is not inside the MSE image's pixel bounding box %s.",
					m_film_sample_space_bounding_box, pixel_bounding_box);
			}

			Bounds2i const cropped_bounding_box{ 
				Point2i{ m_film_sample_space_bounding_box.pMin - pixel_bounding_box.pMin },
				Point2i{ m_film_sample_space_bounding_box.pMax - pixel_bounding_box.pMin }
			};

			image_and_metadata.image = image_and_metadata.image.Crop(cropped_bounding_box);
			CHECK_EQ(image_and_metadata.image.Resolution(), static_cast<Point2i>(m_film_sample_space_bounding_box.Diagonal()));

			return image_and_metadata.image;
		}

		SampledSpectrum sample_path(std::size_t const depth, metropolis_restore_sampler& sampler, ScratchBuffer& scratch_buffer, Point2f& raster_point, SampledWavelengths& lambda) const;
		SampledSpectrum sample_path(std::size_t const s, std::size_t const t, metropolis_restore_sampler& sampler, ScratchBuffer& scratch_buffer, Point2f& raster_point, SampledWavelengths& lambda) const;

		// TODO:
		SampledSpectrum sample_path(std::size_t const depth, uniform_sampler& sampler, ScratchBuffer& scratch_buffer, Point2f& raster_point, SampledWavelengths& lambda) const;
		SampledSpectrum sample_path(std::size_t const s, std::size_t const t, uniform_sampler& sampler, ScratchBuffer& scratch_buffer, Point2f& raster_point, SampledWavelengths& lambda) const;

		Camera const m_camera;

		Film const m_film;
		Bounds2f const m_film_sample_space_bounding_box;

		LightSampler m_light_sampler;

		std::size_t const m_k_max,
			m_bootstrap_sample_count,
			m_film_sample_space_width,
			m_film_sample_space_height,
			m_pixel_count,
			m_sample_count,
			m_realization_count,
			m_thread_count;
		int const m_bootstrap_sample_count_per_pixel,
			m_sample_count_per_pixel;
		Float const m_expected_lifetime,
			m_sigma,
			m_multiplex_weight_correction;
		bool const m_rao_blackwellize,
			m_compute_mse;
		std::chrono::duration<Float, std::nano> m_computation_time = std::chrono::seconds(1),
			m_frame_period = std::chrono::seconds(1);

		std::vector<std::thread> mutable m_thread_pool;

		struct thread_data
		{
			thread_data(std::size_t const pixel_count)
				: unscaled_image(pixel_count)
			{}

			std::mt19937 g;
			std::array<std::vector<sholl::restorable<Float>>, 3> primary_sample_space_data;
			std::size_t generated_sample_count,
				generated_tour_count;
			Float tau;
			std::vector<RGB> unscaled_image;
		}; // struct thread_data

		std::vector<thread_data> mutable m_thread_data;

		void init_thead_data(bootstrap_result const& bootstrap_result)
		{
			for (std::size_t i = 0; i < m_thread_count; ++i)
			{
				m_thread_data[i].g.seed(std::random_device{}());
				m_thread_data[i].primary_sample_space_data[0].resize(bootstrap_result.d_max[0]);
				m_thread_data[i].primary_sample_space_data[1].resize(bootstrap_result.d_max[1]);
				m_thread_data[i].primary_sample_space_data[2].resize(bootstrap_result.d_max[2]);
			}
		}
	}; // class metropolis_restore_integrator
} // namespace sholl


#endif // !HPP_SHOLL_INTEGRATOR_METROPOLIS_RESTORE_INTEGRATOR_INCLUDED
