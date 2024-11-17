// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#ifndef HPP_SHOLL_INTEGRATOR_METROPOLIS_INTEGRATOR_INCLUDED
#define HPP_SHOLL_INTEGRATOR_METROPOLIS_INTEGRATOR_INCLUDED


#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <execution>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <optional>
#include <vector>

#include <Eigen/Eigen>

#include <pbrt/cpu/integrators.h>

#include <sholl/random/discrete_distribution.hpp>
#include <sholl/sampler/local_exploration_sampler.hpp>


namespace pbrt
{
	class metropolis_integrator
		: public Integrator
	{
	public:
		static std::unique_ptr<metropolis_integrator> Create(ParameterDictionary const& parameter_dictionary,
			Primitive aggregate, std::vector<Light> const& light, Camera camera, FileLoc const*)
		{
			if (!camera.Is<PerspectiveCamera>())
				ErrorExit("Only the \"perspective\" camera is currently supported with the \"metropolis\" integrator.");

			int const k_max = parameter_dictionary.GetOneInt("k_max", 10),
				bootstrap_sample_count = parameter_dictionary.GetOneInt("bootstrap_sample_count", static_cast<int>(1e5)),
				sample_count_per_pixel = parameter_dictionary.GetOneInt("sample_count_per_pixel", 1),
				chain_count = parameter_dictionary.GetOneInt("chain_count", 1000),
				realization_count = parameter_dictionary.GetOneInt("realization_count", 1),
				fps = parameter_dictionary.GetOneInt("fps", 1);
			Float const large_step_probability = parameter_dictionary.GetOneFloat("large_step_probability", .3),
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
			if (large_step_probability < 0 || large_step_probability > 1)
				ErrorExit("large_step_probability < 0 || large_step_probability > 1");
			if (sigma <= 0)
				ErrorExit("sigma <= 0");
			if (chain_count < 1)
				ErrorExit("chain_count < 1");
			if (computation_time < 0)
				ErrorExit("computation_time < 0");
			if (realization_count < 1)
				ErrorExit("realization_count < 1");

			return std::make_unique<metropolis_integrator>(aggregate, light, camera,
				static_cast<std::size_t>(k_max),
				static_cast<std::size_t>(bootstrap_sample_count),
				sample_count_per_pixel,
				large_step_probability,
				sigma,
				rao_blackwellize,
				compute_mse,
				realization_count,
				static_cast<std::size_t>(chain_count),
				std::chrono::duration<Float>(computation_time),
				std::chrono::duration<Float>(Float{ 1 } / fps)
			);
		}

		metropolis_integrator(Primitive aggregate, std::vector<Light> const& light, Camera camera,
			std::size_t const k_max, std::size_t const bootstrap_sample_count, int const sample_count_per_pixel, Float const large_step_probability, Float const sigma, bool const rao_blackwellize,
			bool const compute_mse, std::size_t const realization_count,
			std::size_t const chain_count,
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
			  m_bootstrap_sample_count_per_pixel{ static_cast<int>(std::ceil(static_cast<Float>(m_k_max * m_bootstrap_sample_count) / m_pixel_count)) },
			  m_sample_count_per_pixel(sample_count_per_pixel),
			  m_sample_count(m_pixel_count * sample_count_per_pixel),
			  m_large_step_probability(large_step_probability),
			  m_sigma(sigma),
			  m_rao_blackwellize(rao_blackwellize),
			  m_compute_mse(compute_mse),
			  m_realization_count(realization_count),
			  m_chain_count{ chain_count },
			  m_chain_data(m_realization_count),
#ifdef _DEBUG
			  m_thread_count(1),
#else
			  m_thread_count(std::min(m_chain_count, static_cast<std::size_t>(std::thread::hardware_concurrency()))),
#endif // _DEBUG
			  m_thread_pool(m_thread_count - 1),
			  m_thread_data(m_thread_pool.size() + 1, m_pixel_count),
			  m_thread_chain_count(m_chain_count / m_thread_count),
			  m_last_thread_chain_count(m_chain_count - (m_thread_count - 1) * m_thread_chain_count),
			  m_computation_time(computation_time),
			  m_frame_period(frame_period)
		{}

		void Render() override
		{
			auto const reference_image = get_reference_image();

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

				if (m_computation_time > std::chrono::duration<Float, std::nano>::zero())
				{
					if (m_rao_blackwellize)
						compute_equal_time_mse<true>(m_film.GetFilename(), *reference_image);
					else
						compute_equal_time_mse<false>(m_film.GetFilename(), *reference_image);
				}
				else
				{
					if (m_rao_blackwellize)
						compute_mse<true>(m_film.GetFilename(), *reference_image);
					else
						compute_mse<false>(m_film.GetFilename(), *reference_image);
				}
			}
		}

		std::string ToString() const override { return {}; }

	private:
		struct state
		{
			Point2f raster_point;
			SampledWavelengths lambda;

			SampledSpectrum contribution;
			Float target_density;
		}; // struct state

		struct chain_data
		{
			chain_data(int const sample_count_per_pixel,
				Float const large_step_probability,
				Float const sigma)
				: sampler{ sample_count_per_pixel, g, large_step_probability, sigma }
			{}

			std::mt19937 g;
			local_exploration_sampler sampler;

			std::size_t s, t,
				depth;
			state current_state;
			bool initialized = false;

			//std::size_t generated_sample_count;
			std::chrono::duration<Float, std::nano> computation_time;
		}; // struct chain_data

		struct bootstrap_result
		{
			Float c;
			sholl::discrete_distribution<std::size_t> path_depth_distribution;
		};

		struct render_result
		{
			std::size_t generated_sample_count;
			std::chrono::duration<Float, std::nano> elapsed_time;
		};

		template<bool RaoBlackwellize>
		void render(std::string const& filename, std::optional<Image> const& reference_image = std::nullopt)
		{
			bool const equal_time_comparison = m_computation_time > std::chrono::duration<Float, std::nano>::zero();
			if (equal_time_comparison)
				std::cout << "equal time comparison for metropolis\n";
			else
				std::cout << "equal sample count comparison for metropolis\n";

			std::ofstream log(filename + ".log", std::ios::binary);

			bootstrap_result const bootstrap_result = bootstrap();
			if (bootstrap_result.c == 0)
			{
				std::cout << "black image\n";
				return;
			}

			std::cout << "rendering\n" << std::flush;

			init_chain_data();

			std::vector<RGB> value_image(m_pixel_count);
			render_result render_result;
			if (equal_time_comparison)
				render_result = render_for<RaoBlackwellize>(bootstrap_result, m_computation_time, m_chain_data[0], value_image);
			else
				render_result = render<RaoBlackwellize>(bootstrap_result, m_sample_count, m_chain_data[0], value_image);

			Image image{ PixelFormat::Float, static_cast<Point2i>(m_film_sample_space_bounding_box.Diagonal()), { "R", "G", "B" } };

			Float const s = m_pixel_count * bootstrap_result.c / render_result.generated_sample_count;
			for (std::size_t i = 0; i < m_pixel_count; ++i)
			{
				image.SetChannels(Point2i{ static_cast<int>(i % m_film_sample_space_width), static_cast<int>(i / m_film_sample_space_width) }, {
					s * value_image[i][0],
					s * value_image[i][1],
					s * value_image[i][2]
				});
			}

			Float const rendering_time = std::chrono::duration_cast<std::chrono::seconds>(render_result.elapsed_time).count();

			std::stringstream ss;
			ss << "generated sample count per pixel = " << m_sample_count_per_pixel << '\n';
			ss << "rendering time = " << rendering_time << '\n';

			if (reference_image)
				ss << "mse = " << image.MSE(image.AllChannelsDesc(), *reference_image).Average() << '\n';
			ss << std::flush;

			log << ss.str();
			std::cout << ss.str();

			ImageMetadata image_metadata;
			m_camera.InitMetadata(&image_metadata);

			image_metadata.renderTimeSeconds = rendering_time;
			image_metadata.samplesPerPixel = m_sample_count_per_pixel;
			image_metadata.pixelBounds = static_cast<Bounds2i>(m_film_sample_space_bounding_box);
			image_metadata.fullResolution = m_film.FullResolution();

			image.Write(filename, image_metadata);
		}

		template<bool RaoBlackwellize>
		void compute_mse(std::string const& filename, Image const& reference_image)
		{
			std::cout << "equal sample count comparison for metropolis\n";

			std::ofstream log(filename + ".log", std::ios::binary);
			std::ofstream mse_out(filename + ".mse", std::ios::binary);
			mse_out << std::fixed << std::setprecision(4);

			bootstrap_result const bootstrap_result = bootstrap();
			if (bootstrap_result.c == 0)
			{
				std::cout << "black image\n";
				return;
			}

			init_chain_data();

			Float const scale = m_pixel_count * bootstrap_result.c,
				scale_over_realization_count = scale / m_realization_count;

			std::size_t generated_sample_count{};
			std::vector<std::chrono::duration<Float, std::nano>> elapsed_time(m_realization_count);
			std::vector<std::vector<RGB>> unscaled_image(m_realization_count, std::vector<RGB>(m_pixel_count));

			std::size_t const sample_count_per_pixel_k = static_cast<std::size_t>(std::ceil(std::log2(static_cast<Float>(m_sample_count_per_pixel))));
			for (std::size_t k = 0; k <= sample_count_per_pixel_k; ++k)
			{
				std::vector<Image> image(m_realization_count, Image{ PixelFormat::Float, static_cast<Point2i>(m_film_sample_space_bounding_box.Diagonal()), { "R", "G", "B" } });
				std::vector<RGB> unscaled_image_empirical_mean(m_pixel_count);

				std::size_t const sample_count_per_pixel = static_cast<std::size_t>(1 << k);

				std::size_t const sample_count = m_pixel_count * sample_count_per_pixel;

				Float rendering_time_average{},
					mse_average{},
					min_mse = std::numeric_limits<Float>::infinity();

				for (std::size_t realization = 0; realization < m_realization_count; ++realization)
				{
					std::stringstream ss;
					ss << "rendering 2^" << k << "spp (realization " << realization << ")\n" << std::flush;
					log << ss.str();
					std::cout << ss.str();

					render_result const render_result = render<RaoBlackwellize>(bootstrap_result, sample_count - generated_sample_count, m_chain_data[realization], unscaled_image[realization]);
					elapsed_time[realization] += render_result.elapsed_time;

					for (std::size_t i = 0; i < m_pixel_count; ++i)
					{
						Float const
							r = unscaled_image[realization][i][0] / sample_count,
							g = unscaled_image[realization][i][1] / sample_count,
							b = unscaled_image[realization][i][2] / sample_count;

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

					rendering_time_average += rendering_time;
					mse_average += mse;
				}

				generated_sample_count = sample_count;
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

				mse_out << sample_count_per_pixel << '\t'
					<< rendering_time_average << '\t'
					<< mse_average << '\t'
					<< empirical_variance
					<< std::endl << std::flush;
			}
		}

		template<bool RaoBlackwellize>
		void compute_equal_time_mse(std::string const& filename, Image const& reference_image)
		{
			std::cout << "equal time comparison for metropolis\n";

			std::ofstream log(filename + ".log", std::ios::binary);
			std::ofstream mse_out(filename + ".mse", std::ios::binary);
			mse_out << std::fixed << std::setprecision(4);

			bootstrap_result const bootstrap_result = bootstrap();
			if (bootstrap_result.c == 0)
			{
				std::cout << "black image\n";
				return;
			}

			init_chain_data();

			Float const scale = bootstrap_result.c * m_pixel_count,
				scale_over_realization_count = scale / m_realization_count;

			std::vector<std::size_t> generated_sample_count(m_realization_count);
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

				std::chrono::duration<Float, std::nano> computation_time_average{};
				Float generated_sample_count_average{},
					mse_average{},
					min_mse = std::numeric_limits<Float>::infinity(),
					max_mse = -min_mse;

				for (std::size_t realization = 0; realization < m_realization_count; ++realization)
				{
					render_result const render_result = render_for<RaoBlackwellize>(bootstrap_result, frame * m_frame_period - computation_time[realization], m_chain_data[realization], unscaled_image[realization]);

					generated_sample_count[realization] += render_result.generated_sample_count;
					computation_time[realization] += render_result.elapsed_time;

					for (std::size_t i = 0; i < m_pixel_count; ++i)
					{
						Float const
							r = unscaled_image[realization][i][0] / generated_sample_count[realization],
							g = unscaled_image[realization][i][1] / generated_sample_count[realization],
							b = unscaled_image[realization][i][2] / generated_sample_count[realization];

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
						min_mse = mse;
					if (mse > max_mse)
					{
						max_mse = mse;
						write_image = true;
					}

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
					computation_time_average += computation_time[realization];
					mse_average += mse;
				}

				generated_sample_count_average /= m_realization_count;
				computation_time_average /= m_realization_count;
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
					<< static_cast<Float>(computation_time_average.count()) / std::chrono::duration<Float, std::nano>{ 1s }.count() << '\t'
					<< mse_average << '\t'
					<< empirical_variance
					<< std::endl << std::flush;
			}
		}

		bootstrap_result bootstrap() const
		{
			std::cout << "bootstrapping\n" << std::flush;

			ThreadLocal<ScratchBuffer> thread_scratch_buffer([]() { return ScratchBuffer{}; });

			std::vector<std::vector<Float>> bootstrap_weight_per_path_length(m_k_max, std::vector<Float>(m_bootstrap_sample_count));

			auto const bootrapping_begin = std::chrono::high_resolution_clock::now();
			ParallelFor(0, static_cast<std::int64_t>(m_bootstrap_sample_count), [&](std::int64_t first, std::int64_t last)
            {
				ScratchBuffer& scratch_buffer = thread_scratch_buffer.Get();

				for (std::int64_t i = first; i < last; ++i)
                {
					for (std::size_t depth = 0; depth < m_k_max; ++depth)
					{
						std::seed_seq seed{ static_cast<std::size_t>(i), depth };
						std::mt19937 g{ seed };
						local_exploration_sampler sampler{ m_bootstrap_sample_count_per_pixel, g, m_large_step_probability, m_sigma };
						sampler.begin_stream(0);

						Point2f raster_point;
						SampledWavelengths lambda;
						bootstrap_weight_per_path_length[depth][i] =
							sample_path(depth, sampler, scratch_buffer, raster_point, lambda).y(lambda);

						scratch_buffer.Reset();
					}
                }
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

			return { c, { c_per_path_length.begin(), c_per_path_length.end() } };
		}

		template<bool RaoBlackwellize>
		render_result render(bootstrap_result const& bootstrap_result,
			std::size_t const sample_count, std::vector<chain_data>& chain_data, std::vector<RGB>& unscaled_image)
		{
			for (std::size_t thread = 0; thread < m_thread_count; ++thread)
			{
				std::fill(std::execution::par_unseq, m_thread_data[thread].unscaled_image.begin(), m_thread_data[thread].unscaled_image.end(), RGB{});
				m_thread_data[thread].generated_sample_count = 0;
				m_thread_data[thread].computation_time = std::chrono::duration<Float, std::nano>{};
			}

			std::atomic_size_t chain_id{};
			auto const run_chain = [&](std::size_t const thread, std::size_t const chain_sample_count, ScratchBuffer& scratch_buffer)
			{
				std::size_t const chain = chain_id++;

				chain_data[chain].sampler.begin_stream(0);
				std::size_t const depth = bootstrap_result.path_depth_distribution(chain_data[chain].sampler.Get1D());
				Float const u = chain_data[chain].sampler.Get1D();

				std::size_t strategy_count;
				if (depth > 0) [[likely]]
				{
					strategy_count = depth + 2;
					chain_data[chain].s = std::min<std::size_t>(u * strategy_count, strategy_count - 1);
					chain_data[chain].t = strategy_count - chain_data[chain].s;
				}
				else [[unlikely]]
				{
					strategy_count = 1;
					chain_data[chain].s = 0;
					chain_data[chain].t = 2;
				}

				chain_data[chain].current_state.contribution = sample_path(chain_data[chain].s, chain_data[chain].t, chain_data[chain].sampler, scratch_buffer, chain_data[chain].current_state.raster_point, chain_data[chain].current_state.lambda);
				chain_data[chain].current_state.target_density = chain_data[chain].current_state.contribution.y(chain_data[chain].current_state.lambda);
				scratch_buffer.Reset();
				++m_thread_data[thread].generated_sample_count;

				if constexpr (!RaoBlackwellize)
				{
					if (chain_data[chain].current_state.target_density > 0)
						add_splat(m_thread_data[thread].unscaled_image, chain_data[chain].current_state.raster_point, m_film.ToOutputRGB(chain_data[chain].current_state.contribution / chain_data[chain].current_state.target_density, chain_data[chain].current_state.lambda));
				}

				for (std::size_t i = 1; i < chain_sample_count; ++i)
				{
					bool const large_step = chain_data[chain].sampler.begin_iteration();
					chain_data[chain].sampler.begin_stream(0, 2); // skip depth coordinate

					state proposed_state;
					proposed_state.contribution = sample_path(chain_data[chain].s, chain_data[chain].t, chain_data[chain].sampler, scratch_buffer, proposed_state.raster_point, proposed_state.lambda);
					proposed_state.target_density = proposed_state.contribution.y(proposed_state.lambda);
					scratch_buffer.Reset();
					++m_thread_data[thread].generated_sample_count;

					Float alpha;
					bool const accept = chain_data[chain].sampler.end_iteration(chain_data[chain].current_state.target_density, proposed_state.target_density, alpha);

					if constexpr (RaoBlackwellize)
					{
						if (alpha > 0 && proposed_state.target_density > 0)
							add_splat(m_thread_data[thread].unscaled_image, proposed_state.raster_point, alpha * m_film.ToOutputRGB(proposed_state.contribution / proposed_state.target_density, proposed_state.lambda));
						if (chain_data[chain].current_state.target_density > 0)
							add_splat(m_thread_data[thread].unscaled_image, chain_data[chain].current_state.raster_point, (1 - alpha) * m_film.ToOutputRGB(chain_data[chain].current_state.contribution / chain_data[chain].current_state.target_density, chain_data[chain].current_state.lambda));
					}

					if (accept)
						chain_data[chain].current_state = proposed_state;

					if constexpr (!RaoBlackwellize)
					{
						if (chain_data[chain].current_state.target_density > 0)
							add_splat(m_thread_data[thread].unscaled_image, chain_data[chain].current_state.raster_point, m_film.ToOutputRGB(chain_data[chain].current_state.contribution / chain_data[chain].current_state.target_density, chain_data[chain].current_state.lambda));
					}
				}
			};

			std::size_t const
				chain_sample_count = sample_count / m_chain_count,
				thread_count = std::min(m_thread_pool.size() + 1, m_chain_count),
				thread_chain_count = m_chain_count / thread_count,
				last_thread_chain_count = m_chain_count - (thread_count - 1) * thread_chain_count,
				last_chain_sample_count = sample_count - (m_chain_count - 1) * chain_sample_count;

			ThreadLocal<ScratchBuffer> thread_scratch_buffer([]() { return ScratchBuffer{}; });

			auto const work = [&](std::size_t const thread, std::size_t const thread_chain_count)
			{
				ScratchBuffer& scratch_buffer = thread_scratch_buffer.Get();
				for (std::size_t chain = 0; chain < thread_chain_count; ++chain)
					run_chain(thread, chain_sample_count, scratch_buffer);
			};
			auto const last_work = [&]()
			{
				ScratchBuffer& scratch_buffer = thread_scratch_buffer.Get();
				for (std::size_t chain = 0; chain < last_thread_chain_count - 1; ++chain)
					run_chain(thread_count - 1, chain_sample_count, scratch_buffer);
				run_chain(thread_count - 1, last_chain_sample_count, scratch_buffer);
			};

			auto const begin = std::chrono::high_resolution_clock::now();
			for (std::size_t thread = 0; thread < thread_count - 1; ++thread)
				m_thread_pool[thread] = std::thread{ work, thread, thread_chain_count };
			last_work();
			for (auto& thread : m_thread_pool)
				thread.join();

			for (auto const& thread_data : m_thread_data)
			{
				std::transform(std::execution::par_unseq, unscaled_image.begin(), unscaled_image.end(),
					thread_data.unscaled_image.begin(), unscaled_image.begin(), std::plus<>{});
			}

			return { sample_count, std::chrono::high_resolution_clock::now() - begin };
		}

		template<bool RaoBlackwellize>
		render_result render_for(bootstrap_result const& bootstrap_result,
			std::chrono::duration<Float, std::nano> const& computation_time, std::vector<chain_data>& chain_data, std::vector<RGB>& unscaled_image)
		{
			for (std::size_t thread = 0; thread < m_thread_count; ++thread)
			{
				std::fill(std::execution::par_unseq, m_thread_data[thread].unscaled_image.begin(), m_thread_data[thread].unscaled_image.end(), RGB{});
				m_thread_data[thread].generated_sample_count = 0;
				m_thread_data[thread].computation_time = std::chrono::duration<Float, std::nano>{};
			}

			auto const run_chain = [&](std::size_t const thread, std::size_t const chain, std::chrono::duration<Float, std::nano> const& computation_time, ScratchBuffer& scratch_buffer)
			{
				auto const begin = std::chrono::high_resolution_clock::now();

				if (!chain_data[chain].initialized)
				{
					chain_data[chain].initialized = true;

					chain_data[chain].sampler.begin_stream(0);
					chain_data[chain].depth = bootstrap_result.path_depth_distribution(chain_data[chain].sampler.Get1D());
					Float const u = chain_data[chain].sampler.Get1D();

					std::size_t strategy_count;
					if (chain_data[chain].depth > 0) [[likely]]
					{
						strategy_count = chain_data[chain].depth + 2;
						chain_data[chain].s = std::min<std::size_t>(u * strategy_count, strategy_count - 1);
						chain_data[chain].t = strategy_count - chain_data[chain].s;
					}
					else [[unlikely]]
					{
						strategy_count = 1;
						chain_data[chain].s = 0;
						chain_data[chain].t = 2;
					}

					Float const regeneration_density_at_current_state = bootstrap_result.path_depth_distribution.param().pmf[chain_data[chain].depth] / strategy_count;

					chain_data[chain].current_state.contribution = sample_path(chain_data[chain].s, chain_data[chain].t, chain_data[chain].sampler, scratch_buffer, chain_data[chain].current_state.raster_point, chain_data[chain].current_state.lambda) / regeneration_density_at_current_state;
					chain_data[chain].current_state.target_density = chain_data[chain].current_state.contribution.y(chain_data[chain].current_state.lambda);
					scratch_buffer.Reset();
					++m_thread_data[thread].generated_sample_count;

					if constexpr (!RaoBlackwellize)
					{
						if (chain_data[chain].current_state.target_density > 0)
							add_splat(m_thread_data[thread].unscaled_image, chain_data[chain].current_state.raster_point, m_film.ToOutputRGB(chain_data[chain].current_state.contribution / chain_data[chain].current_state.target_density, chain_data[chain].current_state.lambda));
					}

					chain_data[chain].computation_time = std::chrono::high_resolution_clock::now() - begin;
					if (chain_data[chain].computation_time >= computation_time)
					{
						m_thread_data[thread].computation_time += chain_data[chain].computation_time;
						return;
					}
				}

				do
				{
					Float regeneration_density_at_current_state;

					bool const large_step = chain_data[chain].sampler.begin_iteration();
					if (large_step)
					{
						chain_data[chain].sampler.begin_stream(0);
						chain_data[chain].depth = bootstrap_result.path_depth_distribution(chain_data[chain].sampler.Get1D());
						Float const u = chain_data[chain].sampler.Get1D();

						std::size_t strategy_count;
						if (chain_data[chain].depth > 0) [[likely]]
						{
							strategy_count = chain_data[chain].depth + 2;
							chain_data[chain].s = std::min<std::size_t>(u * strategy_count, strategy_count - 1);
							chain_data[chain].t = strategy_count - chain_data[chain].s;
						}
						else [[unlikely]]
						{
							strategy_count = 1;
							chain_data[chain].s = 0;
							chain_data[chain].t = 2;
						}

						regeneration_density_at_current_state = bootstrap_result.path_depth_distribution.param().pmf[chain_data[chain].depth] / strategy_count;
					}
					else
					{
						chain_data[chain].sampler.begin_stream(0, 1); // skip depth
						Float const u = chain_data[chain].sampler.Get1D();

						std::size_t strategy_count;
						if (chain_data[chain].depth > 0) [[likely]]
						{
							strategy_count = chain_data[chain].depth + 2;
							chain_data[chain].s = std::min<std::size_t>(u * strategy_count, strategy_count - 1);
							chain_data[chain].t = strategy_count - chain_data[chain].s;
						}
						else [[unlikely]]
						{
							strategy_count = 1;
							chain_data[chain].s = 0;
							chain_data[chain].t = 2;
						}

						regeneration_density_at_current_state = Float{ 1 } / strategy_count;
					}

					state proposed_state;
					proposed_state.contribution = sample_path(chain_data[chain].s, chain_data[chain].t, chain_data[chain].sampler, scratch_buffer, proposed_state.raster_point, proposed_state.lambda) / regeneration_density_at_current_state;
					proposed_state.target_density = proposed_state.contribution.y(proposed_state.lambda);
					scratch_buffer.Reset();
					++m_thread_data[thread].generated_sample_count;
					//++chain_data[chain].generated_sample_count;

					Float alpha;
					bool const accept = chain_data[chain].sampler.end_iteration(chain_data[chain].current_state.target_density, proposed_state.target_density, alpha);

					if constexpr (RaoBlackwellize)
					{
						if (alpha > 0 && proposed_state.target_density > 0)
							add_splat(m_thread_data[thread].unscaled_image, proposed_state.raster_point, alpha * m_film.ToOutputRGB(proposed_state.contribution / proposed_state.target_density, proposed_state.lambda));
						if (chain_data[chain].current_state.target_density > 0)
							add_splat(m_thread_data[thread].unscaled_image, chain_data[chain].current_state.raster_point, (1 - alpha) * m_film.ToOutputRGB(chain_data[chain].current_state.contribution / chain_data[chain].current_state.target_density, chain_data[chain].current_state.lambda));
					}

					if (accept)
						chain_data[chain].current_state = proposed_state;

					if constexpr (!RaoBlackwellize)
					{
						if (chain_data[chain].current_state.target_density > 0)
							add_splat(m_thread_data[thread].unscaled_image, chain_data[chain].current_state.raster_point, m_film.ToOutputRGB(chain_data[chain].current_state.contribution / chain_data[chain].current_state.target_density, chain_data[chain].current_state.lambda));
					}

					chain_data[chain].computation_time = std::chrono::high_resolution_clock::now() - begin;
				} while (chain_data[chain].computation_time < computation_time);

				m_thread_data[thread].computation_time += chain_data[chain].computation_time;
			};
	
			auto const thread_chain_computation_time  = computation_time / m_thread_chain_count,
				last_thread_chain_computation_time = computation_time / m_last_thread_chain_count;

			ThreadLocal<ScratchBuffer> thread_scratch_buffer([]() { return ScratchBuffer{}; });

			auto const work = [&](std::size_t const thread, std::size_t const thread_chain_offset, std::size_t const thread_chain_count,
				std::chrono::duration<Float, std::nano> const& chain_rendering_time)
			{
				ScratchBuffer& scratch_buffer = thread_scratch_buffer.Get();
				for (std::size_t chain = 0; chain < thread_chain_count; ++chain)
					run_chain(thread, thread_chain_offset + chain, chain_rendering_time, scratch_buffer);
			};

			std::size_t thread_chain_offset{};
			for (std::size_t thread = 0; thread < m_thread_pool.size(); ++thread)
			{
				m_thread_pool[thread] = std::thread{ work, thread, thread_chain_offset, m_thread_chain_count, thread_chain_computation_time };
				thread_chain_offset += m_thread_chain_count;
			}
			work(m_thread_pool.size(), thread_chain_offset, m_last_thread_chain_count, last_thread_chain_computation_time);
			for (auto& thread : m_thread_pool)
				thread.join();

			std::size_t generated_sample_count{};
			for (auto const& thread_data : m_thread_data)
			{
				generated_sample_count += thread_data.generated_sample_count;
				std::transform(std::execution::par_unseq, unscaled_image.begin(), unscaled_image.end(),
					thread_data.unscaled_image.begin(), unscaled_image.begin(), std::plus<>{});
			}

			return {
				generated_sample_count,
				std::max_element(std::execution::par_unseq, m_thread_data.begin(), m_thread_data.end(),
					[](auto const& first, auto const& second) { return first.computation_time < second.computation_time; })->computation_time
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

		SampledSpectrum sample_path(std::size_t const depth, local_exploration_sampler& sampler, ScratchBuffer& scratch_buffer, Point2f& raster_point, SampledWavelengths& lambda) const;
		SampledSpectrum sample_path(std::size_t const s, std::size_t const t, local_exploration_sampler& sampler, ScratchBuffer& scratch_buffer, Point2f& raster_point, SampledWavelengths& lambda) const;

		SampledSpectrum sample_debug_path(Point2f& raster_point, std::size_t const depth, local_exploration_sampler& sampler, ScratchBuffer& scratch_buffer, SampledWavelengths& lambda) const;
		SampledSpectrum sample_debug_path(Point2f& raster_point, std::size_t const s, std::size_t const t, local_exploration_sampler& sampler, ScratchBuffer& scratch_buffer, SampledWavelengths& lambda) const;

		void init_chain_data()
		{
			for (std::size_t realization = 0; realization < m_realization_count; ++realization)
			{
				m_chain_data[realization].reserve(m_chain_count);
				for (std::size_t i = 0; i < m_chain_count; ++i)
				{
					m_chain_data[realization].emplace_back(m_sample_count_per_pixel, m_large_step_probability, m_sigma);
					m_chain_data[realization].back().g.seed(std::random_device{}());
					//m_chain_data[realization].back().g.seed(static_cast<std::mt19937::result_type>(realization * m_chain_count + i));
				}
			}
		}

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
			m_chain_count,
			m_thread_count,
			m_thread_chain_count,
			m_last_thread_chain_count;
		int const m_bootstrap_sample_count_per_pixel,
			m_sample_count_per_pixel;
		Float const m_large_step_probability,
			m_sigma;
		bool const m_rao_blackwellize,
			m_compute_mse;

		std::vector<std::thread> mutable m_thread_pool;

		struct thread_data
		{
			thread_data() = default;
			thread_data(std::size_t const pixel_count)
				: unscaled_image(pixel_count)
			{}

			std::vector<RGB> unscaled_image;

			std::size_t generated_sample_count;
			std::chrono::duration<Float, std::nano> computation_time;
		}; // struct thread_data

		std::vector<std::vector<chain_data>> m_chain_data;
		std::vector<thread_data> mutable m_thread_data;

		std::chrono::duration<Float, std::nano> const m_computation_time,
			m_frame_period;
	}; // class metropolis_integrator
} // namespace sholl


#endif // !HPP_SHOLL_INTEGRATOR_METROPOLIS_INTEGRATOR_INCLUDED
