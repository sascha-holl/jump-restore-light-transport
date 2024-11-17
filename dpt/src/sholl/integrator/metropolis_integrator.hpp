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
#include <chrono>
#include <execution>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>

#include <boost/iterator/counting_iterator.hpp>

#include "../../image.h"
#include "../../path.h"
#include "../../scene.h"

#include "../utility.hpp"


namespace sholl
{
	template<class LargeStep, class SmallStep>
	class metropolis_integrator
	{
	public:
		metropolis_integrator(Scene const* scene, std::shared_ptr<PathFuncLib const> const& path_function_library)
			: m_scene(scene),
			  m_path_function_library(path_function_library),
			  m_pixel_count(static_cast<std::size_t>(scene->camera->film->pixelWidth * scene->camera->film->pixelHeight)),
			  m_bootstrap_sample_count(static_cast<std::size_t>(scene->options->numInitSamples)),
			  m_large_step_probability(scene->options->largeStepProbability),
			  m_sigma(scene->options->sigma),
			  m_rao_blackwellize(scene->options->rao_blackwellize),
			  m_compute_mse(scene->options->compute_mse),
			  m_realization_count(scene->options->realization_count),
			  m_mlt_state{ scene, GeneratePathBidir, PerturbPathBidir, path_function_library->staticFuncMap, path_function_library->staticDervFuncMap },
			  m_direct_buffer{ scene->camera->film->pixelWidth, scene->camera->film->pixelHeight },
			  m_chain_count{ scene->options->chain_count },
			  m_chain_data(m_realization_count, std::vector<chain_data>(m_chain_count)),
			  m_global_cache(m_realization_count),
#ifdef _DEBUG
			  m_thread_count(1),
#else
			  m_thread_count(std::min(m_chain_count, static_cast<std::size_t>(std::thread::hardware_concurrency()))),
#endif // _DEBUG
			  m_thread_pool(m_thread_count - 1),
			  m_thread_data(m_thread_count, thread_data{ scene->camera->film->pixelWidth, scene->camera->film->pixelHeight }),
			  m_thread_chain_count(m_chain_count / m_thread_count),
			  m_last_thread_chain_count(m_chain_count - (m_thread_count - 1) * m_thread_chain_count),
			  m_frame_period(scene->options->frame_period),
			  m_rendering_time(scene->options->rendering_time)
		{
			init_thead_data();
		}

		void render()
		{
			// compute direct buffer
			DirectLighting(m_scene, m_direct_buffer);

			std::cout << "equal time comparison for ";
			if (m_scene->options->mala)
				std::cout << "mala";
			else if (m_scene->options->h2mc)
				std::cout << "h2mc";
			else
				std::cout << "metropolis";
			std::cout << std::endl << std::flush;

			std::optional<Image3> reference{};
			if (!m_scene->options->reference.empty())
				reference = Image3{ m_scene->options->reference };

			if (!m_compute_mse)
			{
				if (m_rao_blackwellize)
					render<true>(m_scene->outputName, reference);
				else
					render<false>(m_scene->outputName, reference);
			}
			else
			{
				if (reference)
				{
					if (m_rao_blackwellize)
						compute_mse<true>(m_scene->outputName, *reference);
					else
						compute_mse<false>(m_scene->outputName, *reference);
				}
				else
					std::cerr << "cannot compute mse without reference image" << std::endl;
			}
		}

	private:
		struct chain_data
		{
			Chain chain;
			LargeStep large_step;
			SmallStep small_step;
			MarkovState current_state;

			bool initialized = false;
			std::chrono::duration<Float, std::nano> rendering_time;
		}; // struct chain_data

		template<bool RaoBlackwellize>
		void render(std::string const& filename, std::optional<Image3> const& reference = std::nullopt)
		{
			std::ofstream log(filename + ".log", std::ios::binary);

			bootstrap_result bootstrap_result = bootstrap();
			if (bootstrap_result.c == 0)
			{
				std::cout << "black image\n";
				return;
			}

			std::cout << "rendering\n" << std::flush;

			init_chain_data(bootstrap_result);

			Float min_mse = std::numeric_limits<Float>::infinity(),
				max_mse = -min_mse,
				average_mse{};
			std::size_t generated_sample_count;
			for (std::size_t realization = 0; realization < m_realization_count; ++realization)
			{
				Image3 image{ m_scene->camera->film->pixelWidth, m_scene->camera->film->pixelHeight };
				render_result const render_result = render_for<RaoBlackwellize>(bootstrap_result, m_rendering_time, m_chain_data[realization], image);
				
				std::transform(std::execution::par_unseq, image.data.begin(), image.data.end(), image.data.begin(),
					[&](auto const& rgb) { return m_pixel_count * rgb / render_result.generated_sample_count; });

				Vector3 const mse_vec = image.MSE(*reference);
				Float const mse = (mse_vec[0] + mse_vec[1] + mse_vec[2]) / 3;

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
					generated_sample_count = render_result.generated_sample_count;
					WriteImage(filename + ".exr", &image);
				}
			}

			average_mse /= m_realization_count;
			std::cout << "min_mse = " << min_mse << std::endl
				<< "max_mse = " << max_mse << std::endl
				<< "average_mse = " << average_mse << std::endl;
			log << "min_mse = " << min_mse << std::endl
				<< "max_mse = " << max_mse << std::endl
				<< "average_mse = " << average_mse << std::endl;
		}

		template<bool RaoBlackwellize>
		void compute_mse(std::string const& filename, Image3 const& reference)
		{
			std::ofstream log(filename + ".log", std::ios::binary);

			std::ofstream mse_out(filename + ".mse", std::ios::binary);
			mse_out << std::fixed << std::setprecision(4);

			bootstrap_result bootstrap_result = bootstrap();
			if (bootstrap_result.c == 0)
			{
				std::cout << "black image\n";
				return;
			}

			init_chain_data(bootstrap_result);

			Float const scale = m_pixel_count,
				scale_over_realization_count = scale / m_realization_count;

			std::vector<std::size_t> generated_sample_count(m_realization_count);
			std::vector<std::chrono::duration<Float, std::nano>> rendering_time(m_realization_count);
			std::vector<Image3> unscaled_image(m_realization_count, Image3{ m_scene->camera->film->pixelWidth, m_scene->camera->film->pixelHeight });

			for (std::size_t frame = 1;; ++frame)
			{
				if (rendering_time[0] >= m_rendering_time)
					break;

				std::stringstream ss;
				ss << "rendering frame " << frame << std::endl;
				log << ss.str();
				std::cout << ss.str();

				std::vector<Image3> image(m_realization_count, Image3{ m_scene->camera->film->pixelWidth, m_scene->camera->film->pixelHeight });
				Image3 unscaled_image_empirical_mean{ m_scene->camera->film->pixelWidth, m_scene->camera->film->pixelHeight };

				std::chrono::duration<Float, std::nano> rendering_time_average{};
				Float generated_sample_count_average{},
					mse_average{},
					min_mse = std::numeric_limits<Float>::infinity(),
					max_mse = -min_mse;

				for (std::size_t realization = 0; realization < m_realization_count; ++realization)
				{
					render_result const render_result = render_for<RaoBlackwellize>(bootstrap_result, frame * m_frame_period - rendering_time[realization], m_chain_data[realization], unscaled_image[realization]);

					generated_sample_count[realization] += render_result.generated_sample_count;
					rendering_time[realization] += render_result.rendering_time;

					std::for_each_n(std::execution::par_unseq, boost::counting_iterator<std::size_t>{ 0 }, m_pixel_count, [&](std::size_t const i)
					{
						Vector3 const rgb = unscaled_image[realization].data[i] / generated_sample_count[realization];
						image[realization].data[i] = scale * rgb;
						unscaled_image_empirical_mean.data[i] += rgb;
					});

					Vector3 const mse_vec = image[realization].MSE(reference);
					Float const mse = (mse_vec[0] + mse_vec[1] + mse_vec[2]) / 3;

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
						ss.str(std::string{});
						ss << filename << '_' << frame << ".exr";
						WriteImage(ss.str(), &image[realization]);
					}

					generated_sample_count_average += generated_sample_count[realization];
					rendering_time_average += rendering_time[realization];
					mse_average += mse;
				}

				generated_sample_count_average /= m_realization_count;
				rendering_time_average /= m_realization_count;
				mse_average /= m_realization_count;

				Image3 image_empirical_mean{ m_scene->camera->film->pixelWidth, m_scene->camera->film->pixelHeight };
				std::transform(std::execution::par_unseq, unscaled_image_empirical_mean.data.begin(), unscaled_image_empirical_mean.data.end(),
					image_empirical_mean.data.begin(), [&](Vector3 const& rgb) { return scale_over_realization_count * rgb; });

				Float const empirical_variance = std::accumulate(/*std::execution::par_unseq,*/ image.begin(), image.end(), Float{}, [&](auto const& acc, auto const& image)
				{
					Vector3 const var = image.MSE(image_empirical_mean);
					return acc + (var[0] + var[1] + var[2]) / 3;
				}) / (m_realization_count - 1);

				using namespace std::chrono_literals;
				mse_out << generated_sample_count_average / m_pixel_count << '\t'
					<< static_cast<Float>(rendering_time_average.count()) / std::chrono::duration<Float, std::nano>{ 1s }.count() << '\t'
					<< mse_average << '\t'
					<< empirical_variance
					<< std::endl << std::flush;
			}
		}

		struct bootstrap_result
		{
			Float c;
			std::shared_ptr<PiecewiseConstant1D> path_depth_distribution;
		};

		struct render_result
		{
			std::size_t generated_sample_count;
			std::chrono::duration<Float, std::nano> rendering_time;
		};

		template<bool RaoBlackwellize>
		render_result render_for(bootstrap_result& bootstrap_result, std::chrono::duration<Float, std::nano> const& rendering_time, std::vector<chain_data>& chain_data, Image3& image)
		{
			for (std::size_t thread = 0; thread < m_thread_count; ++thread)
			{
				std::fill(std::execution::par_unseq, m_thread_data[thread].image.data.begin(), m_thread_data[thread].image.data.end(), Vector3::Zero());
				m_thread_data[thread].generated_sample_count = 0;
				m_thread_data[thread].rendering_time = std::chrono::high_resolution_clock::duration{};
			}

			auto const run_chain = [&](std::size_t const thread, std::size_t const chain, std::chrono::duration<Float, std::nano> const& rendering_time)
			{
				std::uniform_real_distribution<Float> u;

				auto const begin = std::chrono::high_resolution_clock::now();;

				if (!chain_data[chain].initialized)
				{
					chain_data[chain].initialized = true;

					reset(chain_data[chain].chain);
					reset(chain_data[chain].large_step);
					reset(chain_data[chain].small_step);
					reset(chain_data[chain].current_state);

					do {
						++m_thread_data[thread].generated_sample_count;
					} while (!chain_data[chain].large_step.Mutate(m_mlt_state, bootstrap_result.c, chain_data[chain].current_state, m_thread_data[thread].proposed_state, m_thread_data[thread].g, &chain_data[chain].chain));

					ToSubpath(m_thread_data[thread].proposed_state.spContrib.camDepth,
						m_thread_data[thread].proposed_state.spContrib.lightDepth,
						m_thread_data[thread].proposed_state.path);

					update_current_state(static_cast<MarkovState&>(chain_data[chain].current_state), m_thread_data[thread].proposed_state);
					chain_data[chain].current_state.gaussianInitialized = false;
					chain_data[chain].current_state.valid = true;

					chain_data[chain].large_step.lastScoreSum = chain_data[chain].current_state.scoreSum;
					chain_data[chain].large_step.lastScore = chain_data[chain].current_state.spContrib.lsScore;
					chain_data[chain].chain.buffered = false;

					if constexpr (!RaoBlackwellize)
					{
						for (auto const& splat : chain_data[chain].current_state.toSplat)
							add_splat(m_thread_data[thread].image, splat.screenPos, splat.contrib);
					}

					chain_data[chain].rendering_time = std::chrono::high_resolution_clock::now() - begin;
					if (chain_data[chain].rendering_time >= rendering_time)
					{
						m_thread_data[thread].rendering_time += chain_data[chain].rendering_time;
						return;
					}
				}

				do
				{
					bool const is_large_step = u(m_thread_data[thread].g) < m_large_step_probability;
					Float const alpha = is_large_step ?
						chain_data[chain].large_step.Mutate(m_mlt_state, bootstrap_result.c, chain_data[chain].current_state, m_thread_data[thread].proposed_state, m_thread_data[thread].g, &chain_data[chain].chain) :
						chain_data[chain].small_step.Mutate(m_mlt_state, bootstrap_result.c, chain_data[chain].current_state, m_thread_data[thread].proposed_state, m_thread_data[thread].g, &chain_data[chain].chain);
					++m_thread_data[thread].generated_sample_count;

					if constexpr (RaoBlackwellize)
					{
						if (alpha > 0)
						{
							for (auto const& splat : m_thread_data[thread].proposed_state.toSplat)
								add_splat(m_thread_data[thread].image, splat.screenPos, alpha * splat.contrib);
						}
						if (chain_data[chain].current_state.valid)
						{
							for (auto const& splat : chain_data[chain].current_state.toSplat)
								add_splat(m_thread_data[thread].image, splat.screenPos, (1 - alpha) * splat.contrib);
						}
					}

					if (u(m_thread_data[thread].g) < alpha)
					{
						ToSubpath(m_thread_data[thread].proposed_state.spContrib.camDepth,
							m_thread_data[thread].proposed_state.spContrib.lightDepth,
							m_thread_data[thread].proposed_state.path);

						update_current_state(static_cast<MarkovState&>(chain_data[chain].current_state), m_thread_data[thread].proposed_state);
						chain_data[chain].current_state.valid = true;

						if (is_large_step)
						{
							//if (m_thread_data[thread].chain.buffered && m_thread_data[thread].chain.pathWeight > 1e-10)
							//{
							//	int const d = GetDimension(m_thread_data[thread].proposed_state.path);
							//	if (PSS_MIN_LENGTH <= d && d <= PSS_MAX_LENGTH && !m_global_cache.isReady(d))
							//	{
							//		std::lock_guard<std::mutex> global_cache_lock(m_global_cache.getMutex(d));
							//		m_global_cache.push(d, m_thread_data[thread].chain.pss, m_thread_data[thread].chain.v1, m_thread_data[thread].chain.v2,
							//			m_thread_data[thread].chain.path, m_thread_data[thread].chain.spContrib, m_thread_data[thread].chain.pathWeight);
							//	}
							//}

							chain_data[chain].large_step.lastScoreSum = chain_data[chain].current_state.scoreSum;
							chain_data[chain].large_step.lastScore = chain_data[chain].current_state.spContrib.lsScore;
							chain_data[chain].current_state.gaussianInitialized = false;
							chain_data[chain].chain.buffered = false;
						}
						else
						{
							if constexpr (std::is_same_v<SmallStep, MALASmallStep>)
							{
								if (chain_data[chain].small_step.lastMutationType == MutationType::MALASmall)
								{
									chain_data[chain].chain.g = chain_data[chain].chain.prop_new_g;
									chain_data[chain].chain.v1 = chain_data[chain].chain.prop_new_v1;
									chain_data[chain].chain.v2 = chain_data[chain].chain.prop_new_v2;
									++chain_data[chain].chain.t;
									chain_data[chain].chain.buffered = true;
									chain_data[chain].current_state.gaussianInitialized = true;
								}
							}
						}
					}

					if constexpr (!RaoBlackwellize)
					{
						if (chain_data[chain].current_state.valid)
						{
							for (const auto splat : chain_data[chain].current_state.toSplat)
								add_splat(m_thread_data[thread].image, splat.screenPos, splat.contrib);
						}
					}

					chain_data[chain].rendering_time = std::chrono::high_resolution_clock::now() - begin;
				} while (chain_data[chain].rendering_time < rendering_time);

				m_thread_data[thread].rendering_time += chain_data[chain].rendering_time;
			};

			auto const thread_chain_rendering_time = rendering_time / m_thread_chain_count,
				last_thread_chain_rendering_time = rendering_time / m_last_thread_chain_count;

			auto const work = [&](std::size_t const thread, std::size_t const thread_chain_offset, std::size_t const thread_chain_count,
				std::chrono::duration<Float, std::nano> const& chain_rendering_time)
			{
				for (std::size_t chain = 0; chain < thread_chain_count; ++chain)
					run_chain(thread, thread_chain_offset + chain, chain_rendering_time);
			};

			std::size_t thread_chain_offset{};
			for (std::size_t thread = 0; thread < m_thread_pool.size(); ++thread)
			{
				m_thread_pool[thread] = std::thread{ work, thread, thread_chain_offset, m_thread_chain_count, thread_chain_rendering_time };
				thread_chain_offset += m_thread_chain_count;
			}
			work(m_thread_pool.size(), thread_chain_offset, m_last_thread_chain_count, last_thread_chain_rendering_time);
			for (auto& thread : m_thread_pool)
				thread.join();

			std::size_t generated_sample_count{};
			for (auto& thread_data : m_thread_data)
			{
				generated_sample_count += thread_data.generated_sample_count;
				std::transform(std::execution::par_unseq, image.data.begin(), image.data.end(),
					thread_data.image.data.begin(), image.data.begin(), std::plus<>{});
			}

			return {
				generated_sample_count,
				std::max(std::max_element(std::execution::par_unseq, m_thread_data.begin(), m_thread_data.end(),
					[](auto const& first, auto const& second) { return first.rendering_time < second.rendering_time; })->rendering_time, rendering_time)
			};
		}

		bootstrap_result bootstrap() const
		{
			bootstrap_result result;
			result.c = MLTInit(m_mlt_state, static_cast<int>(m_bootstrap_sample_count), result.path_depth_distribution);
			return result;
		}

		void add_splat(Image3& image, Vector2 const& screen_point, Vector3 const& rgb) const
		{
			if (rgb.allFinite())
			{
				std::size_t const pixel_index =
					std::min(static_cast<int>(screen_point[1] * m_scene->camera->film->pixelHeight), m_scene->camera->film->pixelHeight - 1) * m_scene->camera->film->pixelWidth +
					std::min(static_cast<int>(screen_point[0] * m_scene->camera->film->pixelWidth), m_scene->camera->film->pixelWidth - 1);

				Float const luminance = .212671 * rgb[0] + .715160 * rgb[1] + 0.072169 * rgb[2];


				image.data[pixel_index][0] += luminance;
				image.data[pixel_index][1] += luminance;
				image.data[pixel_index][2] += luminance;
			}
		}

		struct thread_data
		{
			thread_data() = default;
			thread_data(int const pixel_width, int const pixel_height)
				: image{ pixel_width, pixel_height }//,
				  // g{ std::random_device{}() } disabled for reporting purposes
			{}

			RNG g;
			MarkovState proposed_state;
			std::vector<SubpathContrib> proposed_contribution;
			Image3 image;

			std::size_t generated_sample_count;
			std::chrono::duration<Float, std::nano> rendering_time;
		}; // struct thread_data

		void init_chain_data(bootstrap_result const& bootstrap_result)
		{
			for (std::size_t realization = 0; realization < m_realization_count; ++realization)
			{
				m_chain_data[realization].reserve(m_chain_count);
				for (std::size_t i = 0; i < m_chain_count; ++i)
				{
					m_chain_data[realization][i].chain.chainId = static_cast<int>(realization * m_chain_count + i);
					m_chain_data[realization][i].chain.globalCache = &m_global_cache[realization];
					m_chain_data[realization][i].chain.ss = m_sigma;

					m_chain_data[realization][i].large_step.lengthDist = bootstrap_result.path_depth_distribution;
					initialize(m_chain_data[realization][i].small_step, m_scene, m_scene->options->maxDepth, m_sigma);
				}
			}
		}

		// this is only used for reporting purposes
		void init_thead_data()
		{
			for (std::size_t i = 0; i < m_thread_count; ++i)
			{
				m_thread_data[i].g.seed(std::random_device{}());
				//m_thread_data[i].g.seed(static_cast<RNG::result_type>(i));
			}
		}

		Scene const* const m_scene;
		std::shared_ptr<PathFuncLib const> m_path_function_library;

		std::size_t const m_pixel_count,
			m_bootstrap_sample_count,
			m_realization_count,
			m_chain_count,
			m_thread_count,
			m_thread_chain_count,
			m_last_thread_chain_count;
		Float const m_large_step_probability,
			m_sigma;
		bool const m_rao_blackwellize,
			m_compute_mse;

		std::vector<GlobalCache> m_global_cache;
		MLTState const m_mlt_state;
		SampleBuffer m_direct_buffer;

		std::vector<std::vector<chain_data>> m_chain_data;
		std::vector<thread_data> m_thread_data;
		std::vector<std::thread> m_thread_pool;

		std::chrono::duration<Float, std::nano> const m_frame_period,
			m_rendering_time;
	}; // class metropolis_integrator
} // namespace sholl


#endif // !HPP_SHOLL_INTEGRATOR_METROPOLIS_INTEGRATOR_INCLUDED
