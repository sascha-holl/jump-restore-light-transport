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
#include <chrono>
#include <execution>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <type_traits>

#include "../../image.h"
#include "../../path.h"
#include "../../scene.h"

#include "../utility.hpp"
#include "../random/metropolis_restore_sampler.hpp"


namespace sholl
{
	template<class LargeStep, class SmallStep>
	class metropolis_restore_integrator
	{
	public:
		metropolis_restore_integrator(Scene const* scene, std::shared_ptr<PathFuncLib const> const& path_function_library)
			: m_scene(scene),
			  m_path_function_library(path_function_library),
			  m_k_max(static_cast<std::size_t>(scene->options->maxDepth) + 1),
			  m_pixel_count(static_cast<std::size_t>(scene->camera->film->pixelWidth * scene->camera->film->pixelHeight)),
			  m_bootstrap_sample_count(static_cast<std::size_t>(scene->options->numInitSamples)),
			  m_sigma(scene->options->sigma),
			  m_expected_lifetime(scene->options->expected_lifetime),
			  m_multiplex_weight_correction{ (static_cast<Float>(m_k_max * m_k_max + 3 * m_k_max) / 2 - 1) / m_k_max },
			  m_rao_blackwellize(scene->options->rao_blackwellize),
			  m_compute_mse(scene->options->compute_mse),
			  m_realization_count(scene->options->realization_count),
			  m_mlt_state{ scene, GeneratePathBidir, PerturbPathBidir, path_function_library->staticFuncMap, path_function_library->staticDervFuncMap },
			  m_direct_buffer{ scene->camera->film->pixelWidth, scene->camera->film->pixelHeight },
#ifdef _DEBUG
			  m_thread_count(1),
#else
			  m_thread_count(scene->options->thread_count_per_core * std::thread::hardware_concurrency()),
#endif // _DEBUG
			  m_thread_pool(m_thread_count - 1),
			  m_thread_data(m_thread_count, thread_data{ m_scene->camera->film->pixelWidth, m_scene->camera->film->pixelHeight }),
			  m_frame_period(scene->options->frame_period),
			  m_rendering_time(scene->options->rendering_time)
		{}

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
			std::cout << " restore" << std::endl << std::flush;

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

			init_thread_data(bootstrap_result);

			Float min_mse = std::numeric_limits<Float>::infinity(),
				max_mse = -min_mse,
				average_mse{};
			std::size_t generated_sample_count;
			std::chrono::duration<Float, std::nano> actual_rendering_time;
			for (std::size_t realization = 0; realization < m_realization_count; ++realization)
			{
				Image3 image{ m_scene->camera->film->pixelWidth, m_scene->camera->film->pixelHeight };
				render_result const render_result = render_for<RaoBlackwellize>(bootstrap_result, m_rendering_time, image);

				double const s = m_pixel_count / render_result.tau;
				std::transform(std::execution::par_unseq, image.data.begin(), image.data.end(), image.data.begin(),
					[&](auto const& rgb) { return s * rgb; });

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
					actual_rendering_time = render_result.actual_rendering_time;
					WriteImage(filename + ".exr", &image);
				}
			}

			using namespace std::chrono_literals;
			average_mse /= m_realization_count;
			std::cout << "min_mse = " << min_mse << std::endl
				<< "max_mse = " << max_mse << std::endl
				<< "average_mse = " << average_mse << std::endl
				<< "actual_rendering_time = " << static_cast<Float>(actual_rendering_time.count()) / std::chrono::duration<Float, std::nano>{ 1s }.count() << std::endl;
			log << "min_mse = " << min_mse << std::endl
				<< "max_mse = " << max_mse << std::endl
				<< "average_mse = " << average_mse << std::endl
				<< "actual_rendering_time = " << static_cast<Float>(actual_rendering_time.count()) / std::chrono::duration<Float, std::nano>{ 1s }.count() << std::endl;
		}

		template<bool RaoBlackwellize>
		void compute_mse(std::string const& filename, Image3 const& reference)
		{
			std::ofstream log(filename + ".log", std::ios::binary);

			std::ofstream mse_out(filename + ".mse", std::ios::binary);
			std::ofstream actual_mse_out(filename + ".amse", std::ios::binary);
			mse_out << std::fixed << std::setprecision(4);
			actual_mse_out << std::fixed << std::setprecision(4);

			bootstrap_result bootstrap_result = bootstrap();
			if (bootstrap_result.c == 0)
			{
				std::cout << "black image\n";
				return;
			}

			init_thread_data(bootstrap_result);

			Float const scale = m_pixel_count,
				scale_over_realization_count = scale / m_realization_count;

			std::vector<std::size_t> generated_sample_count(m_realization_count);
			std::vector<double> tau(m_realization_count);
			std::vector<std::chrono::duration<Float, std::nano>> rendering_time(m_realization_count),
				actual_rendering_time(m_realization_count);
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

				std::chrono::duration<Float, std::nano> rendering_time_average{},
					actual_rendering_time_average{};
				Float generated_sample_count_average{},
					mse_average{},
					min_mse = std::numeric_limits<Float>::infinity(),
					max_mse = -min_mse;

				for (std::size_t realization = 0; realization < m_realization_count; ++realization)
				{
					render_result const render_result = render_for<RaoBlackwellize>(bootstrap_result, frame * m_frame_period - rendering_time[realization], unscaled_image[realization]);

					generated_sample_count[realization] += render_result.generated_sample_count;
					tau[realization] += render_result.tau;
					rendering_time[realization] += render_result.rendering_time;
					actual_rendering_time[realization] += render_result.actual_rendering_time;

					std::for_each_n(std::execution::par_unseq, boost::counting_iterator<std::size_t>{ 0 }, m_pixel_count, [&](std::size_t const i)
					{
						Vector3 const rgb = unscaled_image[realization].data[i] / tau[realization];
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
					actual_rendering_time_average += actual_rendering_time[realization];
					mse_average += mse;
				}

				generated_sample_count_average /= m_realization_count;
				rendering_time_average /= m_realization_count;
				actual_rendering_time_average /= m_realization_count;
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
				actual_mse_out << generated_sample_count_average / m_pixel_count << '\t'
					<< static_cast<Float>(rendering_time_average.count()) / std::chrono::duration<Float, std::nano>{ 1s }.count() << '\t'
					<< mse_average << '\t'
					<< empirical_variance << '\t'
					<< static_cast<Float>(actual_rendering_time_average.count()) / std::chrono::duration<Float, std::nano>{ 1s }.count()
					<< std::endl << std::flush;
			}
		}

		struct rao_blackwellization_state
			: public MarkovState
		{
			Float weight,
				xi;
		}; // struct rao_blackwellization_state

		template<bool RaoBlackwellize>
		using state = std::conditional_t<RaoBlackwellize, rao_blackwellization_state, MarkovState>;

		struct bootstrap_result
		{
			Float c;
			std::vector<std::atomic<Float>> c_per_path_length;
			std::shared_ptr<PiecewiseConstant1D> path_depth_distribution;
		};

		struct render_result
		{
			double tau;
			std::size_t generated_sample_count;
			std::chrono::duration<Float, std::nano> rendering_time,
				actual_rendering_time;
		};

		template<bool RaoBlackwellize>
		render_result render(bootstrap_result& bootstrap_result, std::size_t const sample_count, Image3& image)
		{
			std::size_t const thread_sample_count = sample_count / m_thread_count,
				last_thread_sample_count = sample_count - m_thread_pool.size() * thread_sample_count;

			auto const work = [&](std::size_t const thread, std::size_t const thread_sample_count)
			{
				m_thread_data[thread].generated_sample_count = 0;
				m_thread_data[thread].tau = 0;

				std::uniform_real_distribution<Float> u;

				for (;;)
				{
					reset(m_thread_data[thread].chain);
					reset(static_cast<MarkovState&>(m_thread_data[thread].current_state));
					reset(m_thread_data[thread].large_step);
					reset(m_thread_data[thread].small_step);

					Float const large_step_alpha = m_thread_data[thread].large_step.Mutate(m_mlt_state, bootstrap_result.c, m_thread_data[thread].current_state, m_thread_data[thread].proposed_state, m_thread_data[thread].g, &m_thread_data[thread].chain);
					++m_thread_data[thread].generated_sample_count;

					if (large_step_alpha)
					{
						ToSubpath(m_thread_data[thread].proposed_state.spContrib.camDepth,
							m_thread_data[thread].proposed_state.spContrib.lightDepth,
							m_thread_data[thread].proposed_state.path);

						//std::swap(static_cast<MarkovState&>(m_thread_data[thread].current_state), m_thread_data[thread].proposed_state);
						update_current_state(static_cast<MarkovState&>(m_thread_data[thread].current_state), m_thread_data[thread].proposed_state);
						m_thread_data[thread].current_state.gaussianInitialized = false;
						m_thread_data[thread].current_state.valid = true;

						m_thread_data[thread].chain.buffered = false;

						if constexpr (RaoBlackwellize)
							m_thread_data[thread].current_state.xi = m_thread_data[thread].current_state.weight = 1;

						int const length = GetPathLength(m_thread_data[thread].current_state.spContrib.camDepth, m_thread_data[thread].current_state.spContrib.lightDepth),
							strategy_count = length > 1 ? (m_scene->options->bidirectional ? length + 1 : 2) : 1;
						Float const regeneration_density_at_current_state = bootstrap_result.path_depth_distribution->Pmf(length) / strategy_count;

						metropolis_restore_sampler<Float> restore_sampler{ 1, bootstrap_result.c, m_expected_lifetime };

						Float tau,
							lambda;
						while (restore_sampler.try_begin_iteration(regeneration_density_at_current_state,
							m_thread_data[thread].current_state.spContrib.lsScore * m_multiplex_weight_correction,
							m_thread_data[thread].g, tau, lambda))
						{
							if constexpr (!RaoBlackwellize)
							{
								m_thread_data[thread].tau += tau;
								if (m_thread_data[thread].current_state.valid)
								{
									for (auto const& splat : m_thread_data[thread].current_state.toSplat)
										add_splat(m_thread_data[thread].image, splat.screenPos, tau * splat.contrib);
								}
							}

							Float const alpha = m_thread_data[thread].small_step.Mutate(
								m_mlt_state, bootstrap_result.c, m_thread_data[thread].current_state, m_thread_data[thread].proposed_state, m_thread_data[thread].g, &m_thread_data[thread].chain);
							++m_thread_data[thread].generated_sample_count;

							if (restore_sampler.end_iteration(alpha, m_thread_data[thread].g))
							{
								if constexpr (RaoBlackwellize)
								{
									Float const w = m_thread_data[thread].current_state.xi / lambda;
									m_thread_data[thread].tau += w;
									if (m_thread_data[thread].current_state.valid)
									{
										for (auto const& splat : m_thread_data[thread].current_state.toSplat)
											add_splat(m_thread_data[thread].image, splat.screenPos, w * splat.contrib);
									}
								}

								ToSubpath(m_thread_data[thread].proposed_state.spContrib.camDepth,
									m_thread_data[thread].proposed_state.spContrib.lightDepth,
									m_thread_data[thread].proposed_state.path);

								//std::swap(static_cast<MarkovState&>(m_thread_data[thread].current_state), m_thread_data[thread].proposed_state);
								update_current_state(static_cast<MarkovState&>(m_thread_data[thread].current_state), m_thread_data[thread].proposed_state);
								m_thread_data[thread].current_state.valid = true;

								if constexpr (RaoBlackwellize)
									m_thread_data[thread].current_state.weight = m_thread_data[thread].current_state.xi = 1;

								if constexpr (std::is_same_v<SmallStep, MALASmallStep>)
								{
									if (m_thread_data[thread].small_step.lastMutationType == MutationType::MALASmall)
									{
										m_thread_data[thread].chain.g = m_thread_data[thread].chain.prop_new_g;
										m_thread_data[thread].chain.v1 = m_thread_data[thread].chain.prop_new_v1;
										m_thread_data[thread].chain.v2 = m_thread_data[thread].chain.prop_new_v2;
										++m_thread_data[thread].chain.t;
										m_thread_data[thread].chain.buffered = true;
										m_thread_data[thread].current_state.gaussianInitialized = true;
									}
								}
							}
							else
							{
								if constexpr (RaoBlackwellize)
									m_thread_data[thread].current_state.xi += m_thread_data[thread].current_state.weight *= 1 - alpha;
							}
						}

						if constexpr (!RaoBlackwellize)
						{
							m_thread_data[thread].tau += tau;
							if (m_thread_data[thread].current_state.valid)
							{
								for (auto const& splat : m_thread_data[thread].current_state.toSplat)
									add_splat(m_thread_data[thread].image, splat.screenPos, tau * splat.contrib);
							}
						}
						else
						{
							Float const w = m_thread_data[thread].current_state.xi / lambda;
							m_thread_data[thread].tau += w;
							if (m_thread_data[thread].current_state.valid)
							{
								for (auto const& splat : m_thread_data[thread].current_state.toSplat)
									add_splat(m_thread_data[thread].image, splat.screenPos, w * splat.contrib);
							}
						}
					}

					if (m_thread_data[thread].generated_sample_count >= thread_sample_count) [[unlikely]]
						break;
				}
			};

			for (std::size_t thread = 0; thread < m_thread_count; ++thread)
				std::fill(std::execution::par_unseq, m_thread_data[thread].image.data.begin(), m_thread_data[thread].image.data.end(), Vector3::Zero());

			auto const begin = std::chrono::high_resolution_clock::now();
			for (std::size_t thread = 0; thread < m_thread_pool.size(); ++thread)
				m_thread_pool[thread] = std::thread{ work, thread, thread_sample_count };
			work(m_thread_pool.size(), last_thread_sample_count);
			for (auto& thread : m_thread_pool)
				thread.join();

			std::size_t generated_sample_count{};
			double tau{};
			for (std::size_t thread = 0; thread < m_thread_count; ++thread)
			{
				generated_sample_count += m_thread_data[thread].generated_sample_count;
				tau += m_thread_data[thread].tau;
				std::transform(std::execution::par_unseq, image.data.begin(), image.data.end(),
					m_thread_data[thread].image.data.begin(), image.data.begin(), std::plus<>{});
			}

			return { tau, generated_sample_count, std::chrono::high_resolution_clock::now() - begin };
		}

		template<bool RaoBlackwellize>
		render_result render_for(bootstrap_result const& bootstrap_result, std::chrono::duration<Float, std::nano> const& rendering_time, Image3& image)
		{
			auto const work = [&](std::size_t const thread, std::chrono::duration<Float, std::nano> const& rendering_time)
			{
				std::fill(std::execution::par_unseq, m_thread_data[thread].image.data.begin(), m_thread_data[thread].image.data.end(), Vector3::Zero());
				m_thread_data[thread].tau = 0;
				m_thread_data[thread].generated_sample_count = 0;

				std::uniform_real_distribution<Float> u;
				metropolis_restore_sampler<Float> restore_sampler{ 1, bootstrap_result.c, m_expected_lifetime };

				for (auto const begin = std::chrono::high_resolution_clock::now();;)
				{
					reset(m_thread_data[thread].chain);
					reset(static_cast<MarkovState&>(m_thread_data[thread].current_state));
					reset(m_thread_data[thread].large_step);
					reset(m_thread_data[thread].small_step);

					Float const large_step_alpha = m_thread_data[thread].large_step.Mutate(m_mlt_state, bootstrap_result.c, m_thread_data[thread].current_state, m_thread_data[thread].proposed_state, m_thread_data[thread].g, &m_thread_data[thread].chain);
					++m_thread_data[thread].generated_sample_count;

					if (large_step_alpha)
					{
						ToSubpath(m_thread_data[thread].proposed_state.spContrib.camDepth,
							m_thread_data[thread].proposed_state.spContrib.lightDepth,
							m_thread_data[thread].proposed_state.path);

						update_current_state(static_cast<MarkovState&>(m_thread_data[thread].current_state), m_thread_data[thread].proposed_state);
						m_thread_data[thread].current_state.gaussianInitialized = false;
						m_thread_data[thread].current_state.valid = true;

						m_thread_data[thread].chain.buffered = false;

						if constexpr (RaoBlackwellize)
							m_thread_data[thread].current_state.xi = m_thread_data[thread].current_state.weight = 1;

						int const length = GetPathLength(m_thread_data[thread].current_state.spContrib.camDepth, m_thread_data[thread].current_state.spContrib.lightDepth),
							strategy_count = length > 1 ? length + 1 : 1;
						Float const regeneration_density_at_current_state = bootstrap_result.path_depth_distribution->Pmf(length) / strategy_count;

						Float tau,
							lambda;
						while (restore_sampler.try_begin_iteration(regeneration_density_at_current_state,
							m_thread_data[thread].current_state.spContrib.lsScore * m_multiplex_weight_correction,
							m_thread_data[thread].g, tau, lambda))
						{
							if constexpr (!RaoBlackwellize)
							{
								m_thread_data[thread].tau += tau;
								if (m_thread_data[thread].current_state.valid)
								{
									for (auto const& splat : m_thread_data[thread].current_state.toSplat)
										add_splat(m_thread_data[thread].image, splat.screenPos, tau * splat.contrib);
								}
							}

							Float const alpha = m_thread_data[thread].small_step.Mutate(
								m_mlt_state, bootstrap_result.c, m_thread_data[thread].current_state, m_thread_data[thread].proposed_state, m_thread_data[thread].g, &m_thread_data[thread].chain);
							++m_thread_data[thread].generated_sample_count;

							if (restore_sampler.end_iteration(alpha, m_thread_data[thread].g))
							{
								if constexpr (RaoBlackwellize)
								{
									Float const w = m_thread_data[thread].current_state.xi / lambda;
									m_thread_data[thread].tau += w;
									if (m_thread_data[thread].current_state.valid)
									{
										for (auto const& splat : m_thread_data[thread].current_state.toSplat)
											add_splat(m_thread_data[thread].image, splat.screenPos, w * splat.contrib);
									}
								}

								ToSubpath(m_thread_data[thread].proposed_state.spContrib.camDepth,
									m_thread_data[thread].proposed_state.spContrib.lightDepth,
									m_thread_data[thread].proposed_state.path);

								update_current_state(static_cast<MarkovState&>(m_thread_data[thread].current_state), m_thread_data[thread].proposed_state);
								m_thread_data[thread].current_state.valid = true;

								if constexpr (RaoBlackwellize)
									m_thread_data[thread].current_state.weight = m_thread_data[thread].current_state.xi = 1;

								if constexpr (std::is_same_v<SmallStep, MALASmallStep>)
								{
									if (m_thread_data[thread].small_step.lastMutationType == MutationType::MALASmall)
									{
										m_thread_data[thread].chain.g = m_thread_data[thread].chain.prop_new_g;
										m_thread_data[thread].chain.v1 = m_thread_data[thread].chain.prop_new_v1;
										m_thread_data[thread].chain.v2 = m_thread_data[thread].chain.prop_new_v2;
										++m_thread_data[thread].chain.t;
										m_thread_data[thread].chain.buffered = true;
										m_thread_data[thread].current_state.gaussianInitialized = true;
									}
								}
							}
							else
							{
								if constexpr (RaoBlackwellize)
									m_thread_data[thread].current_state.xi += m_thread_data[thread].current_state.weight *= 1 - alpha;
							}
						}

						if constexpr (!RaoBlackwellize)
						{
							m_thread_data[thread].tau += tau;
							if (m_thread_data[thread].current_state.valid)
							{
								for (auto const& splat : m_thread_data[thread].current_state.toSplat)
									add_splat(m_thread_data[thread].image, splat.screenPos, tau * splat.contrib);
							}
						}
						else
						{
							Float const w = m_thread_data[thread].current_state.xi / lambda;
							m_thread_data[thread].tau += w;
							if (m_thread_data[thread].current_state.valid)
							{
								for (auto const& splat : m_thread_data[thread].current_state.toSplat)
									add_splat(m_thread_data[thread].image, splat.screenPos, w * splat.contrib);
							}
						}
					}

					m_thread_data[thread].rendering_time = std::chrono::high_resolution_clock::now() - begin;
					if (m_thread_data[thread].rendering_time >= rendering_time)
						break;
				}
			};

			auto const begin = std::chrono::high_resolution_clock::now();

			for (std::size_t thread = 0; thread < m_thread_pool.size(); ++thread)
				m_thread_pool[thread] = std::thread{ work, thread, rendering_time };
			work(m_thread_pool.size(), rendering_time);
			for (auto& thread : m_thread_pool)
				thread.join();

			auto const end = std::chrono::high_resolution_clock::now();

			std::size_t generated_sample_count{};
			double tau{};
			for (std::size_t thread = 0; thread < m_thread_count; ++thread)
			{
				generated_sample_count += m_thread_data[thread].generated_sample_count;
				tau += m_thread_data[thread].tau;
				std::transform(std::execution::par_unseq, image.data.begin(), image.data.end(),
					m_thread_data[thread].image.data.begin(), image.data.begin(), std::plus<>{});
			}

			return {
				tau,
				generated_sample_count,
				rendering_time,
				end - begin
			};
		}

		template<bool RaoBlackwellize>
		render_result render_for2(bootstrap_result& bootstrap_result, std::chrono::duration<Float, std::nano> const& rendering_time, Image3& image)
		{
			auto const work = [&](std::size_t const thread, std::chrono::duration<Float, std::nano> const& rendering_time)
			{
				std::fill(std::execution::par_unseq, m_thread_data[thread].image.data.begin(), m_thread_data[thread].image.data.end(), Vector3::Zero());
				m_thread_data[thread].tau = 0;
				m_thread_data[thread].generated_sample_count = 0;

				metropolis_restore_sampler<Float> restore_sampler{ 1, bootstrap_result.c, m_expected_lifetime };
				std::uniform_real_distribution<Float> u;

				for (auto const begin = std::chrono::high_resolution_clock::now();;)
				{
					reset(m_thread_data[thread].chain);
					reset(static_cast<MarkovState&>(m_thread_data[thread].current_state));
					reset(m_thread_data[thread].large_step);
					reset(m_thread_data[thread].small_step);

					Float const large_step_alpha = m_thread_data[thread].large_step.Mutate(m_mlt_state, bootstrap_result.c, m_thread_data[thread].current_state, m_thread_data[thread].proposed_state, m_thread_data[thread].g, &m_thread_data[thread].chain);
					++m_thread_data[thread].generated_sample_count;

					if (large_step_alpha)
					{
						ToSubpath(m_thread_data[thread].proposed_state.spContrib.camDepth,
							m_thread_data[thread].proposed_state.spContrib.lightDepth,
							m_thread_data[thread].proposed_state.path);

						update_current_state(static_cast<MarkovState&>(m_thread_data[thread].current_state), m_thread_data[thread].proposed_state);
						m_thread_data[thread].current_state.gaussianInitialized = false;
						m_thread_data[thread].current_state.valid = true;

						m_thread_data[thread].chain.buffered = false;

						if constexpr (RaoBlackwellize)
							m_thread_data[thread].current_state.xi = m_thread_data[thread].current_state.weight = 1;

						int const length = GetPathLength(m_thread_data[thread].current_state.spContrib.camDepth, m_thread_data[thread].current_state.spContrib.lightDepth),
							strategy_count = length > 1 ? (m_scene->options->bidirectional ? length + 1 : 2) : 1;
						Float const regeneration_density_at_current_state = bootstrap_result.path_depth_distribution->Pmf(length) / strategy_count;

						Float tau,
							lambda;
						while (restore_sampler.try_begin_iteration(regeneration_density_at_current_state,
							m_thread_data[thread].current_state.spContrib.lsScore * m_multiplex_weight_correction,
							m_thread_data[thread].g, tau, lambda))
						{
							if constexpr (!RaoBlackwellize)
							{
								m_thread_data[thread].tau += tau;
								if (m_thread_data[thread].current_state.valid)
								{
									for (auto const& splat : m_thread_data[thread].current_state.toSplat)
										add_splat(m_thread_data[thread].image, splat.screenPos, tau * splat.contrib);
								}
							}

							Float const alpha = m_thread_data[thread].small_step.Mutate(
								m_mlt_state, bootstrap_result.c, m_thread_data[thread].current_state, m_thread_data[thread].proposed_state, m_thread_data[thread].g, &m_thread_data[thread].chain);
							++m_thread_data[thread].generated_sample_count;

							if (restore_sampler.end_iteration(alpha, m_thread_data[thread].g))
							{
								if constexpr (RaoBlackwellize)
								{
									Float const w = m_thread_data[thread].current_state.xi / lambda;
									m_thread_data[thread].tau += w;
									if (m_thread_data[thread].current_state.valid)
									{
										for (auto const& splat : m_thread_data[thread].current_state.toSplat)
											add_splat(m_thread_data[thread].image, splat.screenPos, w * splat.contrib);
									}
								}

								ToSubpath(m_thread_data[thread].proposed_state.spContrib.camDepth,
									m_thread_data[thread].proposed_state.spContrib.lightDepth,
									m_thread_data[thread].proposed_state.path);

								update_current_state(static_cast<MarkovState&>(m_thread_data[thread].current_state), m_thread_data[thread].proposed_state);
								m_thread_data[thread].current_state.valid = true;

								if constexpr (RaoBlackwellize)
									m_thread_data[thread].current_state.weight = m_thread_data[thread].current_state.xi = 1;

								if constexpr (std::is_same_v<SmallStep, MALASmallStep>)
								{
									if (m_thread_data[thread].small_step.lastMutationType == MutationType::MALASmall)
									{
										m_thread_data[thread].chain.g = m_thread_data[thread].chain.prop_new_g;
										m_thread_data[thread].chain.v1 = m_thread_data[thread].chain.prop_new_v1;
										m_thread_data[thread].chain.v2 = m_thread_data[thread].chain.prop_new_v2;
										++m_thread_data[thread].chain.t;
										m_thread_data[thread].chain.buffered = true;
										m_thread_data[thread].current_state.gaussianInitialized = true;
									}
								}
							}
							else
							{
								if constexpr (RaoBlackwellize)
									m_thread_data[thread].current_state.xi += m_thread_data[thread].current_state.weight *= 1 - alpha;
							}
						}

						if constexpr (!RaoBlackwellize)
						{
							m_thread_data[thread].tau += tau;
							if (m_thread_data[thread].current_state.valid)
							{
								for (auto const& splat : m_thread_data[thread].current_state.toSplat)
									add_splat(m_thread_data[thread].image, splat.screenPos, tau * splat.contrib);
							}
						}
						else
						{
							Float const w = m_thread_data[thread].current_state.xi / lambda;
							m_thread_data[thread].tau += w;
							if (m_thread_data[thread].current_state.valid)
							{
								for (auto const& splat : m_thread_data[thread].current_state.toSplat)
									add_splat(m_thread_data[thread].image, splat.screenPos, w * splat.contrib);
							}
						}
					}

					m_thread_data[thread].rendering_time = std::chrono::high_resolution_clock::now() - begin;
					if (m_thread_data[thread].rendering_time >= rendering_time) [[unlikely]]
						break;
				}
			};

			for (std::size_t thread = 0; thread < m_thread_pool.size(); ++thread)
				m_thread_pool[thread] = std::thread{ work, thread, rendering_time };
			work(m_thread_pool.size(), rendering_time);
			for (auto& thread : m_thread_pool)
				thread.join();

			std::size_t generated_sample_count{};
			double tau{};
			for (std::size_t thread = 0; thread < m_thread_count; ++thread)
			{
				generated_sample_count += m_thread_data[thread].generated_sample_count;
				tau += m_thread_data[thread].tau;
				std::transform(std::execution::par_unseq, image.data.begin(), image.data.end(),
					m_thread_data[thread].image.data.begin(), image.data.begin(), std::plus<>{});
			}

			return {
				tau,
				generated_sample_count,
				std::max_element(std::execution::par_unseq, m_thread_data.begin(), m_thread_data.end(),
					[](auto const& first, auto const& second) { return first.rendering_time < second.rendering_time; })->rendering_time
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

				image.data[pixel_index][0] += rgb[0];
				image.data[pixel_index][1] += rgb[1];
				image.data[pixel_index][2] += rgb[2];
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
			double tau;
			Chain chain;
			state<true> current_state;
			Float regeneration_density_at_current_state;
			MarkovState proposed_state;
			Image3 image;

			LargeStep large_step;
			SmallStep small_step;

			bool initialized = false;
			std::size_t generated_sample_count;
			std::chrono::duration<Float, std::nano> rendering_time;
		}; // struct thread_data

		void init_thread_data(bootstrap_result const& bootstrap_result)
		{
			for (std::size_t i = 0; i < m_thread_count; ++i)
			{
				m_thread_data[i].g.seed(std::random_device{}());
				//m_thread_data[i].g.seed(static_cast<RNG::result_type>(i));

				m_thread_data[i].chain.chainId = static_cast<int>(i);
				m_thread_data[i].chain.globalCache = &m_global_cache;
				m_thread_data[i].chain.ss = m_sigma;

				m_thread_data[i].large_step.lengthDist = bootstrap_result.path_depth_distribution;
				initialize(m_thread_data[i].small_step, m_scene, m_scene->options->maxDepth, m_sigma);
			}
		}

		Scene const* const m_scene;
		std::shared_ptr<PathFuncLib const> m_path_function_library;

		std::size_t const m_k_max,
			m_pixel_count,
			m_bootstrap_sample_count,
			m_realization_count,
			m_thread_count;
		Float const m_expected_lifetime,
			m_sigma,
			m_multiplex_weight_correction;
		bool const m_rao_blackwellize,
			m_compute_mse;

		GlobalCache m_global_cache;
		MLTState const m_mlt_state;
		SampleBuffer m_direct_buffer;

		std::vector<thread_data> m_thread_data;
		std::vector<std::thread> m_thread_pool;

		std::chrono::duration<Float, std::nano> const m_frame_period,
			m_rendering_time;
	}; // class metropolis_restore_integrator
} // namespace sholl


#endif // !HPP_SHOLL_INTEGRATOR_METROPOLIS_RESTORE_INTEGRATOR_INCLUDED
