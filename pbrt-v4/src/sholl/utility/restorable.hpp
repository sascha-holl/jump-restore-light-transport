// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#ifndef HPP_SHOLL_UTILITY_RESTORABLE_INCLUDED
#define HPP_SHOLL_UTILITY_RESTORABLE_INCLUDED


#include <cstddef>


namespace sholl
{
	template<typename T>
	struct restorable
	{
		restorable() = default;
		explicit restorable(T const value, std::size_t const last_modification_iteration = 0)
			: value(value),
			  value_backup(value),
			  last_modification_iteration(last_modification_iteration),
			  last_modification_iteration_backup(last_modification_iteration)
		{}

		void backup()
		{
			value_backup = value;
			last_modification_iteration_backup = last_modification_iteration;
		}

		void restore()
		{
			value = value_backup;
			last_modification_iteration = last_modification_iteration_backup;
		}

		T value,
			value_backup;
		std::size_t last_modification_iteration,
			last_modification_iteration_backup;
	}; // struct restorable
} // namespace sholl


#endif // !HPP_SHOLL_UTILITY_RESTORABLE_INCLUDED
