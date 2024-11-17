// Jump Restore Light Transport v1.0
// Copyright (c) Sacha Holl 2024
//
// This file is part of MyProject, which is dual-licensed under:
// 1. The Apache License, Version 2.0 (see root LICENSE file)
// 2. A Commercial License (see root COMMERCIAL_LICENSE file)
//
// SPDX-License-Identifier: Apache-2.0 OR Commercial


#ifndef HPP_SHOLL_ATOMIC_REPLACE_IF_INCLUDED
#define HPP_SHOLL_ATOMIC_REPLACE_IF_INCLUDED


#include <atomic>


namespace sholl
{
    template<typename T, class BinaryPredicate>
    void replace_if(std::atomic<T>& old_value, T const& new_value, BinaryPredicate p) noexcept
    {
        T current_value = old_value;
        while (p(current_value, new_value) &&
            !old_value.compare_exchange_weak(current_value, new_value))
        {}
    }
} // [sholl]


#endif // !HPP_SHOLL_ATOMIC_REPLACE_IF_INCLUDED
