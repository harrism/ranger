/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdint>
#include <iterator>

#ifdef __CUDACC__
#define DEVICE_CALLABLE __host__ __device__
#else
#define DEVICE_CALLABLE
#endif

namespace ranger {

namespace detail {

template <typename T>
struct range_iterator : std::iterator<std::input_iterator_tag, T> {
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;
  using pointer           = value_type*;
  using reference         = value_type&;

  DEVICE_CALLABLE
  range_iterator(T value) : _value(value) {}

  DEVICE_CALLABLE
  reference operator*() { return _value; }

  DEVICE_CALLABLE
  pointer operator->() { return &_value; }

  DEVICE_CALLABLE
  range_iterator& operator++()
  {
    _value++;
    return *this;
  }

  DEVICE_CALLABLE
  friend bool operator==(range_iterator const& lhs, range_iterator const& rhs)
  {
    return lhs._value == rhs._value;
  };

  DEVICE_CALLABLE
  friend bool operator!=(range_iterator const& lhs, range_iterator const& rhs)
  {
    return lhs._value != rhs._value;
  };

 protected:
  T _value;  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
};

template <typename T>
struct step_range_iterator : public range_iterator<T> {
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;
  using pointer           = value_type*;
  using reference         = value_type&;

  DEVICE_CALLABLE
  step_range_iterator(T value, T step) : range_iterator<T>{value}, _step{step} {}

  using detail::range_iterator<T>::_value;

  DEVICE_CALLABLE
  step_range_iterator& operator++()
  {
    _value += _step;
    return *this;
  }

  DEVICE_CALLABLE
  friend bool operator==(step_range_iterator const& a, step_range_iterator const& b)
  {
    return a._step > 0 ? (a._value >= b._value) : (a._value < b._value);
  }

  DEVICE_CALLABLE
  friend bool operator!=(step_range_iterator const& a, step_range_iterator const& b)
  {
    return !(a == b);
  }

 protected:
  T _step;  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
};

}  // namespace detail

template <typename T>
struct ranger {
  using value_type    = T;
  using iterator_type = detail::range_iterator<T>;

  DEVICE_CALLABLE
  ranger(T begin, T end) : _begin{begin}, _end{end} {}

  DEVICE_CALLABLE
  iterator_type begin() { return _begin; }

  DEVICE_CALLABLE
  iterator_type end() { return _end; }

 private:
  iterator_type _begin{};
  iterator_type _end{};
};

template <typename T>
struct step_ranger {
  using value_type    = T;
  using iterator_type = detail::step_range_iterator<T>;

  DEVICE_CALLABLE
  step_ranger(T begin, T end, T step) : _begin{begin, step}, _end{end, step} {}

  DEVICE_CALLABLE
  iterator_type begin() { return _begin; }

  DEVICE_CALLABLE
  iterator_type end() { return _end; }

 private:
  iterator_type _begin{};
  iterator_type _end{};
};

template <typename T>
DEVICE_CALLABLE ranger<T> range(T begin, T end)
{
  return {begin, end};
}

template <typename T>
DEVICE_CALLABLE step_ranger<T> range(T begin, T end, T step)
{
  return {begin, end, step};
}

template <typename T>
DEVICE_CALLABLE ranger<T> range(T size)
{
  return {T(), size};
}

#ifdef __CUDACC__

using thread_index_type = std::int64_t;
using active_mask_type  = std::uint32_t;

namespace detail {
template <typename T>
struct masked_step_range_iterator : public range_iterator<T> {
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;
  using pointer           = value_type*;
  using reference         = value_type&;

  __device__ masked_step_range_iterator(T value, T step, T limit, active_mask_type mask)
    : range_iterator<T>{value}, _step{step}, _limit{limit}, _initial_mask{mask}, _mask{mask}
  {
  }

  using detail::range_iterator<T>::_value;

  struct val_and_mask {
    reference ref;
    active_mask_type mask;
  };

  __device__ val_and_mask operator*() { return {_value, _mask}; }

  __device__ masked_step_range_iterator& operator++()
  {
    _value += _step;
    _mask = __ballot_sync(_initial_mask, _value < _limit);
    return *this;
  }

  DEVICE_CALLABLE
  friend bool operator==(masked_step_range_iterator const& a, masked_step_range_iterator const& b)
  {
    return a._step > 0 ? (a._value >= b._value) : (a._value < b._value);
  }

  DEVICE_CALLABLE
  friend bool operator!=(masked_step_range_iterator const& a, masked_step_range_iterator const& b)
  {
    return !(a == b);
  }

 protected:
  T _step;                 // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
  T _limit;                // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
  active_mask_type
    _initial_mask;         // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
  active_mask_type _mask;  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
};

}  // namespace detail

template <typename T>
struct masked_step_ranger {
  using value_type    = T;
  using iterator_type = detail::masked_step_range_iterator<T>;

  DEVICE_CALLABLE
  masked_step_ranger(T begin, T end, T step, active_mask_type mask)
    : _begin{begin, step, end, mask}, _end{end, step, end, mask}
  {
  }

  DEVICE_CALLABLE
  iterator_type begin() { return _begin; }

  DEVICE_CALLABLE
  iterator_type end() { return _end; }

 private:
  iterator_type _begin{};
  iterator_type _end{};
};

template <typename T>
__device__ step_ranger<thread_index_type> grid_stride_range(T size)
{
  return {
    thread_index_type{blockDim.x} * thread_index_type{blockIdx.x} + thread_index_type{threadIdx.x},
    thread_index_type{size},
    thread_index_type{gridDim.x} * thread_index_type{blockDim.x}};
}

template <typename T>
__device__ step_ranger<thread_index_type> grid_stride_range(T begin, T end)
{
  return {thread_index_type{begin} + thread_index_type{blockDim.x} * thread_index_type{blockIdx.x} +
            thread_index_type{threadIdx.x},
          thread_index_type{end},
          thread_index_type{gridDim.x} * thread_index_type{blockDim.x}};
}

template <typename T>
__device__ step_ranger<thread_index_type> grid_stride_range(T begin, T end, T step)
{
  return {thread_index_type{step} * (thread_index_type{begin} +
                                     thread_index_type{blockDim.x} * thread_index_type{blockIdx.x} +
                                     thread_index_type{threadIdx.x}),
          thread_index_type{end},
          thread_index_type{step} * thread_index_type{gridDim.x} * thread_index_type{blockDim.x}};
}

template <typename T>
__device__ masked_step_ranger<thread_index_type> grid_stride_range(T size, active_mask_type mask)
{
  return {
    thread_index_type{blockDim.x} * thread_index_type{blockIdx.x} + thread_index_type{threadIdx.x},
    thread_index_type{size},
    thread_index_type{gridDim.x} * thread_index_type{blockDim.x},
    mask};
}

template <typename T>
__device__ masked_step_ranger<thread_index_type> grid_stride_range(T begin,
                                                                   T end,
                                                                   active_mask_type mask)
{
  return {thread_index_type{begin} + thread_index_type{blockDim.x} * thread_index_type{blockIdx.x} +
            thread_index_type{threadIdx.x},
          thread_index_type{end},
          thread_index_type{gridDim.x} * thread_index_type{blockDim.x},
          mask};
}

template <typename T>
__device__ masked_step_ranger<thread_index_type> grid_stride_range(T begin,
                                                                   T end,
                                                                   T step,
                                                                   active_mask_type mask)
{
  return {thread_index_type{step} * (thread_index_type{begin} +
                                     thread_index_type{blockDim.x} * thread_index_type{blockIdx.x} +
                                     thread_index_type{threadIdx.x}),
          thread_index_type{end},
          thread_index_type{step} * thread_index_type{gridDim.x} * thread_index_type{blockDim.x},
          mask};
}

#endif

}  // namespace ranger
