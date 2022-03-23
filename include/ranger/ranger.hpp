#include <iterator>

#ifdef __CUDACC__
#define DEVICE_CALLABLE __host__ __device__
#else
#define DEVICE_CALLABLE
#endif

namespace detail {

template <typename T>
struct range_iterator : std::iterator<std::input_iterator_tag, T> {
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;
  using pointer           = value_type*;
  using reference         = value_type&;

  DEVICE_CALLABLE
  range_iterator(T value) : value(value) {}

  DEVICE_CALLABLE
  reference operator*() { return value; }

  DEVICE_CALLABLE
  pointer operator->() { return &value; }

  DEVICE_CALLABLE
  range_iterator& operator++()
  {
    value++;
    return *this;
  }

  DEVICE_CALLABLE
  friend bool operator==(range_iterator const& lhs, range_iterator const& rhs)
  {
    return lhs.value == rhs.value;
  };

  DEVICE_CALLABLE
  friend bool operator!=(range_iterator const& lhs, range_iterator const& rhs)
  {
    return lhs.value != rhs.value;
  };

 protected:
  T value;  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
};

template <typename T>
struct step_range_iterator : public range_iterator<T> {
  DEVICE_CALLABLE
  step_range_iterator(T value, T step) : range_iterator<T>{value}, step{step} {}

  using detail::range_iterator<T>::value;

  DEVICE_CALLABLE
  step_range_iterator& operator++()
  {
    value += step;
    return *this;
  }

  DEVICE_CALLABLE
  friend bool operator==(step_range_iterator const& a, step_range_iterator const& b)
  {
    return a.step > 0 ? (a.value >= b.value) : (a.value < b.value);
  }

  DEVICE_CALLABLE
  friend bool operator!=(step_range_iterator const& a, step_range_iterator const& b)
  {
    return !(a == b);
  }

 protected:
  T step;  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
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

using thread_index_type = int64_t;

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

#endif
