# Ranger: CUDA-enabled range helpers for range-based for loops

Facilities for generating simple index ranges for C++ range-based for loops. Includes support for
CUDA grid-stride ranges.

## Examples:


```
// generate values from 0 to N
for (auto i : range(N)) {
  std::cout << i << std::endl;
}
```

```
// generate values from `begin` to `end`
for (auto i : range(begin, end)) {
  std::cout << i << std::endl;
}
```

```
// generate values stepping by `step` from `begin` to `end`
for (auto i : range(begin, end, step)) {
  std::cout << i << std::endl;
}
```

``` 
// generate values from 0 to N in a kernel
__global__ void size_kernel(int N, int* out)
{
  for (auto i : grid_stride_range(N)) {
    out[i] = i;
  }
}
```

```
// generate values from begin to N in a kernel
__global__ void begin_end_kernel(int begin, end, int* out)
{
  for (auto i : grid_stride_range(begin, end)) {
    out[i-begin] = i;
  }
}
```

```
// generate values stepping by `step` from 0 to N in a kernel
__global__ void step_kernel(int N, int step, int* out)
{
  for (auto i : grid_stride_range(0, N, step)) {
    out[i / step] = i;
  }
}
```

```
// This version of grid_stride_range returns an index and an active_mask that excludes
// threads that step outside the range
template <typename Predicate>
__global__ void valid_if_kernel(active_mask_type* output, thread_index_type size, Predicate pred)
{
  constexpr std::int32_t leader_lane{0};
  constexpr std::int32_t warp_size{32};
  thread_index_type const lane_id{threadIdx.x % warp_size};

  active_mask_type initial_active_mask = 0xFFFF'FFFF;

  for (auto [i, active_mask] : grid_stride_range(size, initial_active_mask)) {
    active_mask_type ballot = __ballot_sync(active_mask, pred(i));
    if (lane_id == leader_lane) { output[i / warp_size] = ballot; }
  }
}
```
