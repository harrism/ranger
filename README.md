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
