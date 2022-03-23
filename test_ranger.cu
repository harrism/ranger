#include <ranger/ranger.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <thrust/device_vector.h>

#include <cstdio>
#include <iostream>

__global__ void size_kernel(int N, int* out)
{
  for (auto i : grid_stride_range(N)) {
    out[i] = i;
  }
}

__global__ void begin_end_kernel(int begin, int end, int* out)
{
  for (auto i : grid_stride_range(begin, end)) {
    out[i - begin] = i;
  }
}

__global__ void step_kernel(int N, int step, int* out)
{
  for (auto i : grid_stride_range(0, N, step)) {
    out[i / step] = i;
  }
}

TEST_CASE("Counting in a loop", "[range]")
{
  SECTION("Counting with a size-only range matches iota")
  {
    auto N = 100;
    std::vector<int> expected(N, 0);
    std::iota(expected.begin(), expected.end(), 0);

    std::vector<int> data(N, 0);

    for (auto i : range(N)) {
      data[i] = i;
    }

    REQUIRE_THAT(data, Catch::Equals(expected));
  }

  SECTION("Counting with a begin-end range matches transformed iota")
  {
    auto N      = 100;
    auto offset = GENERATE(0, 1, 3, 42);
    std::vector expected(N, 0);
    std::iota(expected.begin(), expected.end(), 0);
    std::transform(
      expected.begin(), expected.end(), expected.begin(), [offset](auto i) { return i + offset; });
    std::vector<int> data(N, 0);

    for (auto i : range(offset, N + offset)) {
      data[i - offset] = i;
    }

    REQUIRE_THAT(data, Catch::Equals(expected));
  }

  SECTION("Counting with a stepped range matches transformed iota")
  {
    auto N    = 100;
    auto step = GENERATE(1, 3, 42);
    std::vector expected(N, 0);
    std::iota(expected.begin(), expected.end(), 0);
    std::transform(expected.begin(), expected.end(), expected.begin(), [step, N](auto i) {
      return (i * step) >= N ? 0 : i * step;
    });
    std::vector<int> data(100, 0);

    for (auto i : range(0, N, step)) {
      data[i / step] = i;
    }

    REQUIRE_THAT(data, Catch::Equals(expected));
  }
}

TEST_CASE("Counting in a kernel", "[grid-stride] [range]")
{
  SECTION("Counting with a size-only grid-stride range matches iota")
  {
    auto N = 10000;
    std::vector<int> expected(N, 0);
    std::iota(expected.begin(), expected.end(), 0);

    thrust::device_vector<int> d_data(N, 0);

    auto block_size = GENERATE(32, 128, 160, 512);
    auto grid_size  = GENERATE(1, 10, 100);

    size_kernel<<<block_size, grid_size>>>(N, thrust::raw_pointer_cast(d_data.data()));

    std::vector<int> data(N, 0);
    thrust::copy(d_data.begin(), d_data.end(), data.begin());

    INFO("block_size: " << block_size << " grid_size: " << grid_size);
    REQUIRE_THAT(data, Catch::Equals(expected));
  }

  SECTION("Counting with a begin-end grid-stride range matches transformed iota")
  {
    auto N      = 10000;
    auto offset = GENERATE(0, 1, 3, 42);
    std::vector<int> expected(N, 0);
    std::iota(expected.begin(), expected.end(), 0);
    std::transform(
      expected.begin(), expected.end(), expected.begin(), [offset](auto i) { return i + offset; });

    thrust::device_vector<int> d_data(N, 0);

    auto block_size = GENERATE(32, 128, 160, 512);
    auto grid_size  = GENERATE(1, 10, 100);

    begin_end_kernel<<<block_size, grid_size>>>(
      offset, N + offset, thrust::raw_pointer_cast(d_data.data()));

    std::vector<int> data(N, 0);
    thrust::copy(d_data.begin(), d_data.end(), data.begin());

    INFO("block_size: " << block_size << " grid_size: " << grid_size);
    REQUIRE_THAT(data, Catch::Equals(expected));
  }

  SECTION("Counting with a stepped grid-stride range matches transformed iota")
  {
    auto N    = 10000;
    auto step = GENERATE(1, 3, 42);
    std::vector expected(N, 0);
    std::iota(expected.begin(), expected.end(), 0);
    std::transform(expected.begin(), expected.end(), expected.begin(), [step, N](auto i) {
      return (i * step) >= N ? 0 : i * step;
    });

    thrust::device_vector<int> d_data(N, 0);

    auto block_size = GENERATE(32, 128, 160, 512);
    auto grid_size  = GENERATE(1, 10, 100);

    step_kernel<<<block_size, grid_size>>>(N, step, thrust::raw_pointer_cast(d_data.data()));

    std::vector<int> data(N, 0);
    thrust::copy(d_data.begin(), d_data.end(), data.begin());

    INFO("block_size: " << block_size << " grid_size: " << grid_size << " step: " << step);
    REQUIRE_THAT(data, Catch::Equals(expected));
  }
}
