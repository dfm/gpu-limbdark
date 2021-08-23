#ifndef _JAX_LIMBDARK_KERNELS_H_
#define _JAX_LIMBDARK_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace jax_limbdark {
struct KeplerDescriptor {
  std::int64_t size;
};

void gpu_kepler_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_kepler_f64(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

}  // namespace jax_limbdark

#endif