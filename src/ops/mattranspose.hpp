#ifndef MATTRANSPOSE_HPP
#define MATTRANSPOSE_HPP

#include <Fastor/Fastor.h>

#include "../context/contexts.hpp"
#include "../tensor/tensor.hpp"

template <ValidContext Context, int M, int N>
void mattranspose(Context& ctx, const Tensor<Context, M, N>& A,
                  Tensor<Context, N, M>& At) {
  mattranspose<M, N>(ctx, A.get(), At.get());
}

// Fastor CPU implementation
template <int M, int N>
void mattranspose(CPUContext& ctx, const std::span<float, M * N> A,
                  std::span<float, N * M> At) {
  Fastor::TensorMap<float, M, N> fA(A.data());
  Fastor::TensorMap<float, N, M> fAt(At.data());
  fAt = Fastor::transpose(fA);
}

#endif  // MATTRANSPOSE_HPP