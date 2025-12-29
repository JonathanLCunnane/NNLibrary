#ifndef MATADD_HPP
#define MATADD_HPP

#include <Fastor/Fastor.h>

#include "../context/contexts.hpp"
#include "../tensor/tensor.hpp"

template <ValidContext Context, int M, int N>
void matadd(Context& ctx, const Tensor<Context, M, N>& A,
            const Tensor<Context, M, N>& B, Tensor<Context, M, N>& C,
            bool subtracting_b = false) {
  matadd<M, N>(ctx, A.get(), B.get(), C.get(), subtracting_b);
}

// Fastor CPU implementation
template <int M, int N>
void matadd(CPUContext& ctx, const std::span<float, M * N> A,
            const std::span<float, M * N> B, std::span<float, M * N> C,
            bool subtracting_b = false) {
  Fastor::TensorMap<float, M, N> fA(A.data());
  Fastor::TensorMap<float, M, N> fB(B.data());
  Fastor::TensorMap<float, M, N> fC(C.data());
  if (subtracting_b) {
    fC = fA - fB;
  } else {
    fC = fA + fB;
  }
}

#endif  // MATADD_HPP