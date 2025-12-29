#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <Fastor/Fastor.h>

#include "../context/contexts.hpp"
#include "../tensor/tensor.hpp"

template <ValidContext Context, int M, int K, int N>
void matmul(Context& ctx, const Tensor<Context, M, K>& A,
            const Tensor<Context, K, N>& B, Tensor<Context, M, N>& C) {
  matmul<M, K, N>(ctx, A.get(), B.get(), C.get());
}

template <ValidContext Context, int M, int N>
void matmul(Context& ctx, float scalar, const Tensor<Context, M, N>& B,
            Tensor<Context, M, N>& C) {
  matmul<M, N>(ctx, scalar, B.get(), C.get());
}

// Fastor CPU implementation of Matrix X Matrix
template <int M, int K, int N>
void matmul(CPUContext& ctx, const std::span<float, M * K> A,
            const std::span<float, K * N> B, std::span<float, M * N> C) {
  Fastor::TensorMap<float, M, K> fA(A.data());
  Fastor::TensorMap<float, K, N> fB(B.data());
  Fastor::TensorMap<float, M, N> fC(C.data());
  fC = Fastor::matmul(fA, fB);
}

// Fastor CPU implementation of Scalar X Matrix
template <int M, int N>
void matmul(CPUContext& ctx, float scalar, const std::span<float, M * N> B,
            std::span<float, M * N> C) {
  Fastor::TensorMap<float, M, N> fB(B.data());
  Fastor::TensorMap<float, M, N> fC(C.data());
  fC = scalar * fB;
}

#endif  // MATMUL_HPP