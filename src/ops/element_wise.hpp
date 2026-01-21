#ifndef ELEMENT_WISE_HPP
#define ELEMENT_WISE_HPP

#include <Fastor/Fastor.h>

#include "../context/contexts.hpp"
#include "../tensor/tensor.hpp"

template <ValidContext Context, int M, int N>
void ReLU(Context& ctx, const Tensor<Context, M, N>& input,
          Tensor<Context, M, N>& output) {
  ReLU<M, N>(ctx, input.get(), output.get());
}

// Fastor CPU implementation
template <int M, int N>
void ReLU(CPUContext& ctx, const std::span<float, M * N> input,
          std::span<float, M * N> output) {
  Fastor::TensorMap<float, M, N> fInput(input.data());
  Fastor::TensorMap<float, M, N> fOutput(output.data());
  fOutput = Fastor::max(fInput, 0.0f);
}

template <ValidContext Context, int M, int N>
void ReLUPrime(Context& ctx, const Tensor<Context, M, N>& input,
               const Tensor<Context, M, N>& grad_a_in,
               Tensor<Context, M, N>& grad_z_out) {
  ReLUPrime<M, N>(ctx, input.get(), grad_a_in.get(), grad_z_out.get());
}

// CPU implementation
template <int M, int N>
void ReLUPrime(CPUContext& ctx, const std::span<float, M * N> input,
               const std::span<float, M * N> grad_a_in,
               std::span<float, M * N> grad_z_out) {
  for (size_t i = 0; i < M * N; ++i) {
    grad_z_out[i] = input[i] > 0.0f ? grad_a_in[i] : 0.0f;
  }  // Could not find working Fastor Implementation?
}

template <ValidContext Context, int M, int N>
void sigmoid(Context& ctx, const Tensor<Context, M, N>& input,
             Tensor<Context, M, N>& output) {
  sigmoid<M, N>(ctx, input.get(), output.get());
}

// Fastor CPU implementation
template <int M, int N>
void sigmoid(CPUContext& ctx, const std::span<float, M * N> input,
             std::span<float, M * N> output) {
  Fastor::TensorMap<float, M, N> fInput(input.data());
  Fastor::TensorMap<float, M, N> fOutput(output.data());
  fOutput = 1.0f / (1.0f + Fastor::exp(-fInput));
}

template <ValidContext Context, int M, int N>
void sigmoidPrime(Context& ctx, const Tensor<Context, M, N>& output,
                  const Tensor<Context, M, N>& grad_a_in,
                  Tensor<Context, M, N>& grad_z_out) {
  sigmoidPrime<M, N>(ctx, output.get(), grad_a_in.get(), grad_z_out.get());
}

// Fastor CPU implementation
template <int M, int N>
void sigmoidPrime(CPUContext& ctx, const std::span<float, M * N> output,
                  const std::span<float, M * N> grad_a_in,
                  std::span<float, M * N> grad_z_out) {
  Fastor::TensorMap<float, M, N> fOutput(output.data());
  Fastor::TensorMap<float, M, N> fGradAIn(grad_a_in.data());
  Fastor::TensorMap<float, M, N> fGradZOut(grad_z_out.data());
  fGradZOut = fGradAIn * fOutput * (1.0f - fOutput);
}

#endif  // ELEMENT_WISE_HPP