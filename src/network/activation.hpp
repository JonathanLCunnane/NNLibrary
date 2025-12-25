#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <concepts>

#include "../context/contexts.hpp"
#include "../tensor/tensor.hpp"

template <ValidContext Context, int Out>
class Activation {
 public:
  explicit Activation(Context& ctx) : ctx_(ctx), cached_input_(nullptr) {}

  virtual void forward(Tensor<Context, 1, Out>& input,
                       Tensor<Context, 1, Out>& output) = 0;

  virtual void backward(Tensor<Context, 1, Out>& grad_a_in,
                        Tensor<Context, 1, Out>& grad_z_out) = 0;

 protected:
  Context& ctx_;
  Tensor<Context, 1, Out>* cached_input_;
};

template <ValidContext Context, int Out>
class IdentityActivation : public Activation<Context, Out> {
 public:
  explicit IdentityActivation(Context& ctx) : Activation<Context, Out>(ctx) {}

  void forward(Tensor<Context, 1, Out>& input,
               Tensor<Context, 1, Out>& output) override {
    // Identity forward pass: output = input
  }

  void backward(Tensor<Context, 1, Out>& grad_a_in,
                Tensor<Context, 1, Out>& grad_z_out) override {
    // Identity backward pass: grad_z_out = grad_a_in
  }
};

template <typename T, typename Context, int Out>
concept ValidActivation = std::derived_from<T, Activation<Context, Out>>;

#endif  // ACTIVATION_HPP
