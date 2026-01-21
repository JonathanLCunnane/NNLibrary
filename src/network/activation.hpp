#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <concepts>

#include "../context/contexts.hpp"
#include "../ops/operations.hpp"
#include "../tensor/tensor.hpp"

template <ValidContext Context, int Out>
class Activation {
 public:
  explicit Activation(Context& ctx) : ctx_(ctx) {}

  virtual void forward(Tensor<Context, 1, Out>& input,
                       Tensor<Context, 1, Out>& output) = 0;

  virtual void backward(Tensor<Context, 1, Out>& grad_a_in,
                        Tensor<Context, 1, Out>& grad_z_out) = 0;

 protected:
  Context& ctx_;
};

template <ValidContext Context, int Out>
class IdentityActivation : public Activation<Context, Out> {
 public:
  explicit IdentityActivation(Context& ctx) : Activation<Context, Out>(ctx) {}

  void forward(Tensor<Context, 1, Out>& input,
               Tensor<Context, 1, Out>& output) override {
    output = input;
  }

  void backward(Tensor<Context, 1, Out>& grad_a_in,
                Tensor<Context, 1, Out>& grad_z_out) override {
    grad_z_out = grad_a_in;
  }
};

template <ValidContext Context, int Out>
class ReLUActivation : public Activation<Context, Out> {
 public:
  explicit ReLUActivation(Context& ctx)
      : Activation<Context, Out>(ctx), cached_input_(nullptr) {}

  void forward(Tensor<Context, 1, Out>& input,
               Tensor<Context, 1, Out>& output) override {
    ReLU(this->ctx_, input, output);
    cached_input_ = &input;
  }

  void backward(Tensor<Context, 1, Out>& grad_a_in,
                Tensor<Context, 1, Out>& grad_z_out) override {
    ReLUPrime(this->ctx_, *cached_input_, grad_a_in, grad_z_out);
  }

 private:
  Tensor<Context, 1, Out>* cached_input_;
};

template <ValidContext Context, int Out>
class SigmoidActivation : public Activation<Context, Out> {
 public:
  explicit SigmoidActivation(Context& ctx)
      : Activation<Context, Out>(ctx), cached_output_(nullptr) {}

  void forward(Tensor<Context, 1, Out>& input,
               Tensor<Context, 1, Out>& output) override {
    sigmoid(this->ctx_, input, output);
    cached_output_ = &output;
  }

  void backward(Tensor<Context, 1, Out>& grad_a_in,
                Tensor<Context, 1, Out>& grad_z_out) override {
    sigmoidPrime(this->ctx_, *cached_output_, grad_a_in, grad_z_out);
  }

 private:
  Tensor<Context, 1, Out>* cached_output_;
};

template <ValidContext Context, int Out>
class TanhActivation : public Activation<Context, Out> {
 public:
  explicit TanhActivation(Context& ctx) : Activation<Context, Out>(ctx) {}

  void forward(Tensor<Context, 1, Out>& input,
               Tensor<Context, 1, Out>& output) override {
    // Tanh implementation
  }

  void backward(Tensor<Context, 1, Out>& grad_a_in,
                Tensor<Context, 1, Out>& grad_z_out) override {
    // Tanh derivative implementation
  }
};

template <typename T, typename Context, int Out>
concept ValidActivation = std::derived_from<T, Activation<Context, Out>>;

#endif  // ACTIVATION_HPP
