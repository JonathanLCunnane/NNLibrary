#ifndef LAYER_HPP
#define LAYER_HPP

#include <concepts>

#include "activation.hpp"

template <ValidContext Context, int In, int Out,
          ValidActivation<Context, Out> Activation>
class Layer {
 public:
  static constexpr int kIn = In;
  static constexpr int kOut = Out;
  using kContext = Context;

  Layer(Context& ctx, Activation& act)
      : ctx_(ctx), act_(act), weights_(ctx), biases_(ctx) {
    // Initialize weights and biases here.
  }

  void forward(Tensor<Context, 1, In>& input, Tensor<Context, 1, Out>& output) {
    // Implement forward pass logic here
    act_.forward(input, output);
  }

  void backward(Tensor<Context, 1, Out>& grad_a_in,
                Tensor<Context, 1, In>& grad_z_out) {
    act_.backward(grad_a_in, grad_z_out);
    // Implement backward pass logic here
  }

 private:
  Context& ctx_;
  Activation act_;
  Tensor<Context, In, Out> weights_;
  Tensor<Context, 1, Out> biases_;
  Tensor<Context, 1, Out>* cached_output_;
};

template <typename T, typename Context>
concept ValidLayer = requires {
  // Dimensions must be integer.
  { T::kIn } -> std::convertible_to<int>;
  { T::kOut } -> std::convertible_to<int>;

  // Ensures all layers in a network share the same context type.
  std::same_as<typename T::kContext, Context>;
};

template <ValidContext Context, int In, int Out>
using IdentityLayer = Layer<Context, In, Out, IdentityActivation<Context, Out>>;

template <ValidContext Context, int In>
class LossLayer {
 public:
  LossLayer(Context& ctx) : ctx_(ctx) {}

  virtual float loss(Tensor<Context, 1, In>& predictions,
                     Tensor<Context, 1, In>& targets) = 0;

  virtual void grad(Tensor<Context, 1, In>& grad_yhat_out) = 0;

 protected:
  Context& ctx_;
  Tensor<Context, 1, In>* cached_predictions_;
};

// Specific layer type for Cross Entropy Loss with Softmax activation
template <ValidContext Context, int In>
class CrossEntropyLossLayer : public LossLayer<Context, In> {
 public:
  CrossEntropyLossLayer(Context& ctx) : LossLayer<Context, In>(ctx) {}

  float loss(Tensor<Context, 1, In>& predictions,
             Tensor<Context, 1, In>& targets) override {
    // Implement cross-entropy loss calculation here
    return 0.0f;
  }

  void grad(Tensor<Context, 1, In>& grad_yhat_out) override {
    // Implement gradient calculation here
  }
};

template <typename T, typename Context, int In>
concept ValidLossLayer = std::derived_from<T, LossLayer<Context, In>>;

#endif  // LAYER_HPP
