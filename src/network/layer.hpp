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
      : ctx_(ctx),
        act_(act),
        weights_(ctx),
        biases_(ctx),
        linear_output_(ctx),
        grad_z(ctx),
        cached_input_(nullptr) {
    // Initialize weights and biases here.
  }

  void forward(Tensor<Context, 1, In>& input, Tensor<Context, 1, Out>& output) {
    // Implement forward pass logic here
    cached_input_ = &input;
    act_.forward(linear_output_, output);
  }

  void backward(Tensor<Context, 1, Out>& grad_a_in,
                Tensor<Context, 1, In>& grad_x_out) {
    act_.backward(grad_a_in, grad_z);
    // Implement backward pass logic here
  }

 private:
  Context& ctx_;
  Activation act_;
  Tensor<Context, In, Out> weights_;
  Tensor<Context, 1, Out> biases_;
  Tensor<Context, 1, Out> linear_output_;
  Tensor<Context, 1, Out> grad_z;
  Tensor<Context, 1, In>* cached_input_;
};

template <typename T, typename Context>
concept ValidLayer = requires {
  // Dimensions must be integer.
  { T::kIn } -> std::convertible_to<int>;
  { T::kOut } -> std::convertible_to<int>;

  requires T::kIn > 0;
  requires T::kOut > 0;

  // Ensures all layers in a network share the same context type.
  std::same_as<typename T::kContext, Context>;  // TODO: Make more strict.
};

template <ValidContext Context, int In, int Out>
using IdentityLayer = Layer<Context, In, Out, IdentityActivation<Context, Out>>;

template <ValidContext Context, int In>
class LossLayer {
 public:
  LossLayer(Context& ctx) : ctx_(ctx) {}

  virtual float loss(Tensor<Context, 1, In>& predictions,
                     Tensor<Context, 1, In>& targets) = 0;

  virtual void grad(Tensor<Context, 1, In>& predictions,
                    Tensor<Context, 1, In>& targets,
                    Tensor<Context, 1, In>& grad_out) = 0;

 protected:
  Context& ctx_;
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

  void grad(Tensor<Context, 1, In>& predictions,
            Tensor<Context, 1, In>& targets,
            Tensor<Context, 1, In>& grad_out) override {
    // Implement gradient calculation here
  }
};

template <typename T, typename Context, int In>
concept ValidLossLayer = std::derived_from<T, LossLayer<Context, In>>;

#endif  // LAYER_HPP
