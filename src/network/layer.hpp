#ifndef LAYER_HPP
#define LAYER_HPP

#include <concepts>

#include "../ops/operations.hpp"
#include "activation.hpp"
#include "uniform_distribution.hpp"

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
        cached_grad_z_(ctx),
        cached_weights_grad_(ctx),
        cached_biases_grad_(ctx) {
    initialise_weights_();
    initialise_biases_();
  }

  Layer(Context& ctx, Activation& act,
        UniformDistribution<float>& weight_init_dist)
      : ctx_(ctx),
        act_(act),
        weights_(ctx),
        biases_(ctx),
        linear_output_(ctx),
        cached_grad_z_(ctx),
        cached_weights_grad_(ctx),
        cached_biases_grad_(ctx) {
    initialise_weights_from_(weight_init_dist);
    initialise_biases_();
  }

  void forward(Tensor<Context, 1, In>& input, Tensor<Context, 1, Out>& output) {
    cached_input_ = &input;
    matmul(ctx_, input, weights_, linear_output_);
    matadd(ctx_, linear_output_, biases_, linear_output_);
    act_.forward(linear_output_, output);
  }

  void backward(Tensor<Context, 1, Out>& grad_a_in,
                Tensor<Context, 1, In>& grad_x_out) {
    // Loss w.r.t weights
    act_.backward(grad_a_in, cached_grad_z_);
    Tensor<Context, In, 1> input_T(ctx_);
    mattranspose(ctx_, *cached_input_, input_T);
    matmul(ctx_, input_T, cached_grad_z_, cached_weights_grad_);

    // Loss w.r.t biases
    Tensor<Context, 1, 1> ones(ctx_);
    std::array<std::array<float, 1>, 1> ones_values = {{1.0f}};
    ones.set(ones_values);
    matmul(ctx_, ones, cached_grad_z_, cached_biases_grad_);

    // Loss w.r.t inputs
    Tensor<Context, Out, In> weights_T(ctx_);
    mattranspose(ctx_, weights_, weights_T);
    matmul(ctx_, cached_grad_z_, weights_T, grad_x_out);
  }

  void update_parameters(float learning_rate) {
    // Update weights
    Tensor<Context, In, Out> scaled_weights_grad(ctx_);
    matmul(ctx_, learning_rate, cached_weights_grad_, scaled_weights_grad);
    matadd(ctx_, weights_, scaled_weights_grad, weights_, true);

    // Update biases
    Tensor<Context, 1, Out> scaled_biases_grad(ctx_);
    matmul(ctx_, learning_rate, cached_biases_grad_, scaled_biases_grad);
    matadd(ctx_, biases_, scaled_biases_grad, biases_, true);
  }

  std::span<float, In * Out> get_weights() { return weights_.get(); }

  std::span<float, 1 * Out> get_biases() { return biases_.get(); }

 private:
  Context& ctx_;
  Activation act_;
  Tensor<Context, In, Out> weights_;
  Tensor<Context, 1, Out> biases_;
  Tensor<Context, 1, Out> linear_output_;
  Tensor<Context, 1, Out> cached_grad_z_;
  Tensor<Context, In, Out> cached_weights_grad_;
  Tensor<Context, 1, Out> cached_biases_grad_;
  Tensor<Context, 1, In>* cached_input_;

  void initialise_weights_from_(UniformDistribution<float>& dist) {
    std::array<std::array<float, Out>, In> temp_weights;
    for (size_t i = 0; i < In; ++i) {
      for (size_t j = 0; j < Out; ++j) {
        temp_weights[i][j] = dist();
      }
    }
    weights_.set(temp_weights);
  }

  void initialise_weights_() {
    // We use Xavier Glorot initialization for weights
    float limit = std::sqrt(6.0f / (In + Out));
    StdFloatDistribution dist(-limit, limit);
    initialise_weights_from_(dist);
  }

  void initialise_biases_() {
    std::array<std::array<float, Out>, 1> temp_biases;
    for (size_t i = 0; i < Out; ++i) {
      temp_biases[0][i] = 0.0f;
    }
    biases_.set(temp_biases);
  }
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
