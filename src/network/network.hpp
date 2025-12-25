#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <tuple>

#include "../context/contexts.hpp"
#include "layer.hpp"

template <typename FirstLayer, typename... RemainingLayers>
constexpr bool correct_topology() {
  if constexpr (sizeof...(RemainingLayers) == 0) {
    return true;
  } else {
    using NextLayer = std::tuple_element_t<0, std::tuple<RemainingLayers...>>;
    if constexpr (FirstLayer::kOut != NextLayer::kIn) {
      return false;
    } else {
      return correct_topology<RemainingLayers...>();
    }
  }
}

template <typename... Layers>
concept CorrectlyChainedLayers = sizeof...(Layers) > 0 &&
                                 correct_topology<Layers...>();

template <ValidContext Context, int In, int Out,
          ValidLossLayer<Context, Out> LossLayer, ValidLayer<Context>... Layers>
  requires CorrectlyChainedLayers<Layers...> &&
           (std::tuple_element_t<sizeof...(Layers) - 1,
                                 std::tuple<Layers...>>::kOut == Out) &&
           (std::tuple_element_t<0, std::tuple<Layers...>>::kIn == In)
class Network {
 public:
  Network(Context& ctx, LossLayer loss_layer, Layers... layers)
      : ctx_(ctx),
        loss_layer_(loss_layer),
        layers_(layers...),
        layer_outputs_(Tensor<Context, 1, Layers::kOut>(ctx)...),
        layer_gradients_(Tensor<Context, 1, Layers::kOut>(ctx)...) {}

  Tensor<Context, 1, Out>& forward(Tensor<Context, 1, In>& input) {
    std::get<0>(layers_).forward(input, std::get<0>(layer_outputs_));
    forward_recursive_();
    return std::get<kNumLayers - 1>(layer_outputs_);
  }

  constexpr void backward(Tensor<Context, 1, Out>& targets) {
    loss_layer_.grad(std::get<kNumLayers - 1>(layer_outputs_), targets,
                     std::get<kNumLayers - 1>(layer_gradients_));
    backward_recursive_();
  }

 private:
  constexpr static size_t kNumLayers = sizeof...(Layers);

  Context& ctx_;
  LossLayer loss_layer_;
  std::tuple<Layers...> layers_;
  std::tuple<Tensor<Context, 1, Layers::kOut>...> layer_outputs_;
  std::tuple<Tensor<Context, 1, Layers::kOut>...> layer_gradients_;

  // Recursive compile-time forward pass. We handle the first layer separately.
  template <size_t LayerNum = 1>
    requires(LayerNum < kNumLayers) && (LayerNum >= 1)
  void forward_recursive_() {
    std::get<LayerNum>(layers_).forward(std::get<LayerNum - 1>(layer_outputs_),
                                        std::get<LayerNum>(layer_outputs_));

    if constexpr (LayerNum + 1 < kNumLayers) {
      forward_recursive_<LayerNum + 1>();
    }
  }

  // Recursive compile-time backward pass. We handle the last layer separately.
  template <size_t LayerNum = kNumLayers - 1>
    requires(LayerNum < kNumLayers) && (LayerNum > 0)
  void backward_recursive_() {
    std::get<LayerNum>(layers_).backward(
        std::get<LayerNum>(layer_gradients_),
        std::get<LayerNum - 1>(layer_gradients_));
    if constexpr (LayerNum > 1) {
      backward_recursive_<LayerNum - 1>();
    } else {
      // First layer, we do not need to store the gradient w.r.t. input.
      Tensor<Context, 1, In> grad_x(ctx_);
      std::get<0>(layers_).backward(std::get<0>(layer_gradients_), grad_x);
    }
  }
};

#endif  // NETWORK_HPP
