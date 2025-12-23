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
  Network() {}
};

#endif  // NETWORK_HPP
