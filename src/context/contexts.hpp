#ifndef CONTEXTS_HPP
#define CONTEXTS_HPP

#include <concepts>

#include "devices.hpp"

template <DeviceType Device>
class Context {
 public:
  static constexpr DeviceType kDevice = Device;
};

struct CPUContext : public Context<DeviceType::CPU> {};

template <typename T>
concept ValidContext = std::same_as<T, CPUContext>;

#endif  // CONTEXTS_HPP
