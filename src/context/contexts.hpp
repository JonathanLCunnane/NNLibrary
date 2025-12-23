#ifndef CONTEXTS_HPP
#define CONTEXTS_HPP

#include <concepts>

#include "devices.hpp"

struct Context {
  DeviceType device_type;
  Context(DeviceType device_type) : device_type(device_type) {}
};

struct CPUContext : public Context {
  CPUContext() : Context(DeviceType::CPU) {}
};

template <typename T>
concept ValidContext = std::derived_from<T, Context>;

#endif  // CONTEXTS_HPP
