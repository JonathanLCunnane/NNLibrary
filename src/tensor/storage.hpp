#ifndef STORAGE_HPP
#define STORAGE_HPP

#include <cassert>
#include <cstdlib>
#include <stdexcept>

#include "../context/devices.hpp"

const size_t kAlignment = 64;

template <typename T>
class Storage {
 public:
  Storage(size_t size, DeviceType device_type) : device_type_(device_type) {
    if (device_type == DeviceType::CPU) {
      // For CPU devices we align the memory to kAlignment bytes.
      // This allows for better performance on SIMD operations.
      data_ = (T*)std::aligned_alloc(kAlignment, size * sizeof(T));
    } else {
      throw std::invalid_argument("Unsupported device type for Storage.");
    }
  }

  ~Storage() {
    if (device_type_ == DeviceType::CPU) {
      std::free(data_);
    }
  }

 private:
  T* data_;
  DeviceType device_type_;
};

#endif  // STORAGE_HPP
