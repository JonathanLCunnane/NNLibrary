#ifndef STORAGE_HPP
#define STORAGE_HPP

#include <array>
#include <cassert>
#include <cstdlib>
#include <span>
#include <stdexcept>

#include "../context/devices.hpp"

const size_t kAlignment = 64;

template <typename T, size_t Size, DeviceType Device>
class Storage;

template <typename T, size_t Size>
class Storage<T, Size, DeviceType::CPU> {
 public:
  Storage() {
    // For CPU devices we align the memory to kAlignment bytes.
    // This allows for better performance on SIMD operations.
    data_ = (T*)std::aligned_alloc(kAlignment, Size * sizeof(T));
  }

  explicit Storage(const Storage<T, Size, DeviceType::CPU>& other) {
    data_ = (T*)std::aligned_alloc(kAlignment, Size * sizeof(T));
    std::copy(other.data_, other.data_ + Size, data_);
  }

  explicit Storage(Storage<T, Size, DeviceType::CPU>&& other) noexcept
      : data_(other.data_) {
    other.data_ = nullptr;
  }

  Storage<T, Size, DeviceType::CPU>& operator=(
      const Storage<T, Size, DeviceType::CPU>& other) {
    if (this == &other) {
      return *this;
    }
    std::copy(other.data_, other.data_ + Size, data_);
    return *this;
  }

  Storage<T, Size, DeviceType::CPU>& operator=(
      Storage<T, Size, DeviceType::CPU>&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    std::free(data_);
    data_ = other.data_;
    other.data_ = nullptr;
    return *this;
  }

  template <size_t Rows, size_t Cols>
    requires(Rows* Cols == Size)
  void set(std::array<std::array<T, Cols>, Rows>& values) {
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        data_[i * Cols + j] = values[i][j];
      }
    }
  }

  std::span<T, Size> get() const { return std::span<T, Size>(data_, Size); }

  ~Storage() { std::free(data_); }

 private:
  T* data_;
};

#endif  // STORAGE_HPP
