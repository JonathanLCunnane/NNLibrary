#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <array>
#include <span>

#include "../context/contexts.hpp"
#include "storage.hpp"

template <ValidContext Context, int Rows, int Cols>
class Tensor {
 public:
  explicit Tensor(Context& ctx) : ctx_(ctx) {}
  explicit Tensor(const Tensor<Context, Rows, Cols>& other)
      : data_(other.data_), ctx_(other.ctx_) {}

  Tensor<Context, Rows, Cols>& operator=(
      const Tensor<Context, Rows, Cols>& other) {
    if (this == &other) {
      return *this;
    }
    data_ = other.data_;
    return *this;
  }

  void set(std::array<std::array<float, Cols>, Rows>& values) {
    data_.set(values);
  }

  std::span<float, Rows * Cols> get() const { return data_.get(); }

 private:
  Storage<float, Rows * Cols, Context::kDevice> data_;
  Context& ctx_;
};

#endif  // TENSOR_HPP
