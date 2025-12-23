#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "../context/contexts.hpp"
#include "storage.hpp"

template <ValidContext Context, int Rows, int Cols>
class Tensor {
 public:
  explicit Tensor(Context& ctx)
      : ctx_(ctx), data_(Rows * Cols, ctx.device_type) {}

 private:
  Storage<float> data_;
  Context& ctx_;
};

#endif  // TENSOR_HPP
