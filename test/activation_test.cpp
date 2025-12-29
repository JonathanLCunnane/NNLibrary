#include "../src/network/activation.hpp"

#include <gtest/gtest.h>

#include <array>

#include "../src/context/contexts.hpp"

TEST(ActivationTest, IdentityActivationForward) {
  CPUContext ctx = CPUContext();

  IdentityActivation<CPUContext, 3> identity_act(ctx);

  Tensor<CPUContext, 1, 3> input(ctx);
  Tensor<CPUContext, 1, 3> output(ctx);

  std::array<std::array<float, 3>, 1> input_values = {{{1.0f, -2.0f, 3.0f}}};

  input.set(input_values);

  identity_act.forward(input, output);

  std::span<float, 3> output_span = output.get();
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(output_span[i], input_values[0][i]);
  }
}

TEST(ActivationTest, IdentityActivationBackward) {
  CPUContext ctx = CPUContext();

  IdentityActivation<CPUContext, 2> identity_act(ctx);

  Tensor<CPUContext, 1, 2> grad_a_in(ctx);
  Tensor<CPUContext, 1, 2> grad_z_out(ctx);

  std::array<std::array<float, 2>, 1> grad_a_in_values = {{{0.5f, -1.5f}}};

  grad_a_in.set(grad_a_in_values);

  identity_act.backward(grad_a_in, grad_z_out);

  std::span<float, 2> grad_z_out_span = grad_z_out.get();
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(grad_z_out_span[i], grad_a_in_values[0][i]);
  }
}