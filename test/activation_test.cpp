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

TEST(ActivationTest, ReLUActivationForward) {
  CPUContext ctx = CPUContext();

  ReLUActivation<CPUContext, 3> relu_act(ctx);

  Tensor<CPUContext, 1, 3> input(ctx);
  Tensor<CPUContext, 1, 3> output(ctx);

  std::array<std::array<float, 3>, 1> input_values = {{{1.0f, -2.0f, 3.0f}}};

  input.set(input_values);

  relu_act.forward(input, output);

  std::span<float, 3> output_span = output.get();
  EXPECT_EQ(output_span[0], 1.0f);
  EXPECT_EQ(output_span[1], 0.0f);
  EXPECT_EQ(output_span[2], 3.0f);
}

TEST(ActivationTest, ReLUActivationBackward) {
  CPUContext ctx = CPUContext();

  ReLUActivation<CPUContext, 3> relu_act(ctx);

  Tensor<CPUContext, 1, 3> input(ctx);
  Tensor<CPUContext, 1, 3> output(ctx);

  std::array<std::array<float, 3>, 1> input_values = {{1.0f, -2.0f, 3.0f}};

  input.set(input_values);

  relu_act.forward(input, output);

  Tensor<CPUContext, 1, 3> grad_a_in(ctx);
  Tensor<CPUContext, 1, 3> grad_z_out(ctx);

  std::array<std::array<float, 3>, 1> grad_a_in_values = {{1.5f, 0.5f, -1.0f}};

  grad_a_in.set(grad_a_in_values);

  relu_act.backward(grad_a_in, grad_z_out);

  std::span<float, 3> grad_z_out_span = grad_z_out.get();
  EXPECT_EQ(grad_z_out_span[0], 1.5f);
  EXPECT_EQ(grad_z_out_span[1], 0.0f);
  EXPECT_EQ(grad_z_out_span[2], -1.0f);
}

TEST(ActivationTest, SigmoidActivationForwardBackward) {
  CPUContext ctx = CPUContext();

  SigmoidActivation<CPUContext, 3> sigmoid_act(ctx);

  Tensor<CPUContext, 1, 3> input(ctx);
  Tensor<CPUContext, 1, 3> output(ctx);

  std::array<std::array<float, 3>, 1> input_values = {{-1.0f, 0.0f, 2.0f}};

  input.set(input_values);

  sigmoid_act.forward(input, output);

  std::span<float, 3> output_span = output.get();
  EXPECT_NEAR(output_span[0], 0.268941f, 1e-5f);
  EXPECT_NEAR(output_span[1], 0.5f, 1e-5f);
  EXPECT_NEAR(output_span[2], 0.880797f, 1e-5f);

  Tensor<CPUContext, 1, 3> grad_a_in(ctx);
  Tensor<CPUContext, 1, 3> grad_z_out(ctx);

  std::array<std::array<float, 3>, 1> grad_a_in_values = {{1.0f, 0.5f, 0.2f}};

  grad_a_in.set(grad_a_in_values);

  sigmoid_act.backward(grad_a_in, grad_z_out);
  std::span<float, 3> grad_z_out_span = grad_z_out.get();
  std::array<float, 3> expected_grad_z_out = {0.196612f, 0.125000f, 0.020999f};
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(grad_z_out_span[i], expected_grad_z_out[i], 1e-5f);
  }
}