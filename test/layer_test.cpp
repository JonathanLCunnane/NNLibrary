#include "../src/network/layer.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>

#include "../src/context/contexts.hpp"
#include "../src/network/activation.hpp"
#include "../src/network/uniform_distribution.hpp"

class MockUniformDistribution : public UniformDistribution<float> {
 public:
  MOCK_METHOD(float, Call, ());

  float operator()() override { return Call(); }
};

TEST(LayerTest, LayerInitialization) {
  CPUContext ctx = CPUContext();
  MockUniformDistribution dist;
  EXPECT_CALL(dist, Call()).Times(2 * 3).WillRepeatedly(testing::Return(0.5f));

  IdentityActivation<CPUContext, 3> identity_act(ctx);
  IdentityLayer<CPUContext, 2, 3> layer(ctx, identity_act, dist);

  std::span<float, 2 * 3> weights_span = layer.get_weights();
  for (size_t i = 0; i < 2 * 3; ++i) {
    EXPECT_EQ(weights_span[i], 0.5f);
  }

  std::span<float, 1 * 3> biases_span = layer.get_biases();
  for (size_t i = 0; i < 1 * 3; ++i) {
    EXPECT_EQ(biases_span[i], 0.0f);
  }
}

TEST(LayerTest, IdentityLayerForward) {
  CPUContext ctx = CPUContext();
  MockUniformDistribution dist;
  EXPECT_CALL(dist, Call()).Times(2 * 3).WillRepeatedly(testing::Return(0.5f));

  IdentityActivation<CPUContext, 3> identity_act(ctx);
  IdentityLayer<CPUContext, 2, 3> layer(ctx, identity_act, dist);

  Tensor<CPUContext, 1, 2> input(ctx);
  Tensor<CPUContext, 1, 3> output(ctx);
  Tensor<CPUContext, 1, 3> expected_output(ctx);

  std::array<std::array<float, 2>, 1> input_values = {{1.0f, -2.0f}};
  input.set(input_values);

  layer.forward(input, output);

  std::span<float, 3> output_span = output.get();
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(output_span[i], -0.5f);
  }
}

TEST(LayerTest, IdentityLayerBackward) {
  CPUContext ctx = CPUContext();
  MockUniformDistribution dist;
  EXPECT_CALL(dist, Call()).Times(2 * 3).WillRepeatedly(testing::Return(0.5f));

  IdentityActivation<CPUContext, 3> identity_act(ctx);
  IdentityLayer<CPUContext, 2, 3> layer(ctx, identity_act, dist);

  Tensor<CPUContext, 1, 2> input(ctx);
  Tensor<CPUContext, 1, 3> output(ctx);
  Tensor<CPUContext, 1, 3> expected_output(ctx);
  Tensor<CPUContext, 1, 3> grad_a_in(ctx);
  Tensor<CPUContext, 1, 2> grad_x_out(ctx);

  std::array<std::array<float, 2>, 1> input_values = {{1.0f, -2.0f}};
  input.set(input_values);

  layer.forward(input, output);

  std::array<std::array<float, 3>, 1> grad_a_in_values = {{1.0f, -0.5f, 0.5f}};
  grad_a_in.set(grad_a_in_values);

  layer.backward(grad_a_in, grad_x_out);
  layer.update_parameters(0.5f);

  std::span<float, 2> grad_x_out_span = grad_x_out.get();
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(grad_x_out_span[i], 0.5f);
  }

  std::span<float, 2 * 3> weights_span = layer.get_weights();
  std::array<float, 2 * 3> expected_updated_weights = {0.0f, 0.75f, 0.25f,
                                                       1.5f, 0.0f,  1.0f};
  for (size_t i = 0; i < 2 * 3; ++i) {
    EXPECT_EQ(weights_span[i], expected_updated_weights[i]);
  }

  std::span<float, 1 * 3> biases_span = layer.get_biases();
  std::array<float, 1 * 3> expected_updated_biases = {-0.5f, 0.25f, -0.25f};
  for (size_t i = 0; i < 1 * 3; ++i) {
    EXPECT_EQ(biases_span[i], expected_updated_biases[i]);
  }
}