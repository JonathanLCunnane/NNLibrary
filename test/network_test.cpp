#include "../src/network/network.hpp"

#include <gtest/gtest.h>

#include "../src/context/contexts.hpp"
#include "../src/network/activation.hpp"
#include "../src/network/layer.hpp"

TEST(NetworkTest, MinimalNetworkCompiles) {
  CPUContext ctx = CPUContext();

  IdentityActivation<CPUContext, 1> act1(ctx);
  IdentityLayer<CPUContext, 1, 1> layer1(ctx, act1);

  CrossEntropyLossLayer<CPUContext, 1> loss_layer(ctx);

  Network<CPUContext, 1, 1, CrossEntropyLossLayer<CPUContext, 1>,
          IdentityLayer<CPUContext, 1, 1> >
      network(ctx, loss_layer, layer1);
}

TEST(NetworkTest, LargerNetworkCompiles) {
  CPUContext ctx = CPUContext();

  IdentityActivation<CPUContext, 4> act1(ctx);
  IdentityLayer<CPUContext, 5, 4> layer1(ctx, act1);

  IdentityActivation<CPUContext, 3> act2(ctx);
  IdentityLayer<CPUContext, 4, 3> layer2(ctx, act2);

  CrossEntropyLossLayer<CPUContext, 3> loss_layer(ctx);

  Network<CPUContext, 5, 3, CrossEntropyLossLayer<CPUContext, 3>,
          IdentityLayer<CPUContext, 5, 4>, IdentityLayer<CPUContext, 4, 3> >
      network(ctx, loss_layer, layer1, layer2);
}
