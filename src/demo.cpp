#include <stdio.h>

#include "context/contexts.hpp"
#include "network/activation.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"

int main() {
  CPUContext ctx = CPUContext();

  IdentityActivation<CPUContext, 4> act1(ctx);
  IdentityLayer<CPUContext, 5, 4> layer1(ctx, act1);

  IdentityActivation<CPUContext, 3> act2(ctx);
  IdentityLayer<CPUContext, 4, 3> layer2(ctx, act2);

  CrossEntropyLossLayer<CPUContext, 4> loss_layer(ctx);

  Network<CPUContext, 5, 3, CrossEntropyLossLayer<CPUContext, 3>,
          IdentityLayer<CPUContext, 5, 4>, IdentityLayer<CPUContext, 4, 3> >
      network;

  printf("Hello, World!\n");
  return 0;
}
