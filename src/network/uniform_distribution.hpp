#ifndef UNIFORM_DISTRIBUTION_HPP
#define UNIFORM_DISTRIBUTION_HPP

#include <random>
#include <type_traits>

template <typename T>
  requires std::is_arithmetic_v<T>
class UniformDistribution {
 public:
  virtual T operator()() = 0;
};

class StdFloatDistribution : public UniformDistribution<float> {
 public:
  StdFloatDistribution(float lb, float ub) : dist_(lb, ub) {}

  float operator()() override { return dist_(gen); }

 private:
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_real_distribution<float> dist_;
};

#endif