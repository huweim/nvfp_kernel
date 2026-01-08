#pragma once
#include <torch/all.h>

void reciprocal_approximate_ftz_tensor(torch::Tensor const& input,
                                       torch::Tensor& output);
