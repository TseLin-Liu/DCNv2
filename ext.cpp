/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "name.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dcn_v2_forward", &MyDCN_forward);
  m.def("dcn_v2_backward", &MyDCN_backward);
}