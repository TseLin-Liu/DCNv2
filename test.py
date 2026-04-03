import torch ,time
from Conv2d import _C

import torch.nn as nn

from dcn_v2 import DCN

from dcn_v2_mm import DCN as MyDCN

x = torch.ones((8,64,128,128), dtype=torch.float32).cuda()

dcn = DCN(64, 256, kernel_size=(7, 7), stride=2, padding=3, deformable_groups=1).cuda()
mydcn = MyDCN(64, 256, kernel_size=(7, 7), stride=2, padding=3, deformable_groups=1).cuda()
nn.init.constant_(dcn.weight, 1.0)
mydcn.weight = dcn.weight
mydcn.bias = dcn.bias
mydcn.conv_offset_mask = dcn.conv_offset_mask
#------------------------------------------------------
y = dcn(x)
y4 = mydcn(x)

diff4 = (y - y4).abs()
import pdb;pdb.set_trace()
for i, diff in enumerate([diff4]):
    print(f"Max diff{i+1}: {diff.max().item()}")
    print(f"Mean diff{i+1}: {diff.mean().item()}")
    if diff.max().item() < 5e-4:
        print("结果在误差范围内")
    else:
        print("结果偏差很严重！！！")
