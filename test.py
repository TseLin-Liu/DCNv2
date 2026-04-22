import torch ,time
from Conv2d import _C

import torch.nn as nn

from dcn_v2 import DCN

from dcn_v2_mm import DCN as MyDCN

x = torch.randn((8,64,128,128), dtype=torch.float32).cuda()
x1 = x
dcn = DCN(64, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=1).cuda()
mydcn = MyDCN(64, 256, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=1).cuda()

mydcn.weight = dcn.weight
mydcn.bias = dcn.bias
mydcn.conv_offset_mask = dcn.conv_offset_mask
#------------------------------------------------------
# import pdb;pdb.set_trace()
y = dcn(x)
y.sum().backward()
y4 = mydcn(x1)
y4.sum().backward()

diff4 = (y - y4).abs()
import pdb;pdb.set_trace()
for i, diff in enumerate([diff4]):
    print(f"Max diff{i+1}: {diff.max().item()}")
    print(f"Mean diff{i+1}: {diff.mean().item()}")
    if diff.max().item() < 5e-4:
        print("结果在误差范围内")
    else:
        print("结果偏差很严重！！！")
