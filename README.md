# DCNv2
####  New DCNv2 for faster training and inference

## Performance
#### convolution: B=8， C=64， H=128， W=128， C_out=256， kernel=3x3, stride=1x1, H_out=128, W_out=128

[DCNv2](https://github.com/lbin/DCNv2): Time per kernel: 5.08029 ms, GLOPS: 38162.7

**Ours**: Time per kernel: 2.74774 ms, GLOPS: 70558.9

## The method of GFLOPS computation 
> FLOPS = Batch * Height_out * Width_out * Channels_out * (10.0 * kernel_h * kernel_w * C + 2.0 *  kernel_h * kernel_w)
>
> GFLOPS = FLOPS / (avg_time * 1e9)

## Dependence
- Pytorch: 2.1.2+cu121

- Torchvision: 0.16.2+cu121

- Python: 3.10

## Build
```
bash make.sh
```

## Acknowledgements
A part of the code is borrowed from [lbin/DCNv2](https://github.com/lbin/DCNv2).
