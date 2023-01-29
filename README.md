# EigenPro-pytorch

EigenPro (short for Eigenspace Projections) is a fast iterative solver for Kernel Regression.
Original paper [Kernel machines that adapt to GPUs for effective large batch training](https://arxiv.org/abs/1806.06144)

It has a O(n) space and time complexity with respect to number of samples. 
The algorithm is based on preconditioned SGD and has autotuned hyperparameters to maximize GPU utilization. 
Currently this code has been tested with n=1,000,000 samples.

EigenPro iteration in PyTorch

# Installation
```
pip install git+https://github.com/EigenPro/EigenPro-pytorch.git@pytorch
```

# Test with Laplacian kernel
```python
import torch
from eigenpro import kernels
import eigenpro

n = 1000 # number of samples
d = 100  # dimensions
c = 3    # number of targets

w_star=torch.randn(d, c)
x_train, x_test = torch.randn(n, d), torch.randn(n, d)
y_train, y_test = x_train @ w_star, x_test @ w_star

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_MEM = torch.cuda.get_device_properties(DEVICE).total_memory//1024**3

kernel_fn = lambda x,y: kernels.laplacian(x, y, bandwidth=1.)
model = eigenpro.EigenProRegressor(kernel_fn, x_train, c, device=DEVICE)
_ = model.fit(x_train, y_train, x_test, y_test, epochs=30, print_every=5, mem_gb=GPU_MEM)
print('Laplacian test complete')
```

# Bibtex
```latex
@article{ma2019kernel,
  title={Kernel machines that adapt to GPUs for effective large batch training},
  author={Ma, Siyuan and Belkin, Mikhail},
  journal={Proceedings of Machine Learning and Systems},
  volume={1},
  pages={360--373},
  year={2019}
}
```
