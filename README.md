# EigenPro2-pytorch

EigenPro (short for Eigenspace Projections) is a fast iterative solver for Kernel Regression.  
**Paper:** [Kernel machines that adapt to GPUs for effective large batch training](https://arxiv.org/abs/1806.06144), SysML (2019).  
**Authors:**  Siyuan Ma and Mikhail Belkin. (Bibtex below)

It has a $O(n)$ memory and $O(n^2)$ time complexity with respect to number of samples. \
The algorithm is based on preconditioned SGD and has autotuned hyperparameters to maximize GPU utilization. 

# Installation
```
pip install git+https://github.com/EigenPro/EigenPro-pytorch.git
```
Requires a PyTorch installation

## Stable behavior
Currently this code has been tested with n=1,000,000 samples.\
with Python 3.9 and `PyTorch >= 1.13`


# Test installation with Laplacian kernel
```python
import torch
from eigenpro2.kernels import laplacian
from eigenpro2.models import KernelModel

n = 1000 # number of samples
d = 100  # dimensions
c = 3    # number of targets

w_star=torch.randn(d, c)
x_train, x_test = torch.randn(n, d), torch.randn(n, d)
y_train, y_test = x_train @ w_star, x_test @ w_star

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM = torch.cuda.get_device_properties(DEVICE).total_memory//1024**3 - 1 # GPU memory in GB, keeping aside 1GB for safety
else:
    DEVICE = torch.device("cpu")
    DEV_MEM = 8 # RAM available for computing

kernel_fn = lambda x, y: laplacian(x, y, bandwidth=1.)
model = KernelModel(kernel_fn, x_train, c, device=DEVICE)
result = model.fit(x_train, y_train, x_test, y_test, epochs=30, print_every=5, mem_gb=DEV_MEM)
print('Laplacian test complete!')
```

### Bibtex
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
