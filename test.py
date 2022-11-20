'''Use case: MNIST'''

import mnist
import torch
import numpy as np
from utils import float_x

import kernel
import eigenpro

n, d, c = 1000, 10, 2
w_star=float_x(np.random.randn(d, c))
x_train, x_test = float_x(np.random.randn(n, d)), float_x(np.random.randn(n, d))
y_train, y_test = float_x(x_train @ w_star), float_x(x_test @ w_star)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

kernel_fn = lambda x,y: kernel.ntk_relu(x, y, depth=5)
model = eigenpro.FKR_EigenPro(kernel_fn, x_train, c, device=device)
_ = model.fit(x_train, y_train, x_test, y_test, epochs=[1, 2, 30], mem_gb=12)
print('Test complete')