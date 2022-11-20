import torch
import numpy as np
from utils import float_x
import kernel, eigenpro

def normalize(x):
    return x/np.linalg.norm(x, axis=-1)[:, None]

n, d, c = 1000, 10, 2
w_star=normalize(float_x(np.random.randn(d, c)))
x_train, x_test = normalize(float_x(np.random.randn(n, d))), normalize(float_x(np.random.randn(n, d)))
y_train, y_test = normalize(float_x(x_train @ w_star)), normalize(float_x(x_test @ w_star))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

kernel_fn = lambda x,y: kernel.ntk_relu_normalized(x, y, depth=10)
model = eigenpro.FKR_EigenPro(kernel_fn, x_train, c, device=device)
_ = model.fit(x_train, y_train, x_test, y_test, epochs=[1, 5, 15, 30], mem_gb=12)
print('Test complete')