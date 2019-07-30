'''Use case: MNIST'''

import mnist
import torch

import kernel
import eigenpro


n_class = 10
(x_train, y_train), (x_test, y_test) = mnist.load()
x_train, y_train, x_test, y_test = x_train.astype('float32'), \
    y_train.astype('float32'), x_test.astype('float32'), y_test.astype('float32')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

kernel_fn = lambda x,y: kernel.gaussian(x, y, bandwidth=5)
model = eigenpro.FKR_EigenPro(kernel_fn, x_train, n_class, device=device)
_ = model.fit(x_train, y_train, x_test, y_test, epochs=[1, 2, 5], mem_gb=12)
