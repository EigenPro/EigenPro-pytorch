import torch
import datasets
import eigenpro2
from eigenpro2.kernels import laplacian, ntk_relu_unit_sphere as relu_ntk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_class, (x_train, y_train), (x_test, y_test) = datasets.load('cifar10', DEVICE, split='digits')

x_train=x_train/x_train.norm(dim=-1,keepdim=True)
x_test=x_test/x_test.norm(dim=-1,keepdim=True)

kernel_fn = lambda x, y: laplacian(x, y, bandwidth=1.)
#kernel_fn = lambda x, z: relu_ntk(x, z, depth=3)

model = eigenpro2.KernelModel(kernel_fn, x_train, n_class, device=DEVICE)

results = model.fit(x_train, y_train, x_test, y_test, epochs=20, print_every=2, mem_gb=20)
