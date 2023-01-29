import torch
from utils import float_x
import kernel, eigenpro

normalize = lambda x: x/x.norm(dim=-1,keepdim=True)

n, d, c = 1000, 10, 2
w_star=float_x(torch.randn(d, c))
x_train, x_test = float_x(torch.randn(n, d)), float_x(torch.randn(n, d))
y_train, y_test = float_x(x_train @ w_star), float_x(x_test @ w_star)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_MEM = torch.cuda.get_device_properties(DEVICE).total_memory//1024**3

kernel_fn = lambda x,y: kernel.laplacian(x, y, bandwidth=1.)
model = eigenpro.FKR_EigenPro(kernel_fn, x_train, c, device=DEVICE)
_ = model.fit(x_train, y_train, x_test, y_test, epochs=[1, 5, 15, 30], mem_gb=GPU_MEM)
print('Laplacian test complete')

kernel_fn = lambda x,y: kernel.ntk_relu_unit_sphere(x, y, depth=2)
model = eigenpro.FKR_EigenPro(kernel_fn, normalize(x_train), c, device=DEVICE)
_ = model.fit(normalize(x_train), y_train, normalize(x_test), y_test, epochs=[1, 5, 15, 30], mem_gb=GPU_MEM)
print('normalized NTK test complete')

kernel_fn = lambda x,y: kernel.ntk_relu(x, y, depth=2)
model = eigenpro.FKR_EigenPro(kernel_fn, x_train, c, device=DEVICE)
_ = model.fit(x_train, y_train, x_test, y_test, epochs=[1, 5, 15, 30], mem_gb=GPU_MEM)
print('NTK test complete')
