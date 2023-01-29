import torch
from eigenpro import kernels
import eigenpro

normalize = lambda x: x/x.norm(dim=-1,keepdim=True)

n, d, c = 1000, 10, 2
w_star=torch.randn(d, c)
x_train, x_test = torch.randn(n, d), torch.randn(n, d)
y_train, y_test = x_train @ w_star, x_test @ w_star

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_MEM = torch.cuda.get_device_properties(DEVICE).total_memory//1024**3

kernel_fn = lambda x,y: kernels.laplacian(x, y, bandwidth=1.)
model = eigenpro.EigenProRegressor(kernel_fn, x_train, c, device=DEVICE)
_ = model.fit(x_train, y_train, x_test, y_test, epochs=30, print_every=5, mem_gb=GPU_MEM)
print('Laplacian test complete')
