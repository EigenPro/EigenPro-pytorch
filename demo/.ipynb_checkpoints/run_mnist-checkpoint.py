import torch
import mnist
import kernel
import eigenpro

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_class, (x_train, y_train), (x_test, y_test) = mnist.load('emnist', DEVICE, split='balanced')

kernel_fn = lambda x,y: kernel.laplacian(x, y, bandwidth=1.)

model = eigenpro.FKR_EigenPro(kernel_fn, x_train, n_class, device=DEVICE)

results = model.fit(x_train, y_train, x_test, y_test, epochs=20, print_every=2, mem_gb=20)
