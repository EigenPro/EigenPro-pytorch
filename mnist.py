import torch, os, numpy as np
from torchvision.datasets import MNIST
from torch.nn.functional import one_hot


def unit_range_normalize(samples):
    min_vals = np.min(samples, axis=0)
    max_vals = np.max(samples, axis=0)
    diff = max_vals - min_vals
    diff[diff <= 0.0] = np.maximum(1.0, min_vals[diff <= 0.0])
    normalized = (samples - min_vals) / diff
    return normalized

def load_data():
    train_data = MNIST(os.environ['DATA_DIR'], train=True)
    test_data = MNIST(os.environ['DATA_DIR'], train=False)
    n_class, img_rows, img_cols = 10, 28, 28
    return (
        n_class, img_rows, img_cols,
        (train_data.data, train_data.targets), 
        (test_data.data, test_data.targets),
    )


def load():
    # the data, shuffled and split between train and test sets
    n_class, img_rows, img_cols, (x_train, y_train), (x_test, y_test) = load_data()
    
    x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols).float().numpy()
    x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols).float().numpy()
    # x_train = x_train.astype('float32')/255
    # x_test = x_test.astype('float32')/255
    
    x_train = unit_range_normalize(x_train)
    x_test = unit_range_normalize(x_test)
    y_train = one_hot(y_train, n_class).numpy()
    y_test = one_hot(y_test, n_class).numpy()
    print("Load MNIST dataset.")
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    (a,b), (c,d) = load()