import keras
import numpy as np

from keras.datasets.mnist import load_data


def unit_range_normalize(samples):
	min_vals = np.min(samples, axis=0)
	max_vals = np.max(samples, axis=0)
	diff = max_vals - min_vals
	diff[diff <= 0.0] = np.maximum(1.0, min_vals[diff <= 0.0])
	normalized = (samples - min_vals) / diff
	return normalized


def load():
    # input image dimensions
    n_class = 10
    img_rows, img_cols = 28, 28
    
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_data()
    
    x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    x_train = unit_range_normalize(x_train)
    x_test = unit_range_normalize(x_test)
    y_train = keras.utils.to_categorical(y_train, n_class)
    y_test = keras.utils.to_categorical(y_test, n_class)
    print("Load MNIST dataset.")
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    return (x_train, y_train), (x_test, y_test)
