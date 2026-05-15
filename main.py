import tensorflow as tf
import numpy as np
from train import *

def load_mnist():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train[:, np.newaxis, :, :] / 255.0
    X_test  = X_test[:,  np.newaxis, :, :] / 255.0
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_mnist()

# conduct training on a subset of the data for working without GPU
print("starting training")
train(X_train[:30000], y_train[:30000], epochs=20, batch_size=32, lr=0.02)

# evaluate on a subset of the data. 
print("Train acc:", accuracy(X_train[:5000], y_train[:5000]))
print("Test acc:", accuracy(X_test[:2000], y_test[:2000]))