# CNN From Scratch with NumPy

A convolutional neural network implemented from scratch using NumPy for image classification.
The project includes manual implementations of convolution, max pooling, dense layers, forward/backward propagation, and parameter updates without using deep learning frameworks such as TensorFlow or PyTorch.

---

## Features

- Conv2D layer implementation
- ReLU activation
- Max pooling
- Fully connected dense layer
- Softmax cross-entropy loss
- Backpropagation and gradient descent
- Mini-batch training pipeline
- Prediction and evaluation utilities

---

## Architecture

The network architecture is intentionally minimal:

Input → Conv2D → ReLU → MaxPool → Flatten → Dense → Softmax

Configuration:
- 1 convolutional layer
- 8 filters
- 3×3 kernels
- 1 max pooling layer
- 1 fully connected output layer

---

## Technologies

- Python
- NumPy

No high-level machine learning frameworks were used.

---

## Training

Training uses:
- Mini-batch gradient descent
- Softmax cross-entropy loss
- Manual backward propagation
- Parameter updates implemented from scratch

Example training loop:

train(X_train, y_train, epochs=20, batch_size=32, lr=0.02)

---

## Project Structure

```text id="t55p2m"
## Project Structure

layers.py          # Layer implementations
main.py            # Entry point
train.py           # Training pipeline
requirements.txt   # Dependencies
README.md          # Project documentation

---

## Example Usage

```python
# conduct training on a subset of the data for working without GPU
print("starting training")
train(X_train[:30000], y_train[:30000], epochs=20, batch_size=32, lr=0.02)
```

```python
# evaluate on a subset of the data. 
print("Train acc:", accuracy(X_train[:5000], y_train[:5000]))
print("Test acc:", accuracy(X_test[:2000], y_test[:2000]))
```

---

## Purpose

This project was developed to better understand the internal mechanics of convolutional neural networks, including:
- tensor operations,
- convolution,
- pooling,
- backpropagation,
- gradient computation,
- and optimization workflows.

The focus of the project is educational implementation rather than production-scale deep learning performance.

## Results

The model achieved approximately 95.95% test accuracy on the test dataset using a simple CNN architecture consisting of:
- 1 convolutional layer
- ReLU activation
- max pooling
- 1 fully connected output layer

Despite the minimal architecture and the absence of deep learning frameworks such as TensorFlow or PyTorch, the implementation successfully learned meaningful image representations through manual forward and backward propagation.

This model did not use a random seed, results my vary. 