import numpy as np
from layers import *

# initialize layers
conv = ConvLayer(nFilters=8, kernelSize=3, inChannels=1)
relu = ReLU()
pool = MaxPool(poolSize=2)
flatten = Flatten()
fc = FullyConnected(inSize=8 * 13 * 13, outSize=10)
loss_fn = SoftmaxCrossEntropy()


def train_step(x_batch, y_batch, lr):

    # run a forward pass
    out = conv.forward(x_batch)
    out = relu.forward(out)
    out = pool.forward(out)
    out = flatten.forward(out)
    logits = fc.forward(out)
    loss = loss_fn.forward(logits, y_batch)

    # run a backward pass
    grad = loss_fn.backward()
    grad = fc.backward(grad)
    grad = flatten.backward(grad)
    grad = pool.backward(grad)
    grad = relu.backward(grad)
    grad = conv.backward(grad)

    # update parameters
    fc.update(lr)
    conv.update(lr)

    return loss


def train(X_train, y_train, epochs=20, batch_size=32, lr=0.02):

    # take the sample count of the training data
    N = X_train.shape[0]

    for epoch in range(epochs):

        # shuffle data
        indices = np.random.permutation(N)
        X_train = X_train[indices]
        y_train = y_train[indices]

        # keep track of total loss and batches for averaging
        total_loss = 0
        num_batches = 0

        # for each batch in the training data, train the model.
        for start in range(0, N, batch_size):

            # select the batch
            end = start + batch_size
            x_batch = X_train[start:end]
            y_batch = y_train[start:end]

            # train using a forward and backward pass
            loss = train_step(x_batch, y_batch, lr)

            # accumulate the total loss
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch: {epoch + 1}  Loss: {avg_loss}")


def predict(x):
    out = conv.forward(x)
    out = relu.forward(out)
    out = pool.forward(out)
    out = flatten.forward(out)
    logits = fc.forward(out)

    # here, we dont use softmax since the order of logits remain the same.
    # we need it for training because of the cross-entropy loss. 
    return np.argmax(logits, axis=1)


def accuracy(x, y):

    # calculate accuracy
    preds = predict(x)
    return np.mean(preds == y)