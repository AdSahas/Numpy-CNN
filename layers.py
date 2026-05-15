import numpy as np

'''
Layers implemented:
- ConvLayer
- ReLU
- MaxPool
- Flatten
- FullyConnected
- SoftmaxCrossEntropy
'''

class ConvLayer:
  def __init__(self, nFilters, kernelSize, inChannels):
    self.nFilters = nFilters
    self.kernelSize = kernelSize
    self.inChannels = inChannels

    self.W = np.random.randn(nFilters, inChannels, kernelSize, kernelSize) * 0.1
    self.b = np.zeros(nFilters)

  def forward(self, x):

    # shape: batch, inChannels, height, width
    batch, inChannels, height, width = x.shape

    # size of each kernel (they're all square)
    K = self.kernelSize
    F = self.nFilters # how many different kernels we have

    # define output dimensions
    outH = height - K + 1
    outW = width - K + 1

    # output: one feature map per filter
    out = np.zeros((batch, F, outH, outW))

    # take a "patch" for the portion to convolve over.
    for i in range(outH):
      for j in range(outW):
        patch = x[:, :, i:i+K, j:j+K]

        for f in range(F):
          # dot product
          out[:, f, i, j] = np.sum(patch * self.W[f], axis=(1,2,3)) + self.b[f]

    self.x = x
    return out

  def backward(self, grad):

    # set up gradients of weights and biases.
    self.db = np.sum(grad, axis=(0, 2, 3))
    self.dW = np.zeros_like(self.W)
    dx = np.zeros_like(self.x)

    # shape: batch, inChannels, height, width
    N, F, outH, outW = grad.shape
    K = self.kernelSize

    # for every patch calculate gradient.
    for i in range(outH):
      for j in range(outW):
        patch = self.x[:, :, i:i+K, j:j+K]

        # for every channel, the derivative dL/dW = sum over batch of (dL/dout * patch)
        for f in range(F):
          self.dW[f] += np.sum(
                patch * grad[:, f:f+1, i:i+1, j:j+1],
                axis=0
            )
          # for every channel, the derivative dL/dx = sum over filters of (dL/dout * W)
          dx[:, :, i:i+K, j:j+K] += self.W[f] * grad[:, f:f+1, i:i+1, j:j+1]
    return dx

  def update(self, lr):
    self.W -= lr * self.dW
    self.b -= lr * self.db

class ReLU:
  def __init__(self):
    pass
  def forward(self, x):
    self.x = np.maximum(0, x)
    return self.x
  def backward(self, grad):
    return grad * (self.x > 0)
  
class MaxPool:
  def __init__(self, poolSize):
    self.poolSize = poolSize

  def forward(self, x):
    batch, inChannels, height, width = x.shape
    outH = height//self.poolSize
    outW = width//self.poolSize

    out = np.zeros((batch, inChannels, outH, outW))

    # as in conv, we take a path and find the max value. 
    for i in range(outH):
      for j in range(outW):

        patch = x[:, :, i*self.poolSize:(i+1)*self.poolSize, j*self.poolSize:(j+1)*self.poolSize]
        out[:, :, i, j] = np.max(patch, axis=(2, 3))


    self.x = x
    return out

  def backward(self, grad):

    # the backward is assigns the gradient to the max in a patch.
    batch, inChannels, height, width = self.x.shape
    P = self.poolSize

    dx = np.zeros_like(self.x)

    # for each patch, find the max value and assign gradient to that. 
    for i in range(height//P):
      for j in range(width//P):

        # find the patch with the max value
        patch = self.x[:, :, i*P:(i+1)*P, j*P:(j+1)*P]

        # create a mask to apply to the patch and apply it. 
        maxes = np.max(patch, axis=(2, 3), keepdims = True)
        mask = (patch == maxes)
        dx[:, :, i*P:(i+1)*P, j*P:(j+1)*P] = (mask*grad[:, :, i:i+1, j:j+1])

    return dx
  
class Flatten:
  def __init__(self):
    pass

    # essentially, reshaping the input for the NN
  def forward(self, x):
    self.shape = x.shape
    return x.reshape(x.shape[0], -1)

  def backward(self, grad):

    # backward just transforms the gradient back to the original shape.
    return grad.reshape(self.shape)
  
class FullyConnected:
  def __init__(self, inSize, outSize):

    # set random weights, because we dont want to start with a model that outputs the same thing for every input.
    self.W = np.random.randn(inSize, outSize) * 0.1
    self.b = np.zeros(outSize)

  def forward(self, x):

    # save input for backward pass
    self.x = x
    out = x @ self.W + self.b
    return out

  def backward(self, grad):

    # dw = x^T @ grad, db = sum over batch of grad. dx = grad @ W^T
    self.dw = self.x.T @ grad
    self.db = np.sum(grad, axis=0)

    # this dx will be used for backward passes, so they can update their own params.
    dx = grad @ self.W.T
    return dx

  def update(self, lr):
    self.W -= lr * self.dw
    self.b -= lr * self.db

class SoftmaxCrossEntropy:

    # softmax and cross-entropy are combined here, because they are often used together
    # and this allows for a more efficient backward pass.
    def forward(self, logits, labels):
      self.labels = labels
      N = logits.shape[0]

      # shifted is for numerical stability, to prevent large exponentials.
      # it shifts the logits so the largest exponential will be e^0 = 1.
      shifted = logits - np.max(logits, axis=1, keepdims=True)

      # exp scores is the unnormalized probabilities.
      # probs is used to normalize them.
      exp_scores = np.exp(shifted)
      self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

      #calculate the loss as the negative log of the probability of the correct class.
      correct_probs = self.probs[np.arange(N), labels]
      loss = -np.log(correct_probs + 1e-12)

      return np.mean(loss)

    def backward(self):
      N = self.probs.shape[0]

      grad = self.probs.copy()
      grad[np.arange(N), self.labels] -= 1
      grad /= N

      return grad
    
