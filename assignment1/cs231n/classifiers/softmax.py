import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  for i in range(X.shape[0]):
        f = X[i].dot(W)
        loss += -f[y[i]] + np.log(np.sum(np.exp(f)))
        dW[:,y[i]] += -X[i]
        s = np.sum(np.exp(f))
        for j in range(W.shape[1]):
            dW[:,j] += np.exp(f[j])*X[i]/s
  loss = loss / X.shape[0] + 0.5 * reg * np.sum(W * W)
  dW = dW / X.shape[0] + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  f = X.dot(W)
  loss += -np.sum(f[np.arange(X.shape[0]),y])+np.sum(np.log(np.sum(np.exp(f),axis=1)))
  s = np.sum(np.exp(f),axis = 1)
  coeff = np.exp(f) / s.reshape(X.shape[0], 1)
  coeff[range(X.shape[0]), y] -= 1
  dW = X.T.dot(coeff)
  loss = loss / X.shape[0] + 0.5 * reg * np.sum(W * W)
  dW = dW / X.shape[0] + reg * W
 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

