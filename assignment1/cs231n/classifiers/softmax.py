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

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    f_i = X[i].dot(W)
    #shift f_i for numeric instability: http://cs231n.github.io/linear-classify/#softmax
    f_i -= np.max(f_i)
    softmax = lambda k: np.exp(f_i[k]) / np.sum(np.exp(f_i))
    loss += -1 * np.log(softmax(y[i]))
#    print(dW[i,:])
    for j in range(num_classes):
        dW[:,j] += (softmax(j) - (j == y[i])) * X[i]     

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg*W

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

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

  num_train = X.shape[0]
  num_classes = W.shape[1]
  f_i = X.dot(W)
  
  #shift f_i for numeric instability: http://cs231n.github.io/linear-classify/#softmax
  f_i -= np.max(f_i)
  softmax = lambda k: np.exp(f_i[np.arange(len(f_i)),k]) / np.sum(np.exp(f_i),axis=1)
  loss = np.sum(-1 * np.log(softmax(y))) / num_train
  loss += reg * np.sum(W * W)

  
  #softmax_numerator = np.exp(j)
  softmax_numerator = np.exp(f_i)
  #create N vector of exp sums to normalize
  normalization_per_N = np.sum(np.exp(f_i),axis=1)
  #tile normalization to create NxC matrix
  softmax_denominator = np.tile(normalization_per_N[:,np.newaxis],(1,num_classes))
  softmax_vector =softmax_numerator / softmax_denominator
  y_mask = np.zeros((num_train,num_classes))
  y_mask[np.arange(num_train),y] = 1
  dW = X.T.dot(softmax_vector - y_mask)

  dW /= num_train
  dW += reg*W
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

