from tensorflow import matmul, math, cast, float32
from tensorflow.keras.layers import Layer
from keras.backend import softmax

class DotProductAttention(Layer):
  def __init__(self, **kwargs):
    super(DotProductAttention, self).__init__(**kwargs)

  def call(self, queries, keys, values, d_k, mask=None):
    # Dot product operation between the queries and keys
    scores = matmul(queries, keys, transpose_b=True) / sqrt(d_k)
    if (mask is not None):
      # Set the 1 values to large negative numbers 
      scores += -1e9 * mask
    # Pass attention scores through a softmax function to
    # generate the attention weights
    weights = softmax(scores)

    return matmul(weights, values)