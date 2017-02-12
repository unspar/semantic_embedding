'''
This is an RNN autoencoder.

It learns an encoding of a string into a feature space
by encoding and decoding it repeatedly.

'''

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope


def rnn_encode( batch_size, seq_length, rnn_size,inputs,  rnn_depth=2, state = None, scope=None):
  '''
  This is a learned fold. 
  It is defined over a vector space of size rnn_size.
  It is batch processed in chunks of seq_length
  It is trained in batches of batch_size

  Args:
    batch_size - for training, the number batches to run at once
    seq_length - the number of sequences
    rnn_size - size of the rnn that will be used to fold the sequence
    inputs - list of inputs of dim (rnn_size, )
    rnn_depth - the number of layers in the network
    state - the initial state of the rnn
    scope - the tensorflow scope provided to this chunk

  Returns:
    a tuple (state, encoded)
    where state is a tf tensor of the model state
    and encoded is a tf tensor of encoded output with a shape of
    (batch size, 1, rnn_size)
  '''
  


  #TODO, check the shape of the inputs tensor
  with variable_scope.variable_scope(scope or "rnn_encoder"):
    
    
    cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    enc_cell =  rnn_cell.MultiRNNCell([cell] * rnn_depth, state_is_tuple = True)
    #TODO add in checking to make sure state is compatble here
    initial_state = enc_cell.zero_state(batch_size, tf.float32) if state == None else state

    encoded, state= rnn.rnn(enc_cell, inputs, initial_state=initial_state)
    
    #LSTM is -1 to 1, this transformation fixes that
    scale_w = tf.Variable(tf.random_normal([rnn_size,  rnn_size]), dtype=tf.float32)
    scale_b = tf.Variable(tf.random_normal([1,  rnn_size]), dtype=tf.float32) 

    output = tf.add(tf.matmul(encoded[-1], scale_w), scale_b)

    return output, state



