# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import time

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from seq_encoder import rnn_encode
import char_dataset as cd

url = 'http://mattmahoney.net/dc/'
filename= 'text8.zip'
expected_bytes =  31344016



vocabulary_size = 50000

epochs = 20
epoch_size = 2000
batch_size = 100
sequence_length = 10  #sequence length to hold a word in (can we make thi
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
learning_rate = 0.01
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

def generate_batch(data, data_index, batch_size, num_skips, skip_window):
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  data_length = len(data)
  buf = collections.deque(maxlen=span)
  for _ in range(span):
    buf.append(data[data_index])
    data_index = (data_index + 1) % data_length
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buf[skip_window]
      labels[i * num_skips + j, 0] = buf[target]
    buf.append(data[data_index])
    data_index = (data_index + 1) % data_length
  return batch, labels, data_index


class Encoder():

  def __init__(s, data):
    # Input data.
    s.bs = batch_size
    s.sl = sequence_length
    s.embedding_size = embedding_size
   
    #char_data
    s.dataset = cd.Dataset(data)

    #number of characters that are found in words
    s.vocab_size = s.dataset.distinct_characters
    #inputs = ['testing', 'testing2'...]
    #inputs -> translate to [[12, 44..], [12, 44..] ..]
    #inputs -> right pad with zeros, convert to tensor [bs, sl, 1]
    #note, embedding at this point has not been done yet
    #convert to list of tensor of shape [<char embedding>, <char_embedding>, ..]


    s.input_data = tf.placeholder(tf.int32, shape=[s.bs, s.sl])
    s.train_inputs = tf.placeholder(tf.int32, shape=[s.bs])
    s.train_labels = tf.placeholder(tf.int32, shape=[s.bs, 1])
    s.valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    #embedding list
    # Ops and variables pinned to the CPU because of missing GPU implementation
    #with tf.device('/cpu:0'):
    
    #TODO- work out why I need this variable scope call
    with tf.variable_scope('seq_embedding'):
      char_embedding = tf.get_variable("char_embedding", [s.vocab_size, s.embedding_size])
      inputs = tf.split(1, s.sl, tf.nn.embedding_lookup(char_embedding, s.input_data))
      inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
      #at this point inputs = [ <tensor: bs, embedding_size>]
      
    # Look up embeddings for inputs.
    s.embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    s.embed = tf.nn.embedding_lookup(s.embeddings, s.train_inputs)
    #shape = (bs, embedding_size)
    s.rnn_embedding, state = rnn_encode(s.bs, s.embedding_size, inputs, rnn_depth=3)
    
    # Construct the variables for the NCE loss
    s.nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    s.nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    s.loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=s.nce_weights,
                       biases=s.nce_biases,
                       labels=s.train_labels,
                       inputs=s.rnn_embedding,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    s.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(s.loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    s.norm = tf.sqrt(tf.reduce_sum(tf.square(s.embeddings), 1, keep_dims=True))
    s.normalized_embeddings = s.embeddings / s.norm
    s.valid_embeddings = tf.nn.embedding_lookup(
        s.normalized_embeddings, s.valid_dataset)
    s.similarity = tf.matmul(
        s.valid_embeddings, s.normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    s.init = tf.global_variables_initializer()


def main():
  '''
  a wrapper for as many non-functional parts as I can find
  '''
 
  """Download a file if not present, and make sure it's the right size."""
  local_filename = filename
  if not os.path.exists(filename):
    local_filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(local_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', local_filename)
  else:
    print(statinfo.st_size)
    raise Exception('Failed to verify ' + local_filename + '. Can you get to it with a browser?')
 
  with zipfile.ZipFile('text8.zip', 'r') as zf:
    zf.extractall('./')
    
  f =  open('text8','r')
  words = f.read().split()
  f.close()
  print('Number of words', len(words))

  #build_dataset
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

  data_index = 0
  batch, labels, data_index = generate_batch(data, data_index, batch_size=8, num_skips=2, skip_window=1)
  
  for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

  # Step 4: Build and train a skip-gram model.
  graph = tf.Graph()
  with graph.as_default():
    enc = Encoder(' '.join(words) + ' UNK')


  with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    enc.init.run()

    print("=======================Training Start=============================")
    average_loss = 0
    for epoch in range(epochs):
      starttime = time.time()
      for step in range(epoch_size):
        batch_inputs, batch_labels, data_index= generate_batch(data, data_index,
            batch_size, num_skips, skip_window)


        #generates and pads the words
        str_batch_inputs = [enc.dataset.char_to_num(reverse_dictionary[batch_inputs[i]]) for i in range(batch_size)]

        str_batch_inputs = [( enc.dataset.char_to_num(['']) * (sequence_length - len(i)) + i)[:sequence_length] for i in str_batch_inputs]
        str_batch_inputs = np.array(str_batch_inputs)
        
        feed_dict = {enc.train_inputs: batch_inputs, enc.train_labels: batch_labels, enc.input_data:str_batch_inputs}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([enc.optimizer, enc.loss], feed_dict=feed_dict)
        average_loss += loss_val

      average_loss /= epoch_size
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("=================Average loss:", average_loss, "   epoch: ", epoch, "  duration: ", time.time() - starttime, "==============")
      average_loss = 0

      sim = enc.similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
      print('\n\n')
    final_embeddings = enc.normalized_embeddings.eval()

  # Step 6: Visualize the embeddings.

  try:

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)

  except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

def plot_with_labels(low_dim_embs, labels, img_name='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(img_name)


if __name__ == "__main__":
  main()

