"""
Minimal character-level RNN model. Based on work from Andrej Karpathy.  
Adapted by Tadashi Kamitaki 
"""
import numpy as np
import sys
import signal 

class Dataset():
  '''
  Methods used to read from files
  Translates characters into a vector forms
  Abstracts to a standard for the model
  '''
  #TODO-  Instantiation of this on the SAME dataset should be guarenteed to yield the same dictionaries
  def __init__(self, data):

    self.data = data
    #counter class to find distinct?

    #this is used as a buffer character (eos, etc)
    chars = set(self.data)
    chars.add('') 
    chars = list(chars)

    self.data_length = len(self.data)
    self.distinct_characters = len(chars)
    
    
    template = np.zeros((self.distinct_characters, 1))

    #used later to translate strings from characters to numbers
    self._char_to_num_dict = { ch:i for i, ch in enumerate(chars)}


    #used to translate from numbers to characters alter
    self._num_to_char_dict = { i:ch for i, ch in enumerate(chars)}

    self.place = 0 #pointer to character positionin file


  def to_char(self, vec_array):
    '''converts vectors from the model to human-readable strings from the dataset'''
    #np.argmax returns the index of the largest value in an numpy array
    return [self._num_to_char_dict[np.argmax(vec)] for vec in vec_array] 


  def to_vec(self, char_array):
    '''converts human readable text into numerics for the model'''
    #maps dictionary to numeric values
    num_array =  [self._char_to_num_dict[char] for char in char_array]

    vec_array = []

    #make needed vector and append it to vec_array
    for n in num_array:
      vec = np.zeros((self.distinct_characters, 1))
      vec[n] = 1.0

      vec_array.append(vec)

    return vec_array

        
  def char_to_num(self, char_array):
    '''convets a array of characters into numeric keys'''
    return [self._char_to_num_dict[i] for i in char_array]
  
  def num_to_char(self, num_array):
    '''convets a array of numeric keys into characters'''
    return [self._num_to_char_dict[i] for i in num_array]
      
  def readn(self, n):
    '''returns n characters from the dataset, updates the pointer'''    
    
    n = int(n)
    
    #prevent running over the length
    if self.data_length < (self.place + n): self.place = 0 
  
    start = self.place
    self.place = self.place + n
    return list(self.data[start:self.place])

  def reset(s):
    s.place = 0
    return

  def readn_vec(self, n):
    '''
    this reads from the dataset but converts it into a vector first
    '''
    return self.to_vec(list(self.readn(n)))




