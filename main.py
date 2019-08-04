#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import string
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


# In[ ]:


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer(oov_token='UNK')
    tokenizer.fit_on_texts(lines)
    return tokenizer


# In[ ]:


#reading file and split it into sentences
import re
def read_file(filename):
    with open(filename) as file:
        text = file.read()
    #delete punctuation
    text = re.sub(r'[^\w\s]','',text)
    #delete numbers
    result = ''.join([i for i in text if not i.isdigit()])
    result.lower()
    return result.strip().split('\n')


# In[ ]:


# encode and pad sequences
def encode_sequences(tokenizer, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, padding='post')
    return X


# In[ ]:


# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


# In[ ]:


#split data to train and test
from math import floor
def get_training_and_testing_sets(file_list):
    split = 0.9
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing


# In[ ]:


# prepare english tokenizer
# open the file as read only
eng_lines = read_file('test')
print(eng_lines)
eng_tokenizer = create_tokenizer(eng_lines)
eng_vocab_size = len(eng_tokenizer.word_index) + 1
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))


# In[ ]:


#Persian Tokenizer
prs_lines = read_file('tt')
print(prs_lines)
prs_tokenizer = create_tokenizer(prs_lines)
prs_vocab_size = len(prs_tokenizer.word_index) + 1


# In[ ]:


# prepare training and testing data
eng_training, eng_testing = get_training_and_testing_sets(eng_lines)
eng_training = encode_sequences(eng_tokenizer, eng_training)
eng_testing = encode_sequences(eng_tokenizer, eng_testing)
prs_training, prs_testing = get_training_and_testing_sets(prs_lines)
prs_training = encode_sequences(prs_tokenizer, prs_lines)
Prs_training = encode_output(prs_training, prs_vocab_size)
prs_testing = encode_sequences(prs_tokenizer, prs_lines)
prs_testing = encode_output(prs_testing, prs_vocab_size)
print(prs_testing)
print(eng_training)


# In[ ]:


# model

