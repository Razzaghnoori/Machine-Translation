import string
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from math import floor
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils.vis_utils import plot_model

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer(oov_token='UNK')
    tokenizer.fit_on_texts(lines)
    return tokenizer

# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

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

# encode and pad sequences
def encode_sequences(tokenizer, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, padding='post')
    return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

#split data to train and test
def get_training_and_testing_sets(file_list):
    split = 0.9
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing


# model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model

def main():
    # prepare english tokenizer
    # open the file as read only
    eng_lines = read_file('test')
    eng_tokenizer = create_tokenizer(eng_lines)
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(eng_lines)

    #Persian Tokenizer
    prs_lines = read_file('tt')
    prs_tokenizer = create_tokenizer(prs_lines)
    prs_vocab_size = len(prs_tokenizer.word_index) + 1
    prs_length = max_length(prs_lines)

    # prepare training and testing data
    eng_training, eng_testing = get_training_and_testing_sets(eng_lines)
    eng_training = encode_sequences(eng_tokenizer, eng_training)
    eng_testing = encode_sequences(eng_tokenizer, eng_testing)
    prs_training, prs_testing = get_training_and_testing_sets(prs_lines)
    prs_training = encode_sequences(prs_tokenizer, prs_lines)
    prs_training = encode_output(prs_training, prs_vocab_size)
    prs_testing = encode_sequences(prs_tokenizer, prs_lines)
    prs_testing = encode_output(prs_testing, prs_vocab_size)

    # define model
    model = define_model(prs_vocab_size, eng_vocab_size, prs_length, eng_length, 256)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # summarize defined model
    print(model.summary())
    
    # fit model
    filename = 'model.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(eng_training, prs_training, epochs=30, batch_size=1, validation_data=(eng_testing, prs_testing), callbacks=[checkpoint], verbose=2)

if __name__ == "__main__":
    main()