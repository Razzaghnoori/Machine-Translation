
import re
import string
import numpy  as np
import argparse

from os.path import exists
from pickle import load
from numpy import array
from math import floor
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils.vis_utils import plot_model
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--train', '-t', help='Train if set.', action='store_true')
parser.add_argument('--predict', '-p', help='Predict if set.', action='store_true')
parser.add_argument('--input', '-i', help="Source language file.")
parser.add_argument('--output', '-o', help="Target language file.")
parser.add_argument('--model-path', '-m', \
    help='Path to save the model and load it back again.', default='model.h5')
parser.add_argument('--epochs', '-n', help='Number of epochs', \
    default=100, type=int)
parser.add_argument('--embeddings-dim', '-d', help='Dimension of the embeddings', \
    default=128, type=int)
parser.add_argument('--batch-size', '-b', default=16, type=int)

arguments = parser.parse_args()

# fit a tokenizer
def _create_tokenizer(filename):
    with open(filename) as f:
        tokenizer = Tokenizer(oov_token='UNK')
        tokenizer.fit_on_texts(f)
        return tokenizer

#reading file and split it into sentences
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
def encode_sequences(filename='', text='', max_len=30, to_ohe=False, tokenizer=None):
    if tokenizer is None:
        tokenizer = _create_tokenizer(filename)
    
    if filename:
        with open(filename) as f:
            X = tokenizer.texts_to_sequences(f)
    elif text:
        X = tokenizer.texts_to_sequences(text)

    X = pad_sequences(X, maxlen=max_len, padding='post')
    
    if not to_ohe:
        return X, tokenizer
    return to_categorical(X), tokenizer

def define_model(X, y, tar_vocab_size, n_units):
    model = Sequential()
    model.add(Embedding(K.max(X)+1, n_units, input_length=X.shape[1]))
    model.add(LSTM(n_units))
    model.add(RepeatVector(y.shape[1]))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab_size, activation='softmax')))
    return model
    
def train(model, X, y, model_path):
    # fit model
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', \
        verbose=1, save_best_only=True, mode='min')
    model.fit(X, y, epochs=arguments.epochs, batch_size=arguments.batch_size, \
        validation_split=0.1, callbacks=[checkpoint], verbose=2)

    model.save(model_path)
    return model

def predict(model_path, eng_tokenizer, fa_tokenizer, sents=None, model=None):
    if model is None:
        model = load_model(model_path)
    
    if sents is None:
        sents = [input('> ')]
    x, _ = encode_sequences(text=sents, tokenizer=eng_tokenizer)

    pred = model.predict(x)[0]
    inds = pred.argmax(axis=-1)
    print(' '.join([fa_tokenizer.index_word[x+1] \
        for x in inds.tolist()]).replace(' UNK', ''))


if __name__ == "__main__":
    model = None
    
    X, en_tokenizer = encode_sequences(arguments.input)
    y, fa_tokenizer = encode_sequences(arguments.output, to_ohe=True)

    if exists(arguments.model_path):
        model = load_model(arguments.model_path)
    else:
        fa_vocab_size = y.shape[-1]

        perm = np.random.permutation(X.shape[0])
        X, y = X[perm], y[perm]

        # prepare training and testing data
        #X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

        # define model
        model = define_model(X, y, fa_vocab_size, arguments.embeddings_dim)
        model.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy')
        
        # summarize defined model
        print(model.summary())

    if arguments.train:
        model = train(model, X, y, arguments.model_path)

    if arguments.predict:
        while True:
            predict(arguments.model_path, en_tokenizer, fa_tokenizer, model=model)