import tensorflow as tf
from IPython.display import SVG
from nltk.corpus import stopwords
from utils.db import db
from utils.get_training import noncrime_dataset, crime_dataset
from utils.post_cleaning import process_text_s as process_text
from sampling import Sampling
import numpy as np
import pickle
from gensim.models import Word2Vec

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, GlobalMaxPool1D
from tensorflow.keras.models import Model

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 200
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 100


class RNN:

    def __init__(self, lr=0.025, e=3, d=0.5, bs=64):
        self.learning_rate = lr
        self.epochs = e
        self.dropout = d
        self.batch_size = bs

        self.collect_data()
        self.build_embeddings()
        self.build_model()

    def collect_data(self):
        nc = noncrime_dataset(3000, True, True)
        c = crime_dataset()
        X = np.concatenate((nc, c), axis=0)

        # one hot vector in the form [non-crime, crime]
        labels = np.concatenate(
            (np.array([[1, 0]] * nc.shape[0]), np.array([[0, 1]] * c.shape[0])), axis=0)

        # Fit a tokenizer for integer representation of words
        self.tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        self.tokenizer.fit_on_texts(X)
        sequences = self.tokenizer.texts_to_sequences(X)

        data = pad_sequences(sequences, padding='post',
                             maxlen=MAX_SEQUENCE_LENGTH)

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        num_validation_samples = int(VALIDATION_SPLIT*data.shape[0])

        # Perform over and under sampling
        sample = Sampling(2., .5)
        self.x_train, self.y_train = sample.perform_sampling(
            data[: -num_validation_samples], labels[: -num_validation_samples], [0, 1])
        self.x_val = data[-num_validation_samples:]
        self.y_val = labels[-num_validation_samples:]

    def build_embeddings(self):
        model = Word2Vec.load('1ft.modelFile')

        embedding_matrix = np.random.random(
            (len(self.tokenizer.word_index) + 1, EMBEDDING_DIM))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = model.wv[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        self.embedding_matrix = embedding_matrix

    def build_model(self):
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedding_layer = Embedding(len(self.tokenizer.word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False,
                                    name='embeddings')
        embedded_sequences = embedding_layer(sequence_input)
        x = LSTM(60, return_sequences=True, name='lstm_layer',
                 dropout=self.dropout, recurrent_dropout=self.dropout)(embedded_sequences)
        x = GlobalMaxPool1D()(x)
        x = Dense(50, activation="tanh")(x)
        x = Dropout(self.dropout)(x)
        preds = Dense(2, activation="softmax")(x)

        self.__compile_model(sequence_input, preds)

    def __compile_model(self, sequence_input, preds):
        model = Model(sequence_input, preds)

        optimiser = adam(learning_rate=self.learning_rate, beta_1=0.9,
                         beta_2=0.999, amsgrad=False)

        model.compile(loss='binary_crossentropy',
                      optimizer=optimiser,
                      metrics=['accuracy'])

        self.model = model

    def training_loop(self):
        print('Training progress:')
        history = self.model.fit(self.x_train, self.y_train, epochs=self.epochs,
                                 batch_size=self.batch_size, validation_data=(self.x_val, self.y_val))

    def plot_model(self):
        tf.keras.utils.plot_model(
            self.model, to_file='model.png', show_shapes=True, show_layer_names=True,
            rankdir='LR'
        )

    def classify_post(self, text):
        text_array = np.array([text])

        sequences = self.tokenizer.texts_to_sequences(text_array)
        sequences = pad_sequences(sequences, padding='post',
                                  maxlen=MAX_SEQUENCE_LENGTH)
        return self.model.predict(sequences)[0]


rnn = RNN()
rnn.training_loop()
