from classifier import Classifier
from sampling import Sampling
from utils.metrics_m import recall_m, precision_m, f1_m

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM
from tensorflow.keras.layers import GlobalMaxPool1D, Dropout
from tensorflow.keras.models import Model
from sklearn import metrics
from matplotlib import pyplot as plt
from datetime import datetime
import csv


class RNNClassifer(Classifier):
    def __init__(self, embedding_model):
        super().__init__(embedding_model)
        self.filename = "data/experiment/rnn/rnn.csv"

    def __create_embedding_matrix(self):
        embedding_matrix = np.random.random(
            (len(self.tokenizer.word_index) + 1, self.params['embedding_dim']))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = self.embedding_model.wv[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        self.embedding_matrix = embedding_matrix

    def __fit_tokenizer(self, X):
        texts = []
        for x in X:
            texts.extend([d[0] for d in x])

        self.tokenizer = Tokenizer(num_words=self.params['max_words'])
        self.tokenizer.fit_on_texts(texts)

    def __create_sequences(self, X, labels, sampling=True):
        sequences = self.tokenizer.texts_to_sequences(X)

        data = pad_sequences(sequences, padding='post',
                             maxlen=self.params['max_length'])

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        if sampling:
            sample = Sampling(2., .5)
            x_train, y_train = sample.perform_sampling(data, labels, [0, 1])
        else:
            x_train, y_train = data, labels

        return x_train, y_train

    def __create_model(self):
        sequence_input = Input(
            shape=(self.params['max_length'],), dtype='int32')

        embedding_layer = Embedding(len(self.tokenizer.word_index) + 1,
                                    self.params['embedding_dim'],
                                    weights=[self.embedding_matrix],
                                    input_length=self.params['max_length'],
                                    trainable=False,
                                    name='embeddings')
        embedded_sequences = embedding_layer(sequence_input)
        x = LSTM(60, return_sequences=True, name='lstm_layer',
                 dropout=0.5, recurrent_dropout=0.5)(embedded_sequences)
        x = GlobalMaxPool1D()(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(0.5)(x)
        preds = Dense(2, activation="softmax")(x)

        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', f1_m, precision_m, recall_m],
                      )
        self.model = model

    def train(self, training_data: dict, params: dict):
        self.params = params
        self.__fit_tokenizer(training_data['all'])

        nc = [d[0] for d in training_data['noncrime']]
        c = [d[0] for d in training_data['crime']]
        X = np.concatenate((nc, c), axis=0)
        # one hot vector in the form [non-crime, crime]
        labels = np.concatenate(
            (np.array([[1, 0]] * len(nc)), np.array([[0, 1]] * len(c))), axis=0)
        x_train, y_train = self.__create_sequences(X, labels)

        nc = [d[0] for d in training_data['noncrime_val']]
        c = [d[0] for d in training_data['crime_val']]
        X = np.concatenate((nc, c), axis=0)

        # one hot vector in the form [non-crime, crime]
        labels = np.concatenate(
            (np.array([[1, 0]] * len(nc)), np.array([[0, 1]] * len(c))), axis=0)
        x_val, y_val = self.__create_sequences(X, labels, False)

        self.__create_embedding_matrix()
        self.__create_model()

        history = self.model.fit(x_train, y_train, epochs=self.params["epochs"],
                                 batch_size=32, validation_data=(x_val, y_val))

        self.save_epochs(history)

    def pred(self, example):
        e = np.array([example])
        sequences = self.tokenizer.texts_to_sequences(e)
        sequences = pad_sequences(sequences, padding='post', maxlen=200)
        a = self.model.predict(sequences)
        res = a[0]
        return np.argmax(res)

    def test(self, data):
        super().test(data)
        print(self.groundtruths)
        self.groundtruths = [
            0 if int(i) == -1 else 1 for i in self.groundtruths]
        self.present_results()

    def save_epochs(self, history):
        filename = datetime.now().strftime('data/experiment/rnn/%d%m%Y%H%M.csv')
        with open(filename, "w") as f:
            writer = csv.writer(
                f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(self.params["epochs"]):
                writer.writerow([str(history.history["val_accuracy"][i]),
                                 str(history.history["val_loss"][i]),
                                 str(history.history["val_precision_m"][i]),
                                 str(history.history["val_recall_m"][i]),
                                 str(history.history["val_f1_m"][i])])
