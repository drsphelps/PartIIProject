from classifier import Classifier
from sampling import Sampling

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, GlobalMaxPool1D, Dropout
from tensorflow.keras.models import Model
from sklearn import metrics


class RNNClassifer(Classifier):
    def __init__(self, embedding_model):
        super().__init__(embedding_model)

    def __create_embedding_matrix(self, params):
        embedding_matrix = np.random.random(
            (len(self.tokenizer.word_index) + 1, self.params['embedding_dim']))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = self.embedding_model.wv[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        self.embedding_matrix = embedding_matrix

    def __create_sequences(self, X, labels):
        self.tokenizer = Tokenizer(num_words=self.params['max_words'])
        self.tokenizer.fit_on_texts(X)
        sequences = self.tokenizer.texts_to_sequences(X)

        data = pad_sequences(sequences, padding='post',
                             maxlen=self.params['max_length'])

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        sample = Sampling(2., .5)
        x_train, y_train = sample.perform_sampling(data, labels, [0, 1])

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
                      metrics=['accuracy'])
        self.model = model

    def train(self, training_data: dict, params: dict):
        self.params = params
        nc = [d[0] for d in training_data['noncrime']]
        c = [d[0] for d in training_data['crime']]
        X = np.concatenate((nc, c), axis=0)

        # one hot vector in the form [non-crime, crime]
        labels = np.concatenate(
            (np.array([[1, 0]] * nc.shape[0]), np.array([[0, 1]] * c.shape[0])), axis=0)

        x_train, y_train = self.__create_sequences(X, labels)
        self.__create_embedding_matrix(params)

        history = self.model.fit(x_train, y_train, epochs=2,
                                 batch_size=64, validation_data=(None, None))

    def pred(self, example):
        e = np.array([example[0]])
        sequences = self.tokenizer.texts_to_sequences(e)
        sequences = pad_sequences(sequences, padding='post', maxlen=200)
        res = self.model.predict(sequences)[0]
        return res.index(max(res))

    def test(self, data):
        super().test(data)
        self.groundtruths = [0 if i == -1 else 1 for i in self.groundtruths]
        print(metrics.classification_report(
            self.groundtruths, self.predictions, digits=3))
