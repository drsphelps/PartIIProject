import os
import pickle
import numpy as np
from gensim.models import Doc2Vec
from utils.MonitorCallback import MonitorCallback
from utils.post_cleaning import process_text
from utils.db import db
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from keras.optimizers import adam
from sklearn import svm
from datetime import datetime
import logging

d2v_model = Doc2Vec.load('data/models/2.modelFile')
conn = db()


def build_encoder():
    input_layer = Input(shape=(100, ))
    L0 = Dense(32, activation="sigmoid")(input_layer)
    output = Dense(100, activation="sigmoid")(L0)

    model = Model(inputs=input_layer, outputs=output)
    return model


def train_encoder(optimiser, X_train, X_val):
    model = build_encoder()
    model.compile(optimizer=optimiser,
                  loss='mse',
                  metrics=['mean_squared_error'])

    model.fit(X_train, X_train,
              epochs=10,
              batch_size=64,
              shuffle=True,
              validation_data=(X_val, X_val),
              verbose=1)
    return model


def noncrime_dataset(n, save):
    posts = []
    for post in conn.get_noncrime_posts(10000):
        v = process_text(post[0])
        if len(v) > 10:
            posts.append(d2v_model.infer_vector(post))
    X = np.stack(posts)
    if save:
        with open('ae.data', 'wb') as f:
            pickle.dump(X, f)
    np.random.shuffle(X)
    return X


def load_dataset():
    with open('ae.data', 'rb') as f:
        X = pickle.load(f)
    np.random.shuffle(X)
    X = X / np.linalg.norm(X)
    return X


def load_crime_data():
    tests = []
    for folder in ['crypter', 'stresser', 'rat', 'ewhore']:
        for i in range(0, 500):
            with open('data/' + folder + '_data/' + str(i) + '.data', 'r') as f:
                tests.append(d2v_model.infer_vector(process_text(f.read())))

    tests = np.stack(tests)
    tests = tests / np.linalg.norm(tests)
    return tests


def test_dataset(comparison, predict):
    def func(dataset, limit):
        correct = 0.0
        total = 0.0

        for i in range(0, dataset.shape[0]):
            example = dataset[i:i+1, :]
            mse = np.mean(
                np.power(example - predict(example), 2), axis=1)
            print(mse)

            if comparison(mse, limit):
                correct += 1.0
            total += 1.0

        return correct, total
    return func


if __name__ == '__main__':
    X = None
    with open('ae.data', 'rb') as f:
        X = pickle.load(f)
        np.random.shuffle(X)

    # X = noncrime_dataset(15000, True)

    X = X / np.linalg.norm(X)
    # X, throwaway = train_test_split(X, 0.9)
    X, X_test = train_test_split(X, test_size=0.1)
    X_train, X_val = train_test_split(X, test_size=0.1)

    optimiser = adam(learning_rate=0.0001, beta_1=0.9,
                     beta_2=0.999, amsgrad=False)

    model = train_encoder(optimiser, X_train, X_val)

    import tensorflow as tf
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_layer_names=True,
        rankdir='TB'
    )

    tests = load_crime_data()

    # model = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    # model.fit(X_train)

    # y_pred_train = model.predict(X_train)
    # y_pred_test = model.predict(X_test)
    # y_pred_outliers = model.predict(X_val)
    # y_tests = model.predict(tests)
    # n_error_train = y_pred_train[y_pred_train == -1].size
    # n_error_test = y_pred_test[y_pred_test == -1].size
    # n_error_outliers = y_pred_outliers[y_pred_outliers == -1].size

    # print(X_train.shape)
    # print(y_pred_train[y_pred_train == 1].shape)

    # print(X_test.shape)
    # print(y_pred_test[y_pred_test == 1].shape)

    # print(tests.shape)
    # print(y_tests[y_tests == -1].shape)

    super_test = []
    for thread in [5881918, 1262128, 2804572, 1065115]:
        posts = conn.get_posts_from_thread(thread)
        for p in posts:
            print(process_text(p[1]))
            super_test.append(d2v_model.infer_vector(process_text(p[1])))
    super_test = np.stack(super_test)
    super_test = super_test / np.linalg.norm(super_test)

    greater_comparison = test_dataset(lambda a, b: a > b, model.predict)
    lesser_comparison = test_dataset(lambda a, b: a < b, model.predict)

    nc_correct, nc_total = lesser_comparison(X_test, 0.0001)
    print("Non-crime data accuracy: " + str(nc_correct/nc_total))
    c_correct, c_total = greater_comparison(tests, 0.0001)
    print("Crime data accuracy: " + str(c_correct/c_total))
    s_correct, s_total = lesser_comparison(super_test, 0.0001)
    print("Non-crime from other sources accuracy: " + str(s_correct/s_total))

    print("Super Accuracy: " + str(s_correct/s_total))
    print("Accuracy: " + str((nc_correct+c_correct)/(nc_total+c_total)))
