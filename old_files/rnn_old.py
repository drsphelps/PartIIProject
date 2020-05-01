from nltk.corpus import stopwords
from utils.db import db
from utils.post_cleaning import process_text_s as process_text
import numpy as np
import pickle
from gensim.models import Word2Vec
from sampling import Sampling

from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 200
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 100
conn = db()


def noncrime_dataset(n, save, load=False):
    def load_dataset():
        with open('nc.data', 'rb') as f:
            X = pickle.load(f)
        np.random.shuffle(X)
        return X
    if load:
        return load_dataset()
    posts = []
    for post in conn.get_noncrime_posts(n):
        if len(process_text(post[0])) > 10:
            posts.append(process_text(post[0]))
    print(posts)
    X = np.stack(posts)
    if save:
        with open('nc.data', 'wb') as f:
            pickle.dump(X, f)
    np.random.shuffle(X)
    return X


def crime_dataset():
    def load_crime_data(folder):
        tests = []
        for i in range(0, 500):
            with open('data/' + folder + '_data/' + str(i) + '.data', 'r') as f:
                tests.append(process_text(f.read()))
        return tests

    posts = []
    for f in ['rat', 'ewhore', 'stresser', 'crypter']:
        posts.extend(load_crime_data(f))
    return np.stack(posts)


nc = noncrime_dataset(3000, True, True)
c = crime_dataset()
X = np.concatenate((nc, c), axis=0)
# one hot vector in the form [non-crime, crime]
labels = np.concatenate(
    (np.array([[1, 0]] * nc.shape[0]), np.array([[0, 1]] * c.shape[0])), axis=0)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
word_index = tokenizer.word_index

data = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

num_validation_samples = int(VALIDATION_SPLIT*data.shape[0])
sample = Sampling(2., .5)
x_train, y_train = sample.perform_sampling(
    data[: -num_validation_samples], labels[: -num_validation_samples], [0, 1])
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
print('Number of entries in each category:')
print('training: ', y_train.sum(axis=0))
print('validation: ', y_val.sum(axis=0))


model = Word2Vec.load('1ft.modelFile')

embeddings_index = {}
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = model.wv[word]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(" Completed!")

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False,
                            name='embeddings')
embedded_sequences = embedding_layer(sequence_input)
x = LSTM(60, return_sequences=True, name='lstm_layer',
         dropout=0.5, recurrent_dropout=0.5)(embedded_sequences)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.5)(x)
preds = Dense(2, activation="softmax")(x)
# preds = Dense(2, activation='softmax')(preds)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Training progress:')
history = model.fit(x_train, y_train, epochs=2,
                    batch_size=64, validation_data=(x_val, y_val))

super_test = []
for thread in [5881918, 1262128, 2804572, 1065115]:
    posts = conn.get_posts_from_thread(thread)
    for p in posts:
        print(process_text(p[1]))
        super_test.append(process_text(p[1]))
super_test.append(process_text(
    "The most I've gotten out of one guy is around $450. I kept milking him (started as $30) but then he started asking to vid call me and wouldn't stop. Few days later I acted like my parents caught me and took my phone, I even acted like I'm the e-whore's father and texted the guy LMAO. He said he was just a friend from high school ***IMG***[https://hackforums.net/images/smilies/hehe.gif]***IMG*** I've already got him to buy the flight tickets in my whores name. next he's buying our accommodation in Fiji. I'm surprised he's not even indian, legit just a white American Male."))
super_test = np.stack(super_test)
a = np.array([
    process_text(
        "This isnt about crime, in fact I am just writing about rainbows and ponies, I love ponies so much and rainbows are so pretty I just want to see them everyday"),
    process_text("I'm the best in the world, I make so much money ripping people off, buy my, they will make you a lot of money, very very quickly")])
print(a)
sequences = tokenizer.texts_to_sequences(super_test)
sequences = pad_sequences(sequences, padding='post',
                          maxlen=MAX_SEQUENCE_LENGTH)
print(sequences)
print(sequences[0].shape)
print(model.predict(sequences))
