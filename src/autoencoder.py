import os
import numpy as np
from gensim.models import Doc2Vec
from train_embeddings import MonitorCallback
from post_cleaning import process_text
from utils.db import db
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

# DATA_DIR = 'data/'
d2v_model = Doc2Vec.load('data/models/1.modelFile')


# vectors = []
# for d in ['ewhore_data', 'stresser_data']:
#     directory = DATA_DIR + d
#     for filename in os.listdir(directory):
#         with open(directory+'/'+filename, 'r') as f:
#             v = process_text(f.read())
#             vectors.append(d2v_model.infer_vector(v))

# np_vs = np.stack(vectors)

# training, test = np_vs[:80, :], np_vs[80:, :]

def build_encoder():
    dimensions = 500
    encode = 250

    input_layer = Input(shape=(500, ))
    L1 = Dense(encode, activation="relu")(input_layer)
    L2 = Dense(int(encode/2), activation="tanh")(L1)
    L3 = Dense(int(encode/2), activation="tanh")(L2)
    output = Dense(dimensions, activation="relu")(L3)

    model = Model(inputs=input_layer, outputs=output)
    return model


# print(training.shape)
# print(test.shape)

posts = []
conn = db()

for post in conn.get_noncrime_posts(100):
    v = process_text(post[0])
    if len(v) > 10:
        posts.append(d2v_model.infer_vector(post))

X = np.stack(posts)
print(X.shape)
X_train, X_test = train_test_split(X, test_size=0.2)


model = build_encoder()
model.compile(optimizer="adam",
              loss='mean_squared_error',
              metrics=['accuracy'])

history = model.fit(X_train, X_train,
                    epochs=100,
                    batch_size=int(X_train.shape[0]/10),
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1).history
