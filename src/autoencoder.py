import os
import pickle
import numpy as np
from gensim.models import Doc2Vec
from train_embeddings import MonitorCallback
from post_cleaning import process_text
from utils.db import db
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from keras.optimizers import adam

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

    input_layer = Input(shape=(500, ))
    L1 = Dense(100, activation="tanh")(input_layer)
    L2 = Dense(50, activation="tanh")(L1)
    L3 = Dense(100, activation="tanh")(L2)
    output = Dense(500, activation="tanh")(L3)

    model = Model(inputs=input_layer, outputs=output)
    return model


# print(training.shape)
# print(test.shape)

# posts = []
conn = db()

# for post in conn.get_noncrime_posts(100):
#     v = process_text(post[0])
#     if len(v) > 10:
#         posts.append(d2v_model.infer_vector(post))

# X = np.stack(posts)
# print(X.shape)
# with open('ae.data', 'wb') as f:
#     pickle.dump(X, f)
with open('ae.data', 'rb') as f:
    X = pickle.load(f)

np.random.shuffle(X)

tests = []
for i in range(0, 500):
    with open('data/ewhore_data/' + str(i) + '.data', 'r') as f:
        tests.append(d2v_model.infer_vector(process_text(f.read())))

tests = np.stack(tests)
print(tests.shape)


print(X.shape)
X, X_test = train_test_split(X, test_size=0.1)
print(X.shape)
X_train, X_val = train_test_split(X, test_size=0.1)

X = X / np.linalg.norm(X)

optimiser = adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)

model = build_encoder()
model.compile(optimizer='adadelta',
              loss='binary_crossentropy')

model.fit(X_train, X_train,
          epochs=1,
          batch_size=int(X_train.shape[0]/10),
          shuffle=True,
          validation_data=(X_val, X_val),
          verbose=1)


super_test = []
for thread in [5881918, 1262128, 2804572, 1065115]:
    posts = conn.get_posts_from_thread(thread)
    for p in posts:
        super_test.append(d2v_model.infer_vector(process_text(p[1])))
super_test = np.stack(super_test)


correct = 0.0
total = 0.0

for i in range(0, X_test.shape[0]):
    example = X_test[i:i+1, :]
    mse = np.mean(np.power(example - model.predict(example), 2), axis=1)
    if mse < 0.0001:
        correct += 1.0
    total += 1.0

for i in range(0, tests.shape[0]):
    example = tests[i:i+1, :]
    mse = np.mean(np.power(example - model.predict(example), 2), axis=1)
    if mse > 0.0001:
        correct += 1.0
    total += 1.0

super_correct = 0.0
super_total = 0.0

for i in range(0, super_test.shape[0]):
    example = super_test[i:i+1, :]
    mse = np.mean(np.power(example - model.predict(example), 2), axis=1)
    print(mse)
    if mse < 0.0001:
        super_correct += 1.0
        correct += 1.0
    super_total += 1.0
    total += 1.0

text = "Lol. You could always sell them for Paypal and have the money transferred to your bank account or use a Paypal debit card (They're free, I think mines a mastercard). You can sell BTC for pretty much any kind of currency, Western Union etc."
v = d2v_model.infer_vector(process_text(text))
v = np.array(v)
v = np.reshape(v, (-1, 500))

print(np.mean(np.power(v - model.predict(v), 2), axis=1))


print("Super Accuracy: " + str(super_correct/super_total))
print("Accuracy: " + str(correct/total))

# 5881918
# 1262128
# 2804572
# 1065115
