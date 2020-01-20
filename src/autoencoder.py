import os
import numpy as np
from gensim.models import Doc2Vec
from train_embeddings import MonitorCallback
from post_cleaning import process_text
from utils.db import db
from keras.layers import Input, Dense
from keras.models import Model


# DATA_DIR = 'data/'
# d2v_model = Doc2Vec.load('data/models/1.modelFile')


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
    L1 = Dense(encode)(input_layer)
    L2 = Dense(int(encode/2))(L1)
    L3 = Dense(int(encode/2))(L2)
    L4 = Dense(encode)(L3)
    output = Dense(dimensions)(L4)

    model = Model(inputs=input_layer, outputs=output)
    return model


# print(training.shape)
# print(test.shape)

posts = []
conn = db()

for post in conn.get_noncrime_posts(15000):
    if len(process_text(post[0])) > 10:
        posts.append(post)

print(posts[:100])
print(len(posts))
