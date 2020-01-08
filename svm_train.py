from utils.db import db
from post_cleaning import process_text
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import random
import subprocess

no_part = 10


def rr_split(features):
    partitions = []

    for i in range(0, no_part):
        partitions.append([])

    for i in range(0, len(features)):
        partition = i % no_part
        partitions[partition].append(features[i])

    return partitions


def write_to_file(f_name, data):
    s = '\n'.join([str(value) + " " + ' '.join([str(i+1) + ':' + str(vector[i]) for i in range(0, len(vector))]) for (value, vector) in data])
    with open(f_name, 'w') as f:
        f.write(s)


def train_svm(type1, type2):
    d2v_model = Doc2Vec.load('1.modelFile')
    training = []

    for (idPost, content) in type1:
        vector = d2v_model.infer_vector(doc_words=content)
        training.append((1, vector))

    for (idPost, content) in type2:
        vector = d2v_model.infer_vector(doc_words=content)
        training.append((-1, vector))
    
    write_to_file('training.data', training)

    subprocess.call("./svm_learn training.data model.data", shell=True)


def predict(type1, type2):
    d2v_model = Doc2Vec.load('1.modelFile')
    test = []
    results = {}

    for (idPost, content) in type1:
        vector = d2v_model.infer_vector(doc_words=content)
        test.append((1, vector))

    for (idPost, content) in type2:
        vector = d2v_model.infer_vector(doc_words=content)
        test.append((-1, vector))

    write_to_file('test.data', test)

    subprocess.call("./svm_classify test.data model.data predictions.data", shell=True)

    predicitions = []

    with open('predicitons.data',  'r') as f:
        predicitions.extend([float(p) for p in f.read().split('\n') if p != ""])


    correct = 0
    combined = type1 + type2
    
    for i in range(len(predictions)):
        predicition = "type1" if predictions[i] > 0  else "type2"
        if prediction == "type1" and i < 500:
            correct += 1
        elif prediciton == "type2" and i >= 500:
            correct += 1
        results[combined[i][0]] = (prediciton, 0 if i < 500 else 1)


    print(float(correct)/float(total))
    return results


def build_training(features, index):
    r = []
    for i in range(0, len(features)):
        if i != index:
            r.extend(features[i])
    return r


def cross_validation(f1, f2):
    type1_posts = []
    type2_posts = []

    conn = db()

    with open(f1, 'r') as f:
        for line in f:
            type1_posts.append((line, process_text(conn.get_content_from_post(line, 0))))

    with open(f2, 'r') as f:
        for line in f:
            type2_posts.append((line, process_text(conn.get_content_from_post(line, 0))))

   
    print(type1_posts)
    print(type2_posts)

    conn.close_connection()

    type1_posts = rr_split(type1_posts)
    type2_posts = rr_split(type2_posts)

    for i in range(0, no_part):
        type1_training = build_training(type1_posts, i)
        type2_training = build_training(type2_posts, i)
        train_svm(type1_training, type2_training)

        results = predict(type1_posts[i], type2_posts[i])


    
cross_validation('ewhor_training.data', 'stresser_training.data')
