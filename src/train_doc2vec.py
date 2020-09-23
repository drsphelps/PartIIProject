import psycopg2
from utils.db import db
from utils.post_cleaning import process_text
from utils.MonitorCallback import MonitorCallback
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from clustering import ConstrainedKMeans
from utils.get_training import get_db_records
from datetime import datetime
import pickle
import logging
import os
import time

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    filename=datetime.now().strftime('logs/%d%m%Y%H%M.log'),
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')


def preprocess_records():
    records = get_db_records()
    processed = []

    for i in range(len(records)):
        r = process_text(records[i])
        processed.append(TaggedDocument(r, [i]))

    return processed


def build_model(vs=120, d=0, h=0, w=11):
    monitor = MonitorCallback('1')
    return Doc2Vec(seed=0, dm=d, vector_size=vs,
                   min_count=50, epochs=10, workers=8,
                   hs=h, window=w, callbacks=[monitor])


def create_doc2vec_model():
    with open("/mnt/d/training_data.data", "rb") as f:
        r = pickle.load(f)

    logging.info("Data collected")

    model = build_model()
    model.build_vocab(r)

    logging.info("Vocabulary Built")
    model.train(r, total_examples=model.corpus_count, epochs=model.epochs)

    model.save('word2vec.modelFile')
    model.save('data/models/word2vec.modelFile')
    logging.info("Completed")

    model = Doc2Vec.load('1.modelFile')


def train_doc2vec():
    # r = preprocess_records()
    with open("/mnt/d/training_data.data", "rb") as f:
        r = pickle.load(f)

    logging.info("Loaded training data")
    results = {}

    model = build_model()
    model.build_vocab(r)
    logging.info("Vocabulary Built")
    model.train(r, total_examples=model.corpus_count, epochs=model.epochs)
    model.delete_temporary_training_data(
        keep_doctags_vectors=True, keep_inference=True)
    for i in range(0, 10):
        results[i] = ConstrainedKMeans.d2v_train(model)
        print(results)

    return results


if __name__ == "__main__":
    print(create_doc2vec_model())
