import psycopg2
from utils.db import db
from post_cleaning import process_text
from get_threads import get_from_keyword
from utils.MonitorCallback import MonitorCallback
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils.get_training import get_db_records
from datetime import datetime
import logging
import os
import time

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    filename=datetime.now().strftime('logs/%H%M%d%m%Y.log'),
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')


def preprocess_records():
    records = get_db_records()
    processed = []

    for i in range(len(records)):
        r = process_text(records[i])
        processed.append(TaggedDocument(r, [i]))

    return processed


def build_model():
    monitor = MonitorCallback('1')
    return Doc2Vec(seed=0, dm=0, vector_size=100,
                   min_count=100, epochs=10, workers=9,
                   hs=1, window=10, callbacks=[monitor])


def create_doc2vec_model():
    r = preprocess_records()

    logging.info("Data collected")

    model = build_model()
    model.build_vocab(r)

    logging.info("Vocabulary Built")
    model.train(r, total_examples=model.corpus_count, epochs=model.epochs)

    model.delete_temporary_training_data(
        keep_doctags_vectors=True, keep_inference=True)

    model.save('1.modelFile')
    logging.info("Completed")

    model = Doc2Vec.load('1.modelFile')


if __name__ == "__main__":
    create_doc2vec_model()
