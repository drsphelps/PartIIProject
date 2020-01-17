import psycopg2
from utils.db import db
from post_cleaning import process_text
from get_threads import get_from_keyword
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.test.utils import get_tmpfile
from datetime import datetime
import logging
import os
import time

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', 
                    filename=datetime.now().strftime('logs/%H%M%d%m%Y.log'), 
                    level=logging.DEBUG, 
                    datefmt='%Y-%m-%d %H:%M:%S')

class MonitorCallback(CallbackAny2Vec):

    def __init__(self, path):
        self.epoch = 0
        self.path = path

    def on_epoch_end(self, model):
        logging.info("Epoch " + str(self.epoch) + " completed")
        output_path = get_tmpfile('{}_epoch{}.model'.format(self.path, self.epoch))
        model.save(output_path)
        self.epoch += 1    

def get_db_records():
    conn = db()
    records = []
    forums = [4, 10, 25, 46, 48, 92, 107, 114, 170, 186]
    for forum in forums:
        query = 'SELECT p."Content" FROM "Post" p INNER JOIN "Thread" t ON p."Thread" = t."IdThread" WHERE p."Site" = 0 AND LENGTH(p."Content") > 200 AND t."Forum" =' + str(forum) + 'LIMIT 100000'
        records.extend([r[0] for r in conn.run_query(query)])
    conn.close_connection()


    logging.info("Number of records collected: " + str(len(records)))
    return records


def preprocess_records():
    records = get_db_records()
    processed = []

    for i in range(len(records)):
        r = process_text(records[i])
        processed.append(TaggedDocument(r, [i]))

    return processed


def preprocess_tfidf():
    return [process_text(r) for r in get_db_records()]


def create_tfidf_model():
    dataset = preprocess_tfidf()
    dct = Dictionary(dataset)
    corpus = [dct.doc2bow(line) for line in dataset]
    return TfidfModel(corpus), dct


def infer_tfidf(model, vector, dct):
    bow = dct.doc2bow(vector)
    return model[bow]


def build_model():
    monitor = MonitorCallback('1')
    return Doc2Vec(seed=0, dm=0, vector_size=500,
                   min_count=100, epochs=10, workers=16,
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

    return model
