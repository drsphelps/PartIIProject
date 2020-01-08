import psycopg2
from utils.db import db
from post_cleaning import process_text
from get_threads import get_from_keyword
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import os

class MonitorCallback(CallbackAny2Vec):
    def on_epoch_end(self, model):
        print("Model loss: ", model.get_latest_training_loss())


def get_db_records(n):
    conn = db()
    query = 'SELECT "Content" FROM "Post" WHERE "Site" = 0 AND LENGTH("Content") > 200 LIMIT 1000000'
    records = [r[0] for r in conn.run_query(query)]
    conn.close_connection()


    print(len(records))
    return records


def preprocess_records():
    records = get_db_records(10000)

    for i in range(len(records)):
        r = process_text(records[i])
        yield TaggedDocument(r, [i])


def preprocess_tfidf():
    return [process_text(r) for r in get_db_records(100000)]


def create_tfidf_model():
    dataset = preprocess_tfidf()
    dct = Dictionary(dataset)
    corpus = [dct.doc2bow(line) for line in dataset]
    return TfidfModel(corpus), dct


def infer_tfidf(model, vector, dct):
    bow = dct.doc2bow(vector)
    return m[bow]


def build_model():
    monitor = MonitorCallback()
    return Doc2Vec(seed=0, dm=0, vector_size=100,
                   min_count=2, epochs=10, workers=8,
                   hs=1, window=10)


def create_doc2vec_model():
    r = preprocess_records()
    print("GOT POSTS")

    model = build_model()
    model.build_vocab(r)
    print("VOCAB BUILT")
    model.train(r, total_examples=model.corpus_count, epochs=model.epochs)

    model.delete_temporary_training_data(
        keep_doctags_vectors=True, keep_inference=True)

    model.save('1.modelFile')
    return model

create_doc2vec_model()
