from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from get_training import get_db_records
from utils.post_cleaning import process_text


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
