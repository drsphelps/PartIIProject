from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import similarities
from get_training import get_db_records
from utils.post_cleaning import process_text

import pickle


class TFIDF():
    def __init__(self):
        pass

    def preprocess_tfidf(self):
        return [process_text(r) for r in get_db_records()]

    def create_tfidf_model(self):
        self.dataset = self.preprocess_tfidf()
        self.dct = Dictionary(self.dataset)
        self.dct.filter_extremes(no_below=50)
        corpus = [self.dct.doc2bow(line) for line in self.dataset]
        self.model = TfidfModel(corpus)

    def infer_tfidf(self):
        def infer(vector):
            dim = self.dct.keys()[-1] + 1
            text1 = self.model[self.dct.doc2bow(vector)]
            t1 = []
            for d in range(dim):
                t1_val = [i[1] for i in text1 if i[0] == d]
                if len(t1_val) == 1:
                    t1.append(t1_val[0])
                else:
                    t1.append(0)
            return t1
        return infer

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
