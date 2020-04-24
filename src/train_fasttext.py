from gensim.models import FastText, Word2Vec
from get_training import get_db_records
from utils.post_cleaning import process_text
import pickle
import logging
from datetime import datetime


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    filename=datetime.now().strftime('logs/%d%m%Y%H%M.log'),
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')


def preprocess_records():
    records = get_db_records()
    processed = []

    for i in range(len(records)):
        r = process_text(records[i])
        processed.append(r)

    return processed


def build_model():
    return FastText(size=100, window=10, min_count=50)


def create_fasttext_model():
    r = preprocess_records()
    with open("/mnt/d/ft_training_data.data", "wb") as f:
        pickle.dump(r, f)

    logging.info("Data collected")

    model = build_model()
    model.build_vocab(r)
    model.train(r, total_examples=model.corpus_count, epochs=10)

    model.save('data/models/fasttext.modelFile')

    model = FastText.load('data/models/fasttext.modelFile')


def test_model():
    model = Word2Vec.load('1ft.modelFile')
    print(model.most_similar(['pack']))


if __name__ == "__main__":
    create_fasttext_model()
    # test_model()
