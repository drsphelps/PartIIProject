from gensim.models import FastText, Word2Vec
from utils.get_training import get_db_records
from post_cleaning import process_text


def preprocess_records():
    records = get_db_records()
    processed = []

    for i in range(len(records)):
        r = process_text(records[i])
        processed.append(r)

    return processed


def build_model():
    return FastText(size=100, window=5, min_count=100)


def create_fasttext_model():
    r = preprocess_records()

    model = build_model()
    model.build_vocab(r)
    model.train(r, total_examples=model.corpus_count, epochs=10)

    model.save('1ft.modelFile')

    model = FastText.load('1ft.modelFile')


def test_model():
    model = Word2Vec.load('1ft.modelFile')
    print(model.most_similar(['pack']))


if __name__ == "__main__":
    create_fasttext_model()
    # test_model()
