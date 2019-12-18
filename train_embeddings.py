import psycopg2
from post_cleaning import process_text 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import TfidfModel
from gensim.corpora import Dictionary


def get_db_records(n):
    try:
        connection = psycopg2.connect( host="127.0.0.1",
                                  port="5432",
                                  database="crimebb")
        cursor = connection.cursor()

        query = 'SELECT "Content" FROM "Post" WHERE "Site" = 0 LIMIT ' + str(n)
        cursor.execute(query)
        records = [r[0] for r in cursor.fetchall()]
    except (Exception, psycopg2.Error) as error :
        print ("Error while fetching data from PostgreSQL", error)
    finally:
        #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

    return records

def preprocess_records():
    records = get_db_records(100000)

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
    return Doc2Vec(seed=0, dm=0, vector_size=100,
                   min_count=2, epochs=10, workers=8,
                   hs=1, window=10)

def create_doc2vec_model():
    r = preprocess_records()

    model = build_model()
    model.build_vocab(r)
    model.train(r, total_examples=model.corpus_count, epochs=model.epochs)

    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    model.save('.modelFile')
    return model




m, d = create_tfidf_model()
print(infer_tfidf(m, ['hello', 'world'], d))
