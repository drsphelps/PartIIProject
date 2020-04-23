from utils.db import db
from utils.post_cleaning import process_text_s as process_text
import numpy as np
import pickle


def get_db_records():
    """
    Gets the posts needed to train the word embedding models from a set of relevant forums
    """
    conn = db()
    records = []
    forums = [4, 10, 46, 48, 92, 107, 114,
              170, 186, 222, 293, 248, 167, 262]
    for forum in forums:
        query = 'SELECT p."Content" FROM "Post" p INNER JOIN "Thread" t ON p."Thread" = t."IdThread" WHERE p."Site" = 0 AND LENGTH(p."Content") > 200 AND t."Forum" =' + str(
            forum) + 'LIMIT 10000'
        records.extend([r[0] for r in conn.run_query(query)])
        print("Collected: " + str(forum))
    query = 'SELECT p."Content" FROM "Post" p INNER JOIN "Thread" t ON p."Thread" = t."IdThread" WHERE p."Site" = 0 AND LENGTH(p."Content") > 200 AND t."Forum" = 25 LIMIT 10000'
    records.extend([r[0] for r in conn.run_query(query)])

    conn.close_connection()
    return records


def noncrime_dataset(n, save, load=False):

    def load_dataset():
        with open('nc.data', 'rb') as f:
            X = pickle.load(f)
        np.random.shuffle(X)
        return X

    conn = db()
    if load:
        return load_dataset()
    posts = []
    for post in conn.get_noncrime_posts(n):
        if len(process_text(post[0])) > 10:
            posts.append(process_text(post[0]))
    X = np.stack(posts)
    if save:
        with open('nc.data', 'wb') as f:
            pickle.dump(X, f)
    np.random.shuffle(X)
    conn.close_connection()
    return X


def crime_dataset():
    def load_crime_data(folder):
        tests = []
        for i in range(0, 500):
            with open('data/' + folder + '_data/' + str(i) + '.data', 'r') as f:
                tests.append(process_text(f.read()))
        return tests

    posts = []
    for f in ['rat', 'ewhore', 'stresser', 'crypter']:
        posts.extend(load_crime_data(f))
    print(len(posts))
    return np.stack(posts)
