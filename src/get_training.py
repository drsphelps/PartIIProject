from utils.db import db
from utils.post_cleaning import process_text_s, process_text
from kmeans_point import KMeansPoint
from sklearn.model_selection import train_test_split
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


def noncrime_dataset(n, save=False, load=False, string=False):
    """
    Produces a dataset of noncrime posts, or loads it from a file
    """
    def load_dataset():
        with open('nc.data', 'rb') as f:
            X = pickle.load(f)
        np.random.shuffle(X)
        return X

    pt = process_text_s if string else process_text

    conn = db()

    if load:
        return load_dataset()

    posts = []
    for post in conn.get_noncrime_posts(n*2):
        processed = pt(post[0])
        if len(processed) > 10:
            p = (processed, -1)
            posts.append(p)
    conn.close_connection()
    X = np.stack(posts)

    if save:
        with open('nc.data', 'wb') as f:
            pickle.dump(X, f)

    np.random.shuffle(X)
    return X


def load_crime_data(folder, cl, string=False):
    """
    Loads the crime data from
    """
    pt = process_text_s if string else process_text
    posts = []
    for i in range(0, 500):
        with open('data/' + folder + '_data/' + str(i) + '.data', 'r') as f:
            p = (pt(f.read()), cl)
            posts.append(p)
    return posts


def crime_dataset(string=False):
    """
    Loads the dataset of crime posts
    """
    posts = []
    for i, f in enumerate(['rat', 'ewhore', 'stresser', 'crypter']):
        posts.extend(load_crime_data(f, i, string))
    return np.stack(posts)


def collect_unlabelled_data(forum, number, string=False):
    conn = db()
    posts = []
    for post in conn.get_posts_from_forum(forum, number):
        p = (pt(post[0]), -1)
        posts.append(p)
    conn.close_connection()
    return posts
