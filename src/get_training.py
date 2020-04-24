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


def noncrime_dataset(n, save=False, load=False):
    """
    Produces a dataset of noncrime posts, or loads it from a file
    """
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
        if len(process_text_s(post[0])) > 10:
            posts.append(process_text_s(post[0]))
    X = np.stack(posts)
    if save:
        with open('nc.data', 'wb') as f:
            pickle.dump(X, f)
    np.random.shuffle(X)
    conn.close_connection()
    return X


def load_crime_data(folder):
    tests = []
    for i in range(0, 500):
        with open('data/' + folder + '_data/' + str(i) + '.data', 'r') as f:
            tests.append(f.read())
    return tests


def crime_dataset():
    """
    Loads the dataset of crime posts
    """

    posts = []
    for f in ['rat', 'ewhore', 'stresser', 'crypter']:
        posts.extend(load_crime_data(f))
    print(len(posts))
    return np.stack(posts)


def collect_training_data(load, split, number_unlabelled, d2v_model):
    def collect_unlabelled_data(forum):
        conn = db()
        posts = []
        for post in conn.get_posts_from_forum(forum, number_unlabelled):
            posts.append(post[0])
        conn.close_connection()
        print("Collected: " + str(forum))
        return posts

    # Get labelled data
    labelled_e = load_crime_data("ewhore")
    labelled_s = load_crime_data("stresser")
    labelled_r = load_crime_data("rat")
    labelled_c = load_crime_data("crypter")

    # Collect or load unlabelled data
    if load:
        with open("cluster.data", "rb") as f:
            U_train = pickle.load(f)
    else:
        unlabelled_e = collect_unlabelled_data(170)
        unlabelled_s = collect_unlabelled_data(92)
        unlabelled_r = collect_unlabelled_data(114)
        unlabelled_c = collect_unlabelled_data(299)
        U_train = unlabelled_e + unlabelled_s + unlabelled_r + unlabelled_c
        with open("cluster.data", "wb") as f:
            pickle.dump(U_train, f)

    X_e, X_et = train_test_split(
        labelled_e, test_size=split, shuffle=True)
    X_s, X_st = train_test_split(
        labelled_s, test_size=split, shuffle=True)
    X_r, X_rt = train_test_split(
        labelled_r, test_size=split, shuffle=True)
    X_c, X_ct = train_test_split(
        labelled_c, test_size=split, shuffle=True)

    X_train = X_e + X_s + X_r + X_c
    X_test = X_et + X_st + X_rt + X_ct

    train_size = int((1.-split) * 500)
    test_size = int(split * 500)

    constraints = ([0] * train_size) + ([1] * train_size) + \
        ([2] * train_size) + ([3] * train_size)

    test_constraints = [0] * test_size + [1] * \
        test_size + [2] * test_size + [3] * test_size

    for i in range(len(X_train)):
        X_train[i] = KMeansPoint(d2v_model.infer_vector(
            process_text(X_train[i])), constraint=constraints[i])
    X_train = np.stack(X_train)

    for i in range(len(U_train)):
        U_train[i] = KMeansPoint(d2v_model.infer_vector(
            process_text(U_train[i])), -1)
    if len(U_train) == 0:
        U_train = np.array([])
    else:
        U_train = np.stack(U_train)

    for i in range(len(X_test)):
        X_test[i] = KMeansPoint(d2v_model.infer_vector(
            process_text(X_test[i])), constraint=test_constraints[i], text=X_test[i])
    X_test = np.stack(X_test)

    return X_train, U_train, X_test


def test(self):
    if self.train():
        correct = 0.
        for test in range(self.X_test.shape[0]):
            if self.pred(self.X_test[test]):
                correct += 1.
        return correct / self.X_test.shape[0]
