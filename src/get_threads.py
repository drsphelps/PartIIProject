import psycopg2
from utils.db import db
from post_cleaning import process_text, remove_tags


class get_threads():
    """
    Class with utility functions for creating the labelled dataset
    """

    @staticmethod
    def get_actors():
        """
        Returns the list of CrimeBB key actors
        """
        actors = []
        with open('key_actors.data', 'r') as f:
            line = f.readline()
            while line:
                line = line.strip()[:-2]
                if '\\' not in line and line != '':
                    actors.append(line)
                line = f.readline()
        return actors

    @staticmethod
    def run_query():
        """
        Get threads that each key actor has been involved with across a number of relevant forums
        """
        conn = db()

        forums = [107, 326, 44, 176, 186, 277, 46, 203, 48, 205, 374, 299, 92,
                  106, 10, 103, 91, 126, 339, 43, 114, 47, 113, 120, 70, 170, 4, 308]
        for forum in forums:
            for user in get_threads.get_actors():
                query = 'SELECT DISTINCT t."IdThread", t."Heading", t."NumPosts", t."Author" FROM "Post" p INNER JOIN "Thread" t ON p."Thread" = t."IdThread" WHERE t."Author" = ' + \
                    str(user) + ' AND t."Forum" = ' + str(forum) + ';'
                records = conn.run_query(query)

                for r in records:
                    print(r)

                with open("threads.data", "a", encoding='utf8') as f:
                    for r in records:
                        f.write(str(r[0]) + ' | ' + str(r[1]) +
                                ' | ' + str(r[2]) + ' | ' + str(r[3]) + '\n')

        conn.close_connection()

    @staticmethod
    def build_dataset(keywords):
        """
        Gets threads that include a specific keyword and creates a command line interface to add them to the dataset or not
        """
        conn = db()

        keyword = "'.*(" + '|'.join(keywords) + ").*'"

        query = '''SELECT "IdThread" FROM "Thread" WHERE LOWER("Heading") ~ ''' + \
            keyword + ''' AND "Site" = 0 AND "NumPosts" < 200'''
        threads = conn.run_query(query)

        length = 0

        print(len(threads))
        for thread in threads:
            thread_id = thread[0]

            posts = conn.get_posts_from_thread(thread_id)

            for post in posts:
                pp = process_text(post[1])
                if len(pp) > 5:
                    print(remove_tags(post[1]))
                    add = input()

                    with open(keywords[0]+"_training.data", 'a+') as f:
                        if add == 'y':
                            f.write(str(post[0]) + '\n')

                    with open(keywords[0]+"_training.data", 'r') as f:
                        length = len(f.readlines())

                    if length > 500:
                        print("All done")
                    print(
                        "==================================================================================" + str(length) + "\n")
