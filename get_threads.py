import psycopg2
from utils.db import db


def get_actors():
    actors = []
    with open('key_actors.data', 'r') as f:
        line = f.readline()
        while line:
            line = line.strip()[:-2]
            if '\\' not in line and line != '':
                actors.append(line)
            line = f.readline()
    return actors


def run_query():
    conn = db()

    forums = [107, 326, 44, 176, 186, 277, 46, 203, 48, 205, 374, 299, 92,
              106, 10, 103, 91, 126, 339, 43, 114, 47, 113, 120, 70, 170, 4, 308]
    for forum in forums:
        for user in get_actors():
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


run_query()
