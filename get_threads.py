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


def get_from_keyword(word):
    conn = db()
    ts = []
    ps = [] 

    query = '''SELECT * FROM "Thread" WHERE "Heading" like '%''' + word + '''%' AND "Site" = 0 AND "NumPosts" < 200 ORDER BY "NumPosts" DESC LIMIT 200;'''
    threads = conn.run_query(query)

    for thread in threads:
        ts.append(thread[0])

    for t in ts:
        posts = conn.get_posts_from_thread(t)
        added = 0
        for post in posts:
            if len(ps) < 500 and len(post[1]) > 200:
                ps.append(post)
                added += 1
        if len(ps) >= 500:
            break
    return ps

