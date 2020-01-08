import psycopg2
from utils.db import db
from post_cleaning import process_text, remove_tags

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


def get_from_keyword(word, n):
    conn = db()
    ts = []
    ps = [] 

    query = '''SELECT * FROM "Thread" WHERE "Heading" like '%''' + word + '''%' AND "Site" = 0 AND "NumPosts" < 200 ORDER BY "NumPosts" DESC'''
    threads = conn.run_query(query)
    
    for thread in threads:
        ts.append(thread[0])

    for t in ts:
        posts = conn.get_posts_from_thread(t)
        added = 0
        for post in posts:
            if len(ps) < n and len(post[1]) > 200:
                ps.append(post)
                added += 1
        if len(ps) >= n:
            break
    
    conn.close_connection()

    return ps


def build_dataset(keywords):
    conn = db()

    keyword = "'.*(" + '|'.join(keywords) + ").*'"

    query = '''SELECT "IdThread" FROM "Thread" WHERE LOWER("Heading") ~ ''' + keyword + ''' AND "Site" = 0 AND "NumPosts" < 200'''
    threads = conn.run_query(query)
    
    length = 0

    for thread in threads:
        thread_id = thread[0]

        posts = conn.get_posts_from_thread(thread_id)
        
        for post in posts:
            pp = process_text(post[1])
            if len(pp) > 5:
                if "stresser" not in pp:
                    print(remove_tags(post[1]))
                    add = input()


                    with open(keywords[0]+"_training.data", 'a+') as f:
                        if add == 'y':
                            f.write(str(post[0]) +'\n')
                
                    with open(keywords[0]+"_training.data", 'r') as f:
                        length = len(f.readlines())

                    if length > 500:
                        print("All done")
                    print("==================================================================================" + str(length) + "\n")
