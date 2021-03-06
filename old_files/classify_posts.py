from utils.db import db

threads = []
with open('threads.data', 'r') as f:
    line = f.readline()
    while line:
        threads.append(line)
        line = f.readline()

conn = db()

crime_threads = []
noncrime_threads = []

for thread in threads:
    print(thread)
    threadId = thread.split('|')[0].strip()
    query = 'SELECT "IdPost", "Content" FROM "Post" WHERE "Thread" = ' + \
        threadId + ' AND "Site" = 0'
    posts = sorted(conn.run_query(query))
    
    for post in posts:
        print(post[1].strip('\n'))
        print('\n\n')
    crime = input()
    if crime == 'y':
        crime_threads.append(thread)
    else:
        noncrime_threads.append(thread)
    conn.close_connection()

with open('crime.data', 'a') as f:
    for c in crime_threads:
        f.write(c)

with open('noncrime.data', 'a') as f:
    for n in noncrime_threads:
        f.write(n)
