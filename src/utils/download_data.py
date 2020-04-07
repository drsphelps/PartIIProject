from db import db

with open("data/crypter_training.data", "r") as f:
    posts = f.read().split('\n')[:-1]

conn = db()
i = 0
for post in posts:
    with open("data/crypter_data/" + str(i) + ".data", "w") as g:
        g.write(conn.get_content_from_post(post, 0))
    i += 1
