from utils.db import db


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
