import psycopg2


class db:
    def __init__(self):
        self.connection = self.get_conn()
        self.cursor = self.connection.cursor()

    def get_conn(self):
        try:
            connection = psycopg2.connect(host="127.0.0.1",
                                          port="3333",
                                          database="crimebb",
                                          user="drsp2")
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)

        return connection

    def run_query(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_posts_from_thread(self, threadId):
        query = 'SELECT "IdPost", "Content" FROM "Post" WHERE "Thread" = ' + \
            str(threadId) + 'AND "Site" = 0 AND LENGTH("Content") > 100 ORDER BY "IdPost" ASC LIMIT 10;'
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_content_from_post(self, postId, siteId):
        query = 'SELECT "Content" FROM "Post" WHERE "IdPost" = ' + \
            str(postId) + ' AND "Site" = ' + str(siteId) + ';'
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        return result[0][0]

    def get_noncrime_posts(self, n):
        forums = [222, 65, 293, 248, 167, 32, 262, 128]
        results = []
        for forum in forums:
            query = '''SELECT DISTINCT p."Content" FROM "Post" p INNER JOIN "Thread" t ON t."IdThread" = p."Thread" WHERE t."Site" = 0 AND t."Forum" = ''' + str(
                    forum) + ''' AND LENGTH(p."Content") > 200 AND LOWER(t."Heading") not similar to '%(ewhor|e-whor|hack|crypt|stresser|booter|ddos| rat |dump|phish|exploit|botnet)%' AND LOWER(p."Content") not similar to '%(hidden content.|register)%' LIMIT ''' + str(
                    n)
            self.cursor.execute(query)  
            result = self.cursor.fetchall()
            results.extend(result)
        return results

    def close_connection(self):
        if(self.connection):
            self.cursor.close()
            self.connection.close()#
