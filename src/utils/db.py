import psycopg2


class db:
    """
    Class that provides an interface over the Postgres database, with some common methods
    """

    def __init__(self):
        self.connection = self.get_conn()
        self.cursor = self.connection.cursor()

    def get_conn(self):
        """
        Attempts to get the connection
        """
        try:
            connection = psycopg2.connect(host="127.0.0.1",
                                          port="5432",
                                          database="crimebb",
                                          user="drsp2")
            return connection
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)

    def run_query(self, query):
        """
        Runs and returns the results from an arbitrary query
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_posts_from_thread(self, threadId):
        """
        Gets IDs and content from all the posts in a thread
        """
        query = 'SELECT "IdPost", "Content" FROM "Post" WHERE "Thread" = ' + \
            str(threadId) + 'AND "Site" = 0 AND LENGTH("Content") > 100 ORDER BY "IdPost" ASC LIMIT 10;'
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_content_from_post(self, postId, siteId):
        """
        Gets the content from a specific post on a specific site
        """
        query = 'SELECT "Content" FROM "Post" WHERE "IdPost" = ' + \
            str(postId) + ' AND "Site" = ' + str(siteId) + ';'
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        return result[0][0]

    def get_noncrime_posts(self, n):
        """
        Gets a set of noncrime posts from the database
        """
        # 65, 32, 128
        forums = [222, 293, 248, 167, 262]
        results = []
        for forum in forums:
            query = '''SELECT DISTINCT p."Content" FROM "Post" p INNER JOIN "Thread" t ON t."IdThread" = p."Thread" WHERE t."Site" = 0 AND t."Forum" = ''' + str(
                    forum) + ''' AND LENGTH(p."Content") > 200 AND LOWER(t."Heading") not similar to '%(ewhor|e-whor|hack|crypt|stresser|booter|ddos| rat |dump|phish|exploit|botnet|spoiler)%' AND LOWER(p."Content") not similar to '%(hidden content.|register|spoiler)%' LIMIT ''' + str(n)
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            results.extend(result)
            print(len(results))
        return results

    def get_posts_from_forum(self, fid, n):
        """
        Gets n posts from a specific subforum
        """
        query = '''SELECT DISTINCT p."Content" FROM "Post" p INNER JOIN "Thread" t ON t."IdThread" = p."Thread" WHERE t."Site" = 0 AND t."Forum" = ''' + \
            str(fid) + ''' AND LENGTH(p."Content") > 200 LIMIT ''' + str(n)
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def close_connection(self):
        """
        Closes the connection to the database
        """
        if(self.connection):
            self.cursor.close()
            self.connection.close()
