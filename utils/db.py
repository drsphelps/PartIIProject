import psycopg2


class db:
    def __init__(self):
        self.connection = self.get_conn()
        self.cursor = self.connection.cursor()

    def get_conn(self):
        try:
            connection = psycopg2.connect(host="127.0.0.1",
                                          port="5432",
                                          database="crimebb")
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)

        return connection

    def run_query(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def close_connection(self):
        if(self.connection):
            self.cursor.close()
            self.connection.close()
