import psycopg2

try:
   connection = psycopg2.connect( host="127.0.0.1",
                                  port="5432",
                                  database="crimebb")
   cursor = connection.cursor()

   query = 'SELECT "Content" FROM "Post" LIMIT 100'
   cursor.execute(query)
   records = [r[0] for r in cursor.fetchall()]

   with open("examples.data", "w", encoding='utf8') as f:
       for r in records:
           f.write(r)
           f.write("===LINESPLIT===")

except (Exception, psycopg2.Error) as error :
    print ("Error while fetching data from PostgreSQL", error)

finally:
    #closing database connection.
    if(connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
