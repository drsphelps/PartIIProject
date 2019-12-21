import psycopg2

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
    # TODO: Make this a class of its own
    forums = [107, 326, 44, 176, 186, 277, 46, 203, 48, 205, 374, 299, 92, 106, 10, 103, 91, 126, 339, 43, 114, 47, 113, 120, 70, 170, 4, 308]
    for forum in forums:
        for user in get_actors():
            try:
                connection = psycopg2.connect( host="127.0.0.1",
                                           port="5432",
                                           database="crimebb")
                cursor = connection.cursor()

                query = 'SELECT DISTINCT t."IdThread", t."Heading", t."NumPosts", t."Author" FROM "Post" p INNER JOIN "Thread" t ON p."Thread" = t."IdThread" WHERE t."Author" = ' + str(user) + ' AND t."Forum" = ' + str(forum) + ';'
                cursor.execute(query)
                records = cursor.fetchall()

                for r in records:
                    print(r)

                with open("threads.data", "a", encoding='utf8') as f:
                    for r in records:
                        f.write(str(r[0]) + ' | '  + str(r[1]) + ' | ' + str(r[2]) + ' | ' + str(r[3]) + '\n')

            except (Exception, psycopg2.Error) as error :
                print ("Error while fetching data from PostgreSQL", error)

            finally:
                if(connection):
                    cursor.close()
                    connection.close()

run_query()
