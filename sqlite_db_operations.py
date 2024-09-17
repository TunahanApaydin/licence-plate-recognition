import sqlite3

class DB(object):
    def __init__(self, database_name: str) -> None:
        self.connection = None
        self.database_name  = database_name
        
    def create_sqlite_database(self):
        try:
            self.connection = sqlite3.connect(self.database_name + ".db")
        except sqlite3.Error as error:
            print(error)
        finally:
            self.close_database_connection()
            
    def connect_database(self):
        try:
            self.connection = sqlite3.connect(self.database_name + ".db")
        except sqlite3.Error as error:
            print(error)
        finally:
            print("Connected to DB: {}".format(self.database_name+".db"))
            return self.connection
    
    def close_connection(self):
        try: 
            if self.connection:
                print("Database connection closed.")
                self.connection.close()
        except sqlite3.Error as error:
            print(error)

    def create_table(self, sql_statements: list):
        try:
            cursor = self.connection.cursor()

            for statement in sql_statements:
                cursor.execute(statement)

            self.connection.commit()
            print("Tables created.")
        except sqlite3.Error as error:
            print(error)
    
    def insert_data(self, sql_statement: str, data: tuple):
        """
        
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql_statement, data)
            self.connection.commit()
        except sqlite3.Error as error:
            print(error)
        finally:
            return cursor.lastrowid


    # def add_user(self, connection, user):
    #     sql = """INSERT INTO users(first_name, last_name, gender)
    #             VALUES(?,?,?)"""
        
    #     cursor = connection.cursor()
    #     cursor.execute(sql, user)
    #     connection.commit()

    #     return cursor.lastrowid

    # def add_task(self, connection, task):
    #     sql = """INSERT INTO tasks(task_name, task_owner_id, start_date, end_date)
    #             VALUES(?,?,?,?)"""
        
    #     cursor = connection.cursor()
    #     cursor.execute(sql, task)
    #     connection.commit()

    #     return cursor.lastrowid

    # def add_data(self, database_name):
    #     try:
    #         with sqlite3.connect(database=database_name+".db") as connection:
    #             user = ("Tunahan", "Apaydin", "25")
    #             user_id = add_user(connection=connection, user=user)
    #             print("Created user with the id {}".format(user_id))

    #             tasks = [("Learn SQL", user_id, "10.09.24", "17.09.24"),
    #                     ("Learn Docker", user_id, "18.09.24", "25.09.24")]
                
    #             for task in tasks:
    #                 task_id = add_task(connection=connection, task=task)
    #                 print("Created user with the id {}".format(task_id))

    #     except sqlite3.Error as error:
    #         print(error)
            
    #     finally:
    #         if connection:
    #             connection.close()
    #             print("Database connection closed.")
                
    def get_data(self, sql_query: str):
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql_query)
            query_result = cursor.fetchall()
            
            return query_result
                
        except sqlite3.Error as error:
            print(error)
            
if __name__ == "__main__":
    
    # The IF NOT EXISTS will help us when reconnecting to the database.
    # The query will allow us to check if the table exists, and if it does, nothing is changed.
    # create_table_statements = [
    # """CREATE TABLE IF NOT EXISTS
    # users
    # (userid INTEGER PRIMARY KEY,
    # first_name TEXT,
    # last_name TEXT,
    # gender INT);""",

    # """
    # CREATE TABLE IF NOT EXISTS tasks
    # (taskid INTEGER PRIMARY KEY,
    # task_name TEXT,
    # task_owner_id INT,
    # start_date TEXT,
    # end_date TEXT,
    # FOREIGN KEY (task_owner_id) REFERENCES users (userid));
    # """]
    
    db_name = "test" 
    db = DB(database_name=db_name)
    db.connect_database()
    
    data = ("Ali", "Veli", "Erkek")
    statement = "INSERT INTO {}{} VALUES(?,?,?)".format("users", "(first_name, last_name, gender)")
    db.insert_data(sql_statement=statement, data=data)
    
    select_query = "SELECT {} from {}".format("*", "users")
    query_result = db.get_data(sql_query=select_query)
    print(query_result)
    db.close_connection()
    #add_data(database_name=db_name)
    #create_table(database_name=db_name, sql_statements=create_table_statements)
    #create_sqlite_database(database_name=db_name)