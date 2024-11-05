import sqlite3

class DB(object):
    def __init__(self, database_name: str) -> None:
        self.connection = None
        self.database_name  = database_name
        
    def __create_sqlite_database(self):
        try:
            self.connection = sqlite3.connect(self.database_name + ".db")
        except sqlite3.Error as error:
            print(error)
            
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

    def __create_table(self, sql_statements: list):
        try:
            cursor = self.connection.cursor()

            for statement in sql_statements:
                cursor.execute(statement)

            self.connection.commit()
            print("Tables created.")
        except sqlite3.Error as error:
            print(error)
    
    def __insert_data(self, sql_statement: str, data: tuple):
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql_statement, data)
            self.connection.commit()
        except sqlite3.Error as error:
            print(error)
        finally:
            return cursor.lastrowid
                
    def get_data(self, sql_query: str):
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql_query)
            query_result = cursor.fetchall()
            
            return query_result
                
        except sqlite3.Error as error:
            print(error)
    
    def check_plate_in_database(self, table_name: str, plate: str) -> dict:
        cursor = self.connection.cursor()
        
        query = f"SELECT first_name, last_name FROM {table_name} WHERE licence_plate = ?"
        cursor.execute(query, (plate,))
        
        result = cursor.fetchone()
        
        if result:
            return True, {"first_name": result[0], "last_name": result[1]}
        else:
            return False, None
            
if __name__ == "__main__":
    
    # The IF NOT EXISTS will help us when reconnecting to the database.
    # The query will allow us to check if the table exists, and if it does, nothing is changed.
    
    # create_table_statements = [
    # """CREATE TABLE IF NOT EXISTS
    # users
    # (userid INTEGER PRIMARY KEY,
    # first_name TEXT,
    # last_name TEXT,
    # licence_plate TEXT);""",]
    
    db_name = "registered_lp" 
    db = DB(database_name=db_name)
    db.connect_database()
    
    # db.create_table(create_table_statements)
    
    # data = ("Name", "Surname", "32YN166")
    # statement = "INSERT INTO {}{} VALUES(?,?,?)".format("users", "(first_name, last_name, licence_plate)")
    # db.__insert_data(sql_statement=statement, data=data)
    
    # select_query = "SELECT {} from {}".format("*", "users")
    # query_result = db.get_data(sql_query=select_query)
    # print(query_result)
    
    # result = db.check_plate_in_database(table_name="users", plate="35GZ213")
    # print(result)
    
    db.close_connection()
    #add_data(database_name=db_name)
    #create_table(database_name=db_name, sql_statements=create_table_statements)
    #create_sqlite_database(database_name=db_name)