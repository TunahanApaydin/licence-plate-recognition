import sqlite3

def create_sqlite_database(database_name):
  connection = None

  try:
    connection = sqlite3.connect(database_name + ".db")
  except sqlite3.Error as error:
    print(error)
  finally:
    if connection:
        print("Database created.")
        connection.close()

def create_table(database_name, sql_statements):
    
  try:
    with sqlite3.connect(database=database_name+".db") as connection:
      cursor = connection.cursor()

      for statement in sql_statements:
        cursor.execute(statement)

      connection.commit()
      print("Table created.")
  except sqlite3.Error as error:
    print(error)

def add_user(connection, user):
  sql = """INSERT INTO users(first_name, last_name, gender)
          VALUES(?,?,?)"""
  
  cursor = connection.cursor()
  cursor.execute(sql, user)
  connection.commit()

  return cursor.lastrowid

def add_task(connection, task):
  sql = """INSERT INTO tasks(task_name, task_owner_id, start_date, end_date)
          VALUES(?,?,?,?)"""
  
  cursor = connection.cursor()
  cursor.execute(sql, task)
  connection.commit()

  return cursor.lastrowid

def main(database_name):
  try:
    with sqlite3.connect(database=database_name+".db") as connection:
      user = ("Tunahan", "Apaydin", "25")
      user_id = add_user(connection=connection, user=user)
      print("Created user with the id {}".format(user_id))

      tasks = [("Learn SQL", user_id, "10.09.24", "17.09.24"),
               ("Learn Docker", user_id, "18.09.24", "25.09.24")]
      
      for task in tasks:
        task_id = add_task(connection=connection, task=task)
        print("Created user with the id {}".format(task_id))

  except sqlite3.Error as error:
    print(error)

if __name__ == "__main__":
  db_name = "test"
  # The IF NOT EXISTS will help us when reconnecting to the database.
  # The query will allow us to check if the table exists, and if it does, nothing is changed.
  create_table_statements = [
  """CREATE TABLE IF NOT EXISTS
  users
  (userid INTEGER PRIMARY KEY,
  first_name TEXT,
  last_name TEXT,
  gender INT);""",

  """
  CREATE TABLE IF NOT EXISTS tasks
  (taskid INTEGER PRIMARY KEY,
  task_name TEXT,
  task_owner_id INT,
  start_date TEXT,
  end_date TEXT,
  FOREIGN KEY (task_owner_id) REFERENCES users (userid));
  """]
  

  main(database_name=db_name)
  # create_table(database_name=db_name, sql_statements=create_table_statements)
  # create_sqlite_database(database_name=db_name)
