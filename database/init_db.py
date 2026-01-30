import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DATABASE")


# Create new database 
def create_database():
    
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database='postgres'
    )

  
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE rag_test_vector;")
    cursor.close()
    conn.close()
    print('DATABASE CREATED')



# Enable vector extension 
def enable_vector_extension():
    
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    print("CONNECTED")

    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    cursor.close()
    conn.close()
    print('EXTENSION ENABLED')


if __name__ == "__main__":
    # create_database()
    enable_vector_extension()