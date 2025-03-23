import pymysql
import pandas as pd

# Connessione al database
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='guardian',
        database='bridge',
    )