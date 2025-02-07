import pymysql
import pandas as pd

# Connessione al database
connection = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='guardian',
    database='bridge'
)

query = "SELECT * FROM lavoro"
lavoro_df = pd.read_sql(query, connection)

query = "SELECT * FROM alloggio"
alloggio_df = pd.read_sql(query, connection)

# Chiudi la connessione
connection.close()

# Salva i dati in CSV
lavoro_df.to_csv("Annunci_di_lavoro.csv", index=False, sep=";")
alloggio_df.to_csv("Alloggi.csv", index=False, sep=";")
