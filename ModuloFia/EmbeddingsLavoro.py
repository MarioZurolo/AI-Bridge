import pandas as pd
import mysql.connector
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Configurazione del database MySQL
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "guardian",
    "database": "bridge"
}

# Percorso file CSV
CSV_PATH = "lavori_embeddings.csv"

# Caricare il modello per gli embedding
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

def get_jobs_from_db():
    conn = mysql.connector.connect(**DB_CONFIG)
    query = "SELECT ID, titolo, posizione_lavorativa, info_utili FROM lavoro"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def compute_embedding(text):
    return model.encode(text, convert_to_numpy=True)

def update_embeddings():
    jobs_df = get_jobs_from_db()
    if os.path.exists(CSV_PATH):
        saved_df = pd.read_csv(CSV_PATH)
    else:
        saved_df = pd.DataFrame(columns=["ID", "titolo", "posizione_lavorativa", "info_utili", "Embedding"])
    
    updated_rows = []
    for _, row in jobs_df.iterrows():
        existing = saved_df[saved_df["ID"] == row["ID"]]
        if existing.empty or not all(existing.iloc[0][["titolo", "posizione_lavorativa", "info_utili"]] == row[["titolo", "posizione_lavorativa", "info_utili"]]):
            embedding = compute_embedding(row["titolo"] + " " + row["posizione_lavorativa"] + " " + row["info_utili"])
            row["Embedding"] = embedding.tolist()
            updated_rows.append(row)
        else:
            updated_rows.append(existing.iloc[0])
    
    updated_df = pd.DataFrame(updated_rows)
    updated_df.to_csv(CSV_PATH, index=False)
    return updated_df

if __name__ == "__main__":
    updated_df = update_embeddings()
    print("Embeddings aggiornati e salvati.")
