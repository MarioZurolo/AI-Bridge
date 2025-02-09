import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import ast

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Percorsi ai file
BASE_PATH = os.path.dirname(__file__)
JOBS_EMBEDDINGS_PATH = os.path.join(BASE_PATH, "lavori_embeddings.csv")

# Caricamento modello
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')


def load_job_embeddings():
    """Carica il dataframe con gli embeddings dei lavori, se esiste."""
    if os.path.exists(JOBS_EMBEDDINGS_PATH):
        return pd.read_csv(JOBS_EMBEDDINGS_PATH)
    return None


def compute_embedding(text):
    """Computa l'embedding di un testo usando Sentence-BERT."""
    return model.encode([text], convert_to_numpy=True)[0]


def get_refugee_data(email, db_connection):
    """Recupera i dati del rifugiato dal database."""
    query = """
    SELECT skill, lingue_parlate, titolo_di_studio FROM utente WHERE email = %s
    """
    df = pd.read_sql(query, db_connection, params=(email,))
    return df.iloc[0] if not df.empty else None


def get_job_data(db_connection):
    """Recupera gli annunci di lavoro dal database."""
    query = """
    SELECT id, titolo, posizione_lavorativa, info_utili FROM lavoro
    """
    return pd.read_sql(query, db_connection)


def match_jobs(refugee_email, db_connection):
    """Trova i 3 migliori annunci di lavoro per il rifugiato."""
    logging.info(f"Inizio raccomandazione per {refugee_email}")
    refugee = get_refugee_data(refugee_email, db_connection)
    if refugee is None:
        logging.warning("Rifugiato non trovato nel database")
        return []
    
    # Carica gli embeddings dei lavori
    jobs_df = get_job_data(db_connection)
    jobs_embeddings_df = load_job_embeddings()
    
    # Controlla coerenza tra database e CSV
    embeddings_to_update = []
    jobs_embeddings = []
    
    for _, job in jobs_df.iterrows():
        job_id = job['id']
        matching_row = jobs_embeddings_df[jobs_embeddings_df['id'] == job_id] if jobs_embeddings_df is not None else None
        
        if matching_row is not None and not matching_row.empty:
            # Verifica se i testi coincidono
            stored_title = matching_row.iloc[0]['titolo']
            stored_position = matching_row.iloc[0]['posizione_lavorativa']
            stored_info = matching_row.iloc[0]['info_utili']
            
            if (stored_title == job['titolo'] and
                stored_position == job['posizione_lavorativa'] and
                stored_info == job['info_utili']):
                # Usa embedding già calcolato
                embedding_str = matching_row.iloc[0]['embedding']
                embedding_array = np.array(ast.literal_eval(embedding_str))  # Converte la stringa in array
                jobs_embeddings.append(embedding_array)                
                continue
        
        # Calcola nuovo embedding
        new_embedding = compute_embedding(f"{job['titolo']} {job['posizione_lavorativa']} {job['info_utili']}")
        embeddings_to_update.append({
            'id': job_id,
            'titolo': job['titolo'],
            'posizione_lavorativa': job['posizione_lavorativa'],
            'info_utili': job['info_utili'],
            'embedding': new_embedding.tolist()
        })
        jobs_embeddings.append(new_embedding)
    
    # Se necessario, aggiorna il CSV
    if embeddings_to_update:
        updated_df = pd.DataFrame(embeddings_to_update)
        updated_df.to_csv(JOBS_EMBEDDINGS_PATH, index=False)
    
    # Compute refugee embedding
    refugee_text = f"{refugee['skill']} {refugee['lingue_parlate']} {refugee['titolo_di_studio']}"
    refugee_embedding = compute_embedding(refugee_text)
    
    # Compute cosine similarity
    similarities = cosine_similarity([refugee_embedding], jobs_embeddings)[0]
    
    # Trova i 3 migliori match
    top_indices = np.argsort(similarities)[-3:][::-1]
    top_matches = [{
    "id": int(jobs_df.iloc[idx]['id']),  # Converte np.int64 in int
    "titolo": jobs_df.iloc[idx]['titolo'],
    "similarita": float(similarities[idx])  # Converte np.float32 in float
    } for idx in top_indices]

    
    logging.info(f"Top 3 match per {refugee_email}: {top_matches}")
    return top_matches
