from itertools import product
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pandas as pd

# --- Caricamento dati ---
base_path = os.path.join(os.path.dirname(__file__), '..', 'Dat')
refugees_df = pd.read_csv(os.path.join(base_path, 'rifugiati.csv'))
lavori_df = pd.read_csv(os.path.join(base_path, 'Annunci_di_lavoro.csv'), sep=';')
ground_truth_df = pd.read_csv(os.path.join(base_path, 'Ground_True.csv'), sep=';')

# --- Creazione dizionario Ground Truth ---
ground_truth_dict = {(row['Rifugiati'], row['Annunci']): row['Match'] for _, row in ground_truth_df.iterrows()}

# --- Caricamento modello di embedding ---
model = SentenceTransformer("all-MiniLM-L6-v2")  # Usa un modello SentenceTransformer

# --- Generazione embeddings ---
columns_to_embed_refugees = {'Skill': 'Skill', 'Titolo di studio': 'Titolo di studio', 'Lingue parlate': 'Lingue parlate'}
columns_to_embed_jobs = {'Titolo Annuncio': 'Titolo Annuncio', 'Posizione Lavorativa': 'Posizione Lavorativa', 'Info Utili': 'Info Utili'}

refugees_embeddings = {
    key: model.encode(refugees_df[col].fillna('').tolist(), batch_size=32, convert_to_numpy=True)
    for key, col in columns_to_embed_refugees.items()
}

lavori_embeddings = {
    key: model.encode(lavori_df[col].fillna('').tolist(), batch_size=32, convert_to_numpy=True)
    for key, col in columns_to_embed_jobs.items()
}

# --- Costruzione matrice Ground Truth ---
true_labels = np.zeros((len(refugees_df), len(lavori_df)))

for i, refugee in refugees_df.iterrows():
    for j, job in lavori_df.iterrows():
        true_labels[i, j] = ground_truth_dict.get((refugee['Email'], job['ID']), 0)

# --- Definizione dei pesi da testare ---
weights_grid = [
    {'Skill': s, 'Titolo di studio': t, 'Lingue parlate': l, 'Posizione Lavorativa': p, 'Info Utili': i}
    for s, t, l, p, i in product(
        np.linspace(0.05, 0.5, 10),  # Range ristretto per 'Skill'
        np.linspace(0.05, 0.5, 10),   # Range ristretto per 'Titolo di studio'
        np.linspace(0.05, 0.5, 10),  # Range ristretto per 'Lingue'
        np.linspace(0.05, 0.5, 10),   # Range più ampio per 'Posizione'
        np.linspace(0.05, 0.5, 10)  # Range più ampio per 'Info Utili'
    ) if round(s + t + l + p + i, 1) == 1.0
]
# --- Funzione per valutare un set di pesi ---
def evaluate_weights(weights):
    similarity_matrix = np.zeros((len(refugees_df), len(lavori_df)))

    # Somma ponderata delle similarità
    for key in ['Skill', 'Titolo di studio', 'Lingue parlate']:
        similarity_matrix += weights[key] * cosine_similarity(refugees_embeddings[key], lavori_embeddings['Titolo Annuncio'])

    for key in ['Posizione Lavorativa', 'Info Utili']:
        similarity_matrix += weights[key] * cosine_similarity(refugees_embeddings['Skill'], lavori_embeddings[key])

    # Calcola la soglia per ogni rifugiato (85° percentile individuale)
    thresholds = np.percentile(similarity_matrix, 85, axis=1)

    # Predizione etichette (1 se sopra la soglia, altrimenti 0)
    predicted_labels = (similarity_matrix >= thresholds[:, None]).astype(int)

    return f1_score(true_labels.flatten(), predicted_labels.flatten())

# --- Trova la combinazione migliore di pesi ---
best_weights = max(weights_grid, key=evaluate_weights)
print("Migliori pesi trovati:", best_weights)
