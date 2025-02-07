import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
import nltk
from nltk.corpus import stopwords
import string
import logging
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import defaultdict

# Scarica le stopwords solo una volta
nltk.download('stopwords')

# Stopwords in italiano
stopwords_it = set(stopwords.words('italian'))

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Caricamento dati
base_path = os.path.join(os.path.dirname(__file__), '..', 'Dat')
refugees_df = pd.read_csv(os.path.join(base_path, 'rifugiati.csv'))
lavori_df = pd.read_csv(os.path.join(base_path, 'Annunci_di_lavoro.csv'), sep=';')
ground_truth_df = pd.read_csv(os.path.join(base_path, 'Ground_True.csv'), sep=';')



# Creazione dizionario per ground truth
ground_truth_dict = defaultdict(set)
for _, row in ground_truth_df.iterrows():
    print(row)
    if row['Match'] == 1:
        ground_truth_dict[row['Rifugiati']].add(row['Annunci'])

# Funzione migliorata per rimuovere stopwords e punteggiatura
def remove_stopwords(text):
    if not isinstance(text, str):
        return ""  # Se il valore non è una stringa, restituisce una stringa vuota
    text = text.lower()  # Converti in minuscolo
    text = text.translate(str.maketrans("", "", string.punctuation))  # Rimuove punteggiatura
    return ' '.join(word for word in text.split() if word not in stopwords_it)


# Pre-elaborazione dei testi
columns_to_clean = {
    'Skill': 'Testo_Skill', 'Titolo di studio': 'Testo_Titolo', 'Lingue parlate': 'Testo_Lingue'
}
for col, new_col in columns_to_clean.items():
    refugees_df[new_col] = refugees_df[col].fillna('').apply(remove_stopwords)

columns_to_clean_jobs = {
    'Titolo Annuncio': 'Testo_Titolo_Annuncio', 'Posizione Lavorativa': 'Testo_Posizione', 'Info Utili': 'Testo_Info'
}
for col, new_col in columns_to_clean_jobs.items():
    lavori_df[new_col] = lavori_df[col].fillna('').apply(remove_stopwords)

# Funzione di valutazione
def evaluate_model(model_name):
    model = SentenceTransformer(model_name)
    logging.info(f"Generazione embeddings per {model_name}...")
    
    refugees_embeddings = {
        key: model.encode(refugees_df[col].tolist(), batch_size=32, convert_to_numpy=True)
        for key, col in columns_to_clean.items()
    }
    lavori_embeddings = {
        key: model.encode(lavori_df[col].tolist(), batch_size=32, convert_to_numpy=True)
        for key, col in columns_to_clean_jobs.items()
    }

    logging.info(f"Calcolo similarità per {model_name}...")
    weights = {'Skill': 0.1, 'Titolo di studio': 0.05, 'Lingue parlate': 0.05}
    similarity_matrix = sum(
        weights[key] * cosine_similarity(refugees_embeddings[key], lavori_embeddings['Titolo Annuncio'])
        for key in weights
    )
    similarity_matrix += sum(
        w * cosine_similarity(refugees_embeddings['Skill'], lavori_embeddings[k])
        for k, w in {'Posizione Lavorativa': 0.26, 'Info Utili': 0.54}.items()
    )

    matches, true_labels, predicted_labels = [], [], []
    
    for i, similarities in enumerate(similarity_matrix):
        top_indices = np.argsort(similarities)[-3:][::-1]  # Primi 3 annunci più simili
        refugee_name = refugees_df.iloc[i]['Email']
        print(f"Refugee Email: {refugee_name}, Ground Truth: {ground_truth_dict.get(refugee_name, set())}")

        # Controlla se almeno uno dei tre annunci suggeriti è nel ground truth
        true_match = any(lavori_df.iloc[idx]['ID'] in ground_truth_dict.get(refugee_name, set()) for idx in top_indices)

        true_labels.append(1 if true_match else 0)

        max_similarity = max(similarities[idx] for idx in top_indices)
        threshold = np.percentile(similarity_matrix, 85)
        predicted_labels.append(1 if max_similarity >= threshold else 0)

    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    accuracy = accuracy_score(true_labels, predicted_labels)

    return {"model": model_name, "precision": precision, "recall": recall, "f1_score": f1, "accuracy": accuracy}

models = ['distilbert-base-nli-mean-tokens', 'all-MiniLM-L6-v2', 'multi-qa-mpnet-base-dot-v1']
results = [evaluate_model(model) for model in models]
logging.info("Valutazione completata.")

results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(10, 6))
results_df.set_index('model')[['precision', 'recall', 'f1_score', 'accuracy']].plot(kind='bar')
plt.title('Confronto delle Prestazioni dei Modelli')
plt.xlabel('Modello')
plt.ylabel('Valore')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()