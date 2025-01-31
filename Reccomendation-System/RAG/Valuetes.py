import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Caricamento dati
base_path = os.path.join(os.path.dirname(__file__), '..', 'Dat')
refugees_df = pd.read_csv(os.path.join(base_path, 'rifugiati.csv'))
lavori_df = pd.read_csv(os.path.join(base_path, 'Annunci_di_lavoro.csv'), sep=';')
ground_truth_df = pd.read_csv(os.path.join(base_path, 'Ground_True.csv'), sep=';')

# Verifica colonne necessarie
ground_truth_dict = ground_truth_df.set_index('Rifugiati')['Annunci'].to_dict()

# Stopwords manuali
stopwords = set("""
    a adesso ai al alla allo allora altre altri altro anche ancora avere aveva avevano ben buono che chi cinque comprare
    con consecutivi consecutivo cosa cui da del della dello dentro deve devo di dove due e è ecco fare fine fino fra gente giu
    ha hai hanno ho il indietro invece io la lavoro le lei lo loro lui lungo ma me meglio molta molti molto nei nella no noi
    nome nostro nove nuovi nuovo o oltre ora otto peggio pero persone piu poco primo promesso qua quarto quasi quattro quello
    questo qui quindi quinto rispetto sara secondo sei sembra sembrava senza sette sia siamo siete solo sono sopra soprattutto
    sotto stati stato stesso su subito sul sulla tanto te tempo terzo tra tre triplo ultimo un una uno va vai voi volte vostro /
""".split())

def remove_stopwords(text):
    return ' '.join(word for word in text.split() if word.lower() not in stopwords)

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
    weights = {'Skill': 0.4, 'Titolo di studio': 0.05, 'Lingue parlate': 0.05}
    similarity_matrix = sum(
        weights[key] * cosine_similarity(refugees_embeddings[key], lavori_embeddings['Titolo Annuncio'])
        for key in weights
    )
    similarity_matrix += sum(
        w * cosine_similarity(refugees_embeddings['Skill'], lavori_embeddings[k])
        for k, w in {'Posizione Lavorativa': 0.3, 'Info Utili': 0.3}.items()
    )

    global_threshold = np.percentile(similarity_matrix, 75)
    matches, true_labels, predicted_labels = [], [], []
    
    for i, similarities in enumerate(similarity_matrix):
        top_indices = np.argsort(similarities)[-3:][::-1]
        refugee_name = refugees_df.iloc[i]['Email']
        ground_truth_id = ground_truth_dict.get(refugee_name)
        found_match = False

        for idx in top_indices:
            if similarities[idx] >= global_threshold:
                found_match = True
                matches.append({
                    "Rifugiato": refugee_name,
                    "ID Annuncio": lavori_df.iloc[idx]["ID"],
                    "Titolo Annuncio": lavori_df.iloc[idx]["Titolo Annuncio"],
                    "Somiglianza": similarities[idx]
                })
        
        if not found_match:
            best_idx = np.argmax(similarities)
            matches.append({
                "Rifugiato": refugee_name,
                "ID Annuncio": lavori_df.iloc[best_idx]["ID"],
                "Titolo Annuncio": lavori_df.iloc[best_idx]["Titolo Annuncio"],
                "Somiglianza": similarities[best_idx]
            })

        true_labels.append(1 if ground_truth_id else 0)
        predicted_labels.append(1 if ground_truth_id and any(lavori_df.iloc[idx]['ID'] == ground_truth_id for idx in top_indices) else 0)

    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))

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
