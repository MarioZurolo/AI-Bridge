import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np
import os
import logging
import datasets
import string
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# Scarica le stopwords solo una volta
nltk.download('stopwords')

# Stopwords in italiano
stopwords_it = set(stopwords.words('italian'))

# 📌 CONFIGURAZIONE LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_path = os.path.join(os.path.dirname(__file__), '..', 'Dat')

# 📌 CARICAMENTO DATASET PER IL FINE-TUNING
df_finetune = pd.read_csv(os.path.join(base_path, 'fineTuning', 'FineTuning.csv'), sep=';')
train_examples = [InputExample(texts=[row['Rifugiato'], row['Annuncio']], label=float(row['Label'])) for _, row in df_finetune.iterrows()]

# 📌 CARICAMENTO DEI DATI PER LA VALUTAZIONE

refugees_df = pd.read_csv(os.path.join(base_path, 'rifugiati.csv'))
lavori_df = pd.read_csv(os.path.join(base_path, 'Lavori_Uniti.csv'), sep=';')
ground_truth_df = pd.read_csv(os.path.join(base_path, 'Ground_True.csv'), sep=';')

# 📌 CREAZIONE DELLA LISTA PER IL FINE-TUNING
train_examples = [
    InputExample(texts=[row['Rifugiato'], row['Annuncio']], label=float(row['Label']))
    for _, row in df_finetune.iterrows()
]

def evaluate_tfidf_baseline():
    """Valuta il modello usando TF-IDF + Similarità Coseno."""
    logging.info("Calcolo delle similarità usando TF-IDF...")
    
    # Unione dei testi per la rappresentazione vettoriale
    refugees_texts = refugees_df['Testo_Skill'] + " " + refugees_df['Testo_Titolo'] + " " + refugees_df['Testo_Lingue']
    jobs_texts = lavori_df['Testo_Titolo_Annuncio'] + " " + lavori_df['Testo_Posizione'] + " " + lavori_df['Testo_Info']

    # Creazione della matrice TF-IDF
    vectorizer = TfidfVectorizer()
    refugees_tfidf = vectorizer.fit_transform(refugees_texts)
    jobs_tfidf = vectorizer.transform(jobs_texts)

    # Calcolo delle similarità coseno
    similarity_matrix = cosine_similarity(refugees_tfidf, jobs_tfidf)

    true_labels, predicted_labels = [], []
    
    for i, similarities in enumerate(similarity_matrix):
        top_indices = np.argsort(similarities)[-3:][::-1]  # Prendi i 3 annunci più simili
        refugee_name = refugees_df.iloc[i]['Email']
        
        true_match = any(lavori_df.iloc[idx]['ID'] in ground_truth_dict.get(refugee_name, set()) for idx in top_indices)
        true_labels.append(1 if true_match else 0)

        max_similarity = max(similarities[idx] for idx in top_indices)
        threshold = np.percentile(similarity_matrix, 80)
        predicted_labels.append(1 if max_similarity >= threshold else 0)

    return {
        "model": "TF-IDF + Coseno",
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
        "accuracy": accuracy_score(true_labels, predicted_labels)
    }


# 📌 FINE-TUNING DEL MODELLO
def fine_tune_model(model_name, train_examples, epochs=2, batch_size=32):
    logging.info(f"Avvio fine-tuning su {model_name}...")

    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)  # Usa una loss più robusta

    warmup_steps = int(0.1 * len(train_dataloader) * epochs)

    model.fit(train_objectives=[(train_dataloader, train_loss)], 
              epochs=epochs, 
              warmup_steps=warmup_steps, 
              output_path="modello_finetuned")
    
    model.save("modello_finetuned")
    
    logging.info("Fine-tuning completato e modello salvato.")
    return model

# 📌 ESEGUE IL FINE-TUNING
finetuned_model = fine_tune_model("multi-qa-mpnet-base-dot-v1", train_examples)


# 📌 PULIZIA DEL TESTO
def clean_text(text):
    """Pulisce il testo rimuovendo punteggiatura e stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip().translate(str.maketrans("", "", string.punctuation))
    return ' '.join(word for word in text.split() if word not in stopwords_it)


# Pre-elaborazione dei testi
columns_to_clean = {'Skill': 'Testo_Skill', 'Titolo di studio': 'Testo_Titolo', 'Lingue parlate': 'Testo_Lingue'}
columns_to_clean_jobs = {'Titolo': 'Testo_Titolo_Annuncio', 'Posizione lavorativa': 'Testo_Posizione', 'Info utili': 'Testo_Info'}

for col, new_col in columns_to_clean.items():
    refugees_df[new_col] = refugees_df[col].fillna('').apply(clean_text)

for col, new_col in columns_to_clean_jobs.items():
    lavori_df[new_col] = lavori_df[col].fillna('').apply(clean_text)

# 📌 FUNZIONE DI VALUTAZIONE
def evaluate_model(model_name):
    """Valuta il modello calcolando precision, recall, F1 e accuracy."""
    model = SentenceTransformer(model_name)
    logging.info(f"Generazione embeddings per {model_name}...")

    refugees_embeddings = {key: model.encode(refugees_df[col].tolist(), batch_size=32, convert_to_numpy=True) for key, col in columns_to_clean.items()}
    lavori_embeddings = {key: model.encode(lavori_df[col].tolist(), batch_size=32, convert_to_numpy=True) for key, col in columns_to_clean_jobs.items()}

    logging.info(f"Calcolo similarità per {model_name}...")
    weights = {'Skill': 0.1, 'Titolo di studio': 0.05, 'Lingue parlate': 0.05}
    similarity_matrix = sum(weights[key] * cosine_similarity(refugees_embeddings[key], lavori_embeddings['Titolo']) for key in weights)
    similarity_matrix += sum(w * cosine_similarity(refugees_embeddings['Skill'], lavori_embeddings[k]) for k, w in {'Posizione lavorativa': 0.26, 'Info utili': 0.54}.items())

    # 📊 Visualizzazione della distribuzione dei punteggi di similarità
    import seaborn as sns

    scores_positive = [cosine_similarity([refugees_embeddings['Skill'][i]], [lavori_embeddings['Titolo'][i]])[0,0] 
                    for i in range(len(refugees_df))]
    scores_negative = [cosine_similarity([refugees_embeddings['Skill'][i]], [lavori_embeddings['Titolo'][i-1]])[0,0]  
                    for i in range(len(refugees_df))]

    sns.histplot(scores_positive, color="green", label="Positivi", kde=True)
    sns.histplot(scores_negative, color="red", label="Negativi", kde=True)
    plt.legend()
    plt.title("Distribuzione della Similarità tra Skill e Titolo Annuncio")
    plt.show()

    true_labels, predicted_labels = [], []

    for i, similarities in enumerate(similarity_matrix):
        top_indices = np.argsort(similarities)[-3:][::-1]
        refugee_name = refugees_df.iloc[i]['Email']
        true_match = any(lavori_df.iloc[idx]['ID'] in ground_truth_dict.get(refugee_name, set()) for idx in top_indices)
        true_labels.append(1 if true_match else 0)

        max_similarity = max(similarities[idx] for idx in top_indices)
        threshold = np.percentile(similarity_matrix, 80)
        predicted_labels.append(1 if max_similarity >= threshold else 0)

    return {
        "model": model_name,
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
        "accuracy": accuracy_score(true_labels, predicted_labels)
    }


# 📌 CREAZIONE DIZIONARIO PER GROUND TRUTH
ground_truth_dict = defaultdict(set)
for _, row in ground_truth_df.iterrows():
    if row['Match'] == 1:
        ground_truth_dict[row['Rifugiati']].add(row['Annunci'])

# 📌 LISTA MODELLI DA VALUTARE
models = ['distilbert-base-nli-mean-tokens', 'all-MiniLM-L6-v2', 'multi-qa-mpnet-base-dot-v1', 'modello_finetuned']
results = [evaluate_model(model) for model in models]

# Aggiungi il confronto con TF-IDF
results.append(evaluate_tfidf_baseline())

# 📌 RISULTATI DELLA VALUTAZIONE
results_df = pd.DataFrame(results)
print(results_df)

# 📌 VISUALIZZAZIONE DEI RISULTATI
plt.figure(figsize=(10, 6))
results_df.set_index('model')[['precision', 'recall', 'f1_score', 'accuracy']].plot(kind='bar')
plt.title('Confronto delle Prestazioni dei Modelli')
plt.xlabel('Modello')
plt.ylabel('Valore')
plt.xticks(rotation=45)
plt.legend(loc='best')
ax = results_df.set_index('model')[['precision', 'recall', 'f1_score', 'accuracy']].plot(kind='bar', figsize=(10, 6))
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
plt.show()
