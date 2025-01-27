import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento modello
model = SentenceTransformer('all-MiniLM-L6-v2')

# Caricamento dati
refugees_df = pd.read_csv('C:/Users/utente/Bridge-AI/AI-Bridge/Reccomendation-System/Dat/rifugiati.csv')
lavori_df = pd.read_csv('C:/Users/utente/Bridge-AI/AI-Bridge/Reccomendation-System/Dat/Annunci_di_Lavoro.csv')

# Gestione valori mancanti
refugees_df.fillna('', inplace=True)
lavori_df.fillna('', inplace=True)

# Generazione profili e embeddings
refugee_profiles = [
    f"{r['Skill']}" for _, r in refugees_df.iterrows()
]
refugee_embeddings = model.encode(refugee_profiles)

lavori_annunci = [
    f"{j['Info utili']}" for _, j in lavori_df.iterrows()
]
lavori_embeddings = model.encode(lavori_annunci)

# Calcola la similarità coseno
similarity_matrix = util.cos_sim(refugee_embeddings, lavori_embeddings).numpy()

# Ordina le raccomandazioni per ogni rifugiato
recommendations = np.argsort(similarity_matrix, axis=1)[:, ::-1]

# Stampa le raccomandazioni in formato leggibile con valori di similarità e skill del rifugiato
for i, refugee in enumerate(refugees_df['Nome']):
    print(f"\nRaccomandazioni per {refugee}:")
    print(f"  Skill del rifugiato: {refugees_df.iloc[i]['Skill']}")
    for rank, job_idx in enumerate(recommendations[i, :3]):  # Mostra le prime 3 raccomandazioni
        job = lavori_df.iloc[job_idx]
        similarity_score = similarity_matrix[i, job_idx]
        print(f"  {rank + 1}. {job['Titolo']} presso {job['Nome azienda']} ({job['Posizione lavorativa']}) - Similarità: {similarity_score:.4f}")


similarity_scores = similarity_matrix.flatten()
plt.figure(figsize=(10, 6))
sns.histplot(similarity_scores, bins=20, kde=True, color='blue')
plt.title('Distribuzione dei Punteggi di Similarità', fontsize=14)
plt.xlabel('Punteggio di Similarità', fontsize=12)
plt.ylabel('Frequenza', fontsize=12)
plt.show()
