import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
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

# Generazione embeddings separati per ogni campo (Skill, Titolo di studio)
refugee_embeddings = []
for _, r in refugees_df.iterrows():
    skill_embedding = model.encode(r['Skill'])
    titolo_embedding = model.encode(r['Titolo di studio'])
    
    # Media degli embeddings
    combined_embedding = np.mean([skill_embedding, titolo_embedding], axis=0)
    refugee_embeddings.append(combined_embedding)

# Converte la lista in un array NumPy
refugee_embeddings = np.array(refugee_embeddings)

# Generazione degli embeddings per gli annunci di lavoro
lavori_embeddings = []
for _, j in lavori_df.iterrows():
    job_embedding = model.encode(f"{j['Titolo']} {j['Posizione lavorativa']} {j['Info utili']}")
    lavori_embeddings.append(job_embedding)

# Converte la lista in un array NumPy
lavori_embeddings = np.array(lavori_embeddings)

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

# Distribuzione dei punteggi di similarità
similarity_scores = similarity_matrix.flatten()
plt.figure(figsize=(10, 6))
sns.histplot(similarity_scores, bins=20, kde=True, color='blue')
plt.title('Distribuzione dei Punteggi di Similarità', fontsize=14)
plt.xlabel('Punteggio di Similarità', fontsize=12)
plt.ylabel('Frequenza', fontsize=12)
plt.show()
