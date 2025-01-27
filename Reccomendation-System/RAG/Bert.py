import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento modello BERT-based fine-tuned
# Puoi usare distilbert-base-nli-mean-tokens che è fine-tuned su un task di similarità semantica
model = SentenceTransformer('distilbert-base-nli-mean-tokens')  # Alternativa: 'bert-base-nli-mean-tokens'

# Caricamento dati
refugees_df = pd.read_csv('C:/Users/utente/Bridge-AI/AI-Bridge/Reccomendation-System/Dat/rifugiati.csv')
lavori_df = pd.read_csv('C:/Users/utente/Bridge-AI/AI-Bridge/Reccomendation-System/Dat/Annunci_di_Lavoro.csv')

# Gestione valori mancanti
refugees_df.fillna('', inplace=True)
lavori_df.fillna('', inplace=True)

# Calcolo degli embeddings con più campi
refugee_embeddings = []
for _, r in refugees_df.iterrows():
    # Calcolo degli embeddings per i campi rilevanti
    skill_embedding = model.encode(r['Skill'])
    titolo_embedding = model.encode(r['Titolo di studio'])
    
    # Media ponderata degli embeddings (puoi regolare i pesi)
    combined_embedding = np.mean([skill_embedding * 0.6, titolo_embedding * 0.4], axis=0)
    refugee_embeddings.append(combined_embedding)

refugee_embeddings = np.array(refugee_embeddings)

# Embeddings per gli annunci di lavoro con più campi
lavori_embeddings = []
for _, j in lavori_df.iterrows():
    # Combina "Info utili", "Titolo" e "Posizione lavorativa"
    info_embedding = model.encode(j['Info utili'])
    titolo_lavoro_embedding = model.encode(j['Titolo'])
    posizione_embedding = model.encode(j['Posizione lavorativa'])
    
    # Media ponderata degli embeddings
    combined_embedding = np.mean(
        [info_embedding * 0.5, titolo_lavoro_embedding * 0.3, posizione_embedding * 0.2], axis=0
    )
    lavori_embeddings.append(combined_embedding)

lavori_embeddings = np.array(lavori_embeddings)

# Calcola la similarità coseno
similarity_matrix = util.cos_sim(refugee_embeddings, lavori_embeddings).numpy()

# Raccomandazioni ordinate
recommendations = np.argsort(similarity_matrix, axis=1)[:, ::-1]

# Output migliorato e salvataggio in CSV
recommendations_list = []
for i, refugee in enumerate(refugees_df['Nome']):
    refugee_skill = refugees_df.iloc[i]['Skill']
    for rank, job_idx in enumerate(recommendations[i, :3]):  # Mostra le prime 3 raccomandazioni
        job = lavori_df.iloc[job_idx]
        similarity_score = similarity_matrix[i, job_idx]
        
        # Salva i dati per esportazione
        recommendations_list.append({
            'Nome rifugiato': refugee,
            'Skill rifugiato': refugee_skill,
            'Titolo lavoro': job['Titolo'],
            'Posizione lavorativa': job['Posizione lavorativa'],
            'Info utili': job['Info utili'],
            'Punteggio similarità': similarity_score
        })

# Salvataggio in un CSV
recommendations_df = pd.DataFrame(recommendations_list)
recommendations_df.to_csv('raccomandazioni_lavoro.csv', index=False)

# Distribuzione dei punteggi di similarità
similarity_scores = similarity_matrix.flatten()
plt.figure(figsize=(10, 6))
sns.histplot(similarity_scores, bins=20, kde=True, color='blue')
plt.title('Distribuzione dei Punteggi di Similarità', fontsize=14)
plt.xlabel('Punteggio di Similarità', fontsize=12)
plt.ylabel('Frequenza', fontsize=12)
plt.show()
