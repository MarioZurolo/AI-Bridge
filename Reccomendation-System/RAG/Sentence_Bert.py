import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lista manuale di stop words italiane
manual_italian_stopwords = [
    "a", "adesso", "ai", "al", "alla", "allo", "allora", "altre", "altri", "altro", 
    "anche", "ancora", "avere", "aveva", "avevano", "ben", "buono", "che", "chi", 
    "cinque", "comprare", "con", "consecutivi", "consecutivo", "cosa", "cui", "da", 
    "del", "della", "dello", "dentro", "deve", "devo", "di", "dove", "due", "e", 
    "è", "ecco", "fare", "fine", "fino", "fra", "gente", "giu", "ha", "hai", "hanno", 
    "ho", "il", "indietro", "invece", "io", "la", "lavoro", "le", "lei", "lo", 
    "loro", "lui", "lungo", "ma", "me", "meglio", "molta", "molti", "molto", 
    "nei", "nella", "no", "noi", "nome", "nostro", "nove", "nuovi", "nuovo", 
    "o", "oltre", "ora", "otto", "peggio", "pero", "persone", "piu", "poco", 
    "primo", "promesso", "qua", "quarto", "quasi", "quattro", "quello", "questo", 
    "qui", "quindi", "quinto", "rispetto", "sara", "secondo", "sei", "sembra", 
    "sembrava", "senza", "sette", "sia", "siamo", "siete", "solo", "sono", "sopra", 
    "soprattutto", "sotto", "stati", "stato", "stesso", "su", "subito", "sul", 
    "sulla", "tanto", "te", "tempo", "terzo", "tra", "tre", "triplo", "ultimo", 
    "un", "una", "uno", "va", "vai", "voi", "volte", "vostro", "/"
]

# Funzione per rimuovere le stopwords italiane
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in manual_italian_stopwords])

# Caricamento dati
base_path = os.path.join(os.path.dirname(__file__), '..', 'Dat')
refugees_df = pd.read_csv(os.path.join(base_path, 'rifugiati.csv'))
lavori_df = pd.read_csv(os.path.join(base_path, 'Annunci_di_lavoro.csv'), sep=';')

# Controllo colonne mancanti
required_cols_refugees = ['Skill', 'Lingue parlate', 'Titolo di studio', 'Nome']
required_cols_jobs = ['Info Utili', 'Titolo Annuncio', 'Posizione Lavorativa', 'ID']
assert all(col in refugees_df.columns for col in required_cols_refugees), "Mancano colonne nei dati dei rifugiati"
assert all(col in lavori_df.columns for col in required_cols_jobs), "Mancano colonne nei dati degli annunci"

# Pre-processamento: separiamo skill, titolo, lingue, etc.
refugees_df['Testo_Skill'] = refugees_df['Skill'].fillna('').apply(remove_stopwords)
refugees_df['Testo_Titolo'] = refugees_df['Titolo di studio'].fillna('').apply(remove_stopwords)
refugees_df['Testo_Lingue'] = refugees_df['Lingue parlate'].fillna('').apply(remove_stopwords)

lavori_df['Testo_Titolo_Annuncio'] = lavori_df['Titolo Annuncio'].fillna('').apply(remove_stopwords)
lavori_df['Testo_Posizione'] = lavori_df['Posizione Lavorativa'].fillna('').apply(remove_stopwords)
lavori_df['Testo_Info'] = lavori_df['Info Utili'].fillna('').apply(remove_stopwords)

# Modello Sentence-BERT per generare embeddings contestuali
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Generazione degli embeddings separati
logging.info("Generazione degli embeddings separati...")

# Generazione embedding per skill, titolo, lingue e altre info
rifugiati_skill_embeddings = model.encode(refugees_df['Testo_Skill'].tolist(), batch_size=32, convert_to_numpy=True)
lavori_titolo_embeddings = model.encode(lavori_df['Testo_Titolo_Annuncio'].tolist(), batch_size=32, convert_to_numpy=True)

rifugiati_titolo_embeddings = model.encode(refugees_df['Testo_Titolo'].tolist(), batch_size=32, convert_to_numpy=True)
lavori_posizione_embeddings = model.encode(lavori_df['Testo_Posizione'].tolist(), batch_size=32, convert_to_numpy=True)

rifugiati_lingue_embeddings = model.encode(refugees_df['Testo_Lingue'].tolist(), batch_size=32, convert_to_numpy=True)
lavori_info_embeddings = model.encode(lavori_df['Testo_Info'].tolist(), batch_size=32, convert_to_numpy=True)

# Calcolo delle similarità separato
logging.info("Calcolo delle similarità separato...")

# Skill vs Titolo Annuncio
similarity_skill_titolo = cosine_similarity(rifugiati_skill_embeddings, lavori_titolo_embeddings)

# Skill vs Posizione Lavorativa
similarity_skill_posizione = cosine_similarity(rifugiati_skill_embeddings, lavori_posizione_embeddings)

# Skill vs Info Utili
similarity_skill_info = cosine_similarity(rifugiati_skill_embeddings, lavori_info_embeddings)

# Titolo vs Titolo Annuncio (opzionale)
similarity_titolo = cosine_similarity(rifugiati_titolo_embeddings, lavori_titolo_embeddings)

# Lingue vs Info Utili
similarity_lingue_info = cosine_similarity(rifugiati_lingue_embeddings, lavori_info_embeddings)

# Aggregazione delle similarità con pesi personalizzati
agg_similarities = (similarity_skill_titolo * 0.5 + 
                    similarity_skill_posizione * 0.3 + 
                    similarity_skill_info * 0.3 +
                    similarity_titolo * 0.05 +  # Aggiunta la similarità tra titolo e titolo annuncio
                    similarity_lingue_info * 0.05)  # Aggiunta la similarità tra lingue e info utili

# Threshold dinamico per la similarità
logging.info("Calcolo del threshold dinamico...")
global_threshold = np.percentile(agg_similarities, 75)  # Top 25% dei valori di similarità

# Identifichiamo i tre lavori più simili per ogni rifugiato con fallback
matches = []
for i, similarities in enumerate(agg_similarities):
    top_indices = np.argsort(similarities)[-3:][::-1]  # Indici dei tre lavori più simili
    refugee_name = refugees_df.iloc[i]["Email"]
    
    found_match = False  # Flag per controllare se troviamo almeno un match
    for idx in top_indices:
        similarity_score = similarities[idx]
        if similarity_score >= global_threshold:  # Usa il threshold dinamico
            found_match = True
            matches.append({
                "Rifugiato": refugee_name,
                "ID Annuncio": lavori_df.iloc[idx]["ID"],
                "Titolo Annuncio": lavori_df.iloc[idx]["Titolo Annuncio"],
                "Somiglianza": similarity_score
            })
    
    # Fallback: se nessun match supera il threshold, scegli il miglior match disponibile
    if not found_match and len(similarities) > 0:
        best_idx = np.argmax(similarities)
        matches.append({
            "Rifugiato": refugee_name,
            "ID Annuncio": lavori_df.iloc[best_idx]["ID"],
            "Titolo Annuncio": lavori_df.iloc[best_idx]["Titolo Annuncio"],
            "Somiglianza": similarities[best_idx]
        })

# Creiamo un DataFrame per visualizzare i risultati
logging.info("Creazione del DataFrame dei risultati...")
matches_df = pd.DataFrame(matches)

# Ordiniamo i risultati per rifugiato e somiglianza decrescente
matches_df = matches_df.sort_values(by=["Rifugiato", "Somiglianza"], ascending=[True, False])

# Salviamo i risultati in un file CSV
output_path = os.path.join(base_path, 'risultati_finali.csv')
matches_df.to_csv(output_path, index=False)
logging.info(f"Risultati salvati in {output_path}")

# Visualizzazione opzionale del riepilogo dei risultati
# Logging dei risultati intermedi
logging.info(f"Embeddings generati per rifugiati: {rifugiati_skill_embeddings.shape}")
logging.info(f"Embeddings generati per lavori: {lavori_titolo_embeddings.shape}")
logging.info(f"Threshold dinamico: {global_threshold}")
logging.info(f"Top 3 lavori per rifugiato: {matches_df[['Rifugiato', 'Titolo Annuncio', 'Somiglianza']].head(10)}")

# Caricamento del ground truth
ground_truth_df = pd.read_csv(os.path.join(base_path, 'Ground_True.csv'), sep=';')

# Verifica se le colonne necessarie esistono nel file di ground truth
assert 'Rifugiati' in ground_truth_df.columns, "Manca la colonna 'Rifugiati' nel ground truth"
assert 'Annunci' in ground_truth_df.columns, "Manca la colonna 'Annunci' nel ground truth"

# Creazione di un dizionario per il ground truth
ground_truth_dict = dict(zip(ground_truth_df['Rifugiati'], ground_truth_df['Annunci']))

# Calcolo dell'accuratezza
correct_matches = 0
total_refugees = 0

# Verifica per ogni rifugiato
for i, similarities in enumerate(agg_similarities):
    refugee_email = refugees_df.iloc[i]["Email"]
    
    # Otteniamo le top 3 posizioni
    top_indices = np.argsort(similarities)[-3:][::-1]  # I tre lavori più simili
    
    # Verifica se l'ID del lavoro del ground truth è tra i top 3
    ground_truth_id = ground_truth_dict.get(refugee_email)
    
    if ground_truth_id and any(lavori_df.iloc[idx]["ID"] == ground_truth_id for idx in top_indices):
        correct_matches += 1
    
    total_refugees += 1

# Calcolare la precisione
accuracy = correct_matches / total_refugees if total_refugees > 0 else 0

# Stampa dei risultati
logging.info(f"Accuratezza del sistema: {accuracy * 100:.2f}%")