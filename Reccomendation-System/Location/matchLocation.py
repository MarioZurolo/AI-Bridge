import requests
import random
import os
import logging
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','ModuloFia')))
from Connector import get_db_connection
import sqlite3

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Key HERE Maps
API_KEY = "9ACYaFQNcHs-xlA9MY6pk148Z5MzEhko0I8zhqih1CE"

# Funzione per caricare province -> regione
def load_province_region_mapping():
    df = pd.read_csv("/Users/mariozurolo/AI-Bridge/Reccomendation-System/Location/province.csv", sep=";")  # Aggiungi sep=";"
    return dict(zip(df["provincia"], df["regione"]))


province_to_region = load_province_region_mapping()


query_alloggi = """
SELECT alloggio.id AS id_alloggio, alloggio.metratura, alloggio.max_persone, 
       ind.provincia, ind.latitudine, ind.longitudine
FROM alloggio
JOIN indirizzo ind ON alloggio.indirizzo = ind.id
WHERE ind.latitudine != 0 AND ind.longitudine != 0;
"""


# **Ottenere informazioni sul percorso**
def get_route_info(origin, destination, mode="car"):
    lat1, lon1 = origin
    lat2, lon2 = destination
    routing_url = (
        f"https://router.hereapi.com/v8/routes?transportMode={mode}"
        f"&origin={lat1},{lon1}&destination={lat2},{lon2}"
        f"&return=summary&apiKey={API_KEY}"
    )
    
    try:
        response = requests.get(routing_url, timeout=5)
        response.raise_for_status()
        data = response.json()

        print("Risposta API:", data)  # Puoi rimuovere questa linea dopo il debug

        if "routes" in data and len(data["routes"]) > 0:
            summary = data["routes"][0]["sections"][0]["summary"]
            distance_km = summary["length"] / 1000
            travel_time_min = summary["duration"] // 60
            return distance_km, travel_time_min
        else:
            logging.error("Errore: Nessuna rotta trovata nella risposta.")
            return float('inf'), float('inf')  # Restituisci valori infiniti in caso di errore
    except requests.RequestException as e:
        logging.error(f"Errore richiesta HERE API: {e}")
    return float('inf'), float('inf')

# **Generazione popolazione iniziale**
def generate_population(size, alloggi_df, lavori_df):
    print("Generazione popolazione in corso...")
    population = []
    for _ in range(size):
        solution = {row['id_annuncio']: random.choice(alloggi_df['id_alloggio'].tolist()) for _, row in lavori_df.iterrows()}
        population.append(solution)
    return population

# **Calcolo fitness**
def fitness(solution, alloggi_df, lavori_df):
    total_score = 0
    for job_id, alloggio_id in solution.items():
        job = lavori_df[lavori_df['id_annuncio'] == job_id].iloc[0]
        alloggio = alloggi_df[alloggi_df['id_alloggio'] == alloggio_id].iloc[0]

        job_coords = (job['latitudine'], job['longitudine'])
        alloggio_coords = (alloggio['latitudine'], alloggio['longitudine'])

        dist_km, time_min = get_route_info(job_coords, alloggio_coords)

        # Penalità e bonus
        total_score -= dist_km * 1  # Penalità distanza
        if dist_km < 50:
            total_score += 50  # Bonus vicinanza
        if time_min > 120:
            total_score -= (time_min - 120) * 0.5  # Penalità tempo
        if time_min < 30:
            total_score += (30 - time_min) * 5  # Bonus tempo breve
            
    return total_score

# **Selezione per torneo**
def tournament_selection(population, lavori_df, alloggi_df, tournament_size=3):
    selected = []
    for _ in range(len(population) // 2):
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=lambda sol: fitness(sol, alloggi_df, lavori_df), reverse=True)
        selected.append(tournament[0])
    return selected

# **Crossover a due punti**
def two_point_crossover(parent1, parent2):
    length = len(parent1)
    point1, point2 = sorted(random.sample(range(length), 2))
    child1, child2 = parent1.copy(), parent2.copy()
    for rifugiato in parent2:
        if point1 <= rifugiato < point2:
            child1[rifugiato] = parent2[rifugiato]
            child2[rifugiato] = parent1[rifugiato]
    return child1, child2

# **Mutazione**
def strong_mutation(solution, alloggi_df):
    rifugiato = random.choice(list(solution.keys()))
    solution[rifugiato] = random.choice(alloggi_df['id_alloggio'].tolist())
    return solution

# **Algoritmo genetico**
def genetic_algorithm(alloggi_df, lavori_df, generations=10, population_size=10):
    print("Algoritmo genetico in esecuzione...")
    population = generate_population(population_size, alloggi_df, lavori_df)
    print("Popolazione iniziale generata.")
    best_solution, best_score = None, float('-inf')

    for generation in range(generations):
        print(f"Generazione {generation}")
        selected = tournament_selection(population, lavori_df, alloggi_df)
        print("Selezione torneo completata.")
        new_generation = []
        
        while len(new_generation) < population_size:
            p1, p2 = random.sample(selected, 2)
            child1, child2 = two_point_crossover(p1, p2)
            child1 = strong_mutation(child1, alloggi_df)
            child2 = strong_mutation(child2, alloggi_df)
            new_generation.extend([child1, child2])
        
        population = new_generation
        
        # Pass the alloggi_df and lavori_df to fitness using lambda
        current_best = max(population, key=lambda sol: fitness(sol, alloggi_df, lavori_df))
        print("current_best:", current_best)
        current_best_score = fitness(current_best, alloggi_df, lavori_df)

        if current_best_score > best_score:
            best_solution = current_best
            best_score = current_best_score
        
        if generation % 10 == 0 or generation == generations - 1:
            logging.info(f"Generazione {generation}, miglior punteggio: {best_score}")

    return best_solution


def match_housing(job_recommendations, db_connection):
    # Forzare la conversione degli ID in interi
    try:
        recommended_job_ids = [int(job["id"]) for job in job_recommendations]  # Converte tutti gli ID in interi
    except ValueError as e:
        logging.error(f"Errore nella conversione degli ID: {e}")
        return {"error": "ID dei lavori non validi, impossibile eseguire la query."}
    
    # Verifica che tutti gli ID siano interi
    if not all(isinstance(id, int) for id in recommended_job_ids):
        logging.error(f"Gli ID dei lavori non sono tutti interi: {recommended_job_ids}")
        return {"error": "Tutti gli ID dei lavori devono essere numeri interi."}

    logging.info(f"Recommended job IDs: {recommended_job_ids}")
    
    # Verifica se ci sono ID da elaborare
    if not recommended_job_ids:
        logging.error("Nessun ID di lavoro raccomandato trovato.")
        return {}

    # Creazione della query con segnaposto dinamico
    placeholders = ", ".join(["?"] * len(recommended_job_ids))  # Crea il numero corretto di `?`
    query = f"""
        SELECT al.id, al.titolo, ind.provincia, ind.latitudine, ind.longitudine
        FROM lavoro al
        JOIN indirizzo ind ON al.indirizzo_id = ind.id
        WHERE al.id IN ({placeholders})
              AND ind.latitudine != 0
              AND ind.longitudine != 0;
    """
    logging.info(f"Query finale: {query}")

    # Creare un placeholder con gli ID veri
    placeholders = ", ".join(map(str, recommended_job_ids))  # Converte ogni ID in stringa
    query = f"""
    SELECT al.id AS id_annuncio, al.titolo, ind.provincia, ind.latitudine, ind.longitudine
    FROM lavoro al
    JOIN indirizzo ind ON al.indirizzo_id = ind.id
    WHERE al.id IN ({placeholders})
          AND ind.latitudine != 0
          AND ind.longitudine != 0;
    """

    logging.info(f"Query finale: {query}")
    lavori_df = pd.read_sql_query(query, db_connection)
    logging.info(lavori_df)
    # Se non ci sono risultati, logga un errore e ritorna un dizionario vuoto
    if lavori_df.empty:
        logging.error("Nessun annuncio di lavoro valido trovato tra le raccomandazioni.")
        return {}

    # Verifica che la colonna 'id_annuncio' sia presente
    if 'id_annuncio' not in lavori_df.columns:
        logging.error(f"Colonna 'id_annuncio' non trovata in lavori_df. Colonne disponibili: {lavori_df.columns}")
        return {"error": "'id_annuncio' non trovato nel DataFrame dei lavori"}
    
    for job_id in lavori_df['id_annuncio']:
        job = lavori_df[lavori_df['id_annuncio'] == job_id].iloc[0]
        logging.info(f"Accessing job ID: {job_id} -> {job}")

    # Caricamento alloggi dal database
    alloggi_df = pd.read_sql_query(query_alloggi, db_connection)
    logging.info(alloggi_df)

    logging.info(f"Colonne lavori_df: {lavori_df.columns}")
    logging.info(f"Colonne alloggi_df: {alloggi_df.columns}")
    # Filtraggio alloggi per provincia dei lavori consigliati
    job_provinces = set(lavori_df["provincia"])
    region_clusters = {province_to_region.get(prov, "Sconosciuto") for prov in job_provinces}

    alloggi_filtrati = alloggi_df[alloggi_df['provincia'].isin(job_provinces)]
    if alloggi_filtrati.empty:
        alloggi_filtrati = alloggi_df[alloggi_df['provincia'].map(province_to_region).isin(region_clusters)]
    logging.info(alloggi_filtrati)

    # Applica l'algoritmo genetico per trovare il miglior abbinamento
    best_allocation = genetic_algorithm(alloggi_filtrati, lavori_df)

    return best_allocation




