import requests
import random
import os
import logging
import pandas as pd

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Key HERE Maps
API_KEY = "Zd69mVUjvr-570Ex5u1Nu52-zvYyUzAeD1jKaxLcouk"

# Dati di esempio
risultati = [
    {"Rifugiato": "fcarlson@example.net", "ID Annuncio": 201, "Titolo Annuncio": "Insegnante di supporto per certificazioni linguistiche", "Indirizzo": "Milano, MI, Via Monte Rosa, 15, 20149", "Somiglianza": 8.912886663821075, "latitudine": 45.4719, "longitudine": 9.15311},
    {"Rifugiato": "meyersjason@example.org", "ID Annuncio": 129, "Titolo Annuncio": "Creatore di contenuti multilingue per aziende", "Indirizzo": "Palermo, PA, Via Flotta, 56, 90139", "Somiglianza": 9.038126547272268, "latitudine": 38.11197, "longitudine": 13.35061}
]

alloggi = [
    {"ID": 1, "Metratura": 169, "Max persone": 8, "Indirizzo": "Stretto Milena, 12, 36100, Anconetta (VI)", "latitudine": 45.5661, "longitudine": 11.5686},
    {"ID": 2, "Metratura": 166, "Max persone": 8, "Indirizzo": "Vicolo Letizia, 66 Piano 7, 87022, Sant'Angelo Di Cetraro (CS)", "latitudine": 39.55521, "longitudine": 15.95348},
    {"ID": 3, "Metratura": 184, "Max persone": 7, "Indirizzo": "Via Mastroianni, 8 Appartamento 34, 22045, Lambrugo (CO)", "latitudine": 45.75779, "longitudine": 9.24152},
    {"ID": 4, "Metratura": 65, "Max persone": 9, "Indirizzo": "Strada Persico, 3, 36040, Valdastico (VI)", "latitudine": 45.86633, "longitudine": 11.36133},
    {"ID": 5, "Metratura": 32, "Max persone": 7, "Indirizzo": "Canale Comencini, 38048, Sover (TN)", "latitudine": 46.22238, "longitudine": 11.31401}
]

# Converti i dati di esempio in DataFrame
lavori_df = pd.DataFrame(risultati)
alloggi_df = pd.DataFrame(alloggi)

route_cache = {}  # Dizionario per memorizzare le risposte API

# Funzione per ottenere informazioni sul percorso
def get_route_info(origin, destination, mode="car"):
    key = (origin, destination, mode)
    if key in route_cache:
        return route_cache[key]  

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
        logging.info(f"API Response: {data}")

        if "routes" in data and len(data["routes"]) > 0:
            summary = data["routes"][0]["sections"][0]["summary"]
            distance_km = summary["length"] / 1000
            travel_time_min = summary["duration"] // 60
            transport_bonus = 1 if mode == "publicTransport" else 0
            route_cache[key] = (distance_km, travel_time_min)
            return distance_km, travel_time_min, transport_bonus

    except requests.RequestException as e:
        logging.error(f"Errore richiesta HERE API: {e}")

    return float('inf'), float('inf') 

# Funzione per generare la popolazione iniziale
def generate_population(size, alloggi_df, lavori_df):
    population = []
    for _ in range(size):
        solution = {rifugiato['ID Annuncio']: random.choice(alloggi_df['ID'].tolist()) for rifugiato in risultati}
        population.append(solution)
    return population

# Funzione per calcolare la fitness di una soluzione
def fitness(solution):
    total_score = 0
    distance_penalty_weight = 1  # Fattore di penalità per la distanza
    proximity_bonus_weight = 50  # Fattore di premio per la vicinanza (se la distanza è bassa)
    time_penalty_weight = 0.5   # Fattore di penalità per il tempo
    time_bonus_weight = 5       # Fattore di premio per il tempo (se il tempo è breve)

    for rifugiato_id, alloggio_id in solution.items():
        job = lavori_df[lavori_df['id_annuncio'] == rifugiato_id].iloc[0]
        job_coords = (job['latitudine'], job['longitudine'])

        alloggio = alloggi_df[alloggi_df['id_alloggio'] == alloggio_id].iloc[0]
        alloggio_coords = (alloggio['latitudine'], alloggio['longitudine'])

        dist_km, time_min = get_route_info(job_coords, alloggio_coords)

        
        # Penalizza la distanza
        total_score -= dist_km * distance_penalty_weight

        # Premia la vicinanza
        if dist_km < 50:
            total_score += proximity_bonus_weight

        # Penalizza il tempo (se il tempo di viaggio è elevato)
        if time_min > 120:
            total_score -= (time_min - 120) * time_penalty_weight

        # Premia il tempo (se il tempo di viaggio è inferiore a una soglia, es. 30 minuti)
        if time_min < 30:
            total_score += (30 - time_min) * time_bonus_weight
    
    return total_score

# Funzione di selezione (Selezione per torneo)
def tournament_selection(population, tournament_size=3):
    selected = []
    for _ in range(len(population) // 2):
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=fitness, reverse=True)
        selected.append(tournament[0])
    return selected

# Funzione di crossover a due punti
def two_point_crossover(parent1, parent2):
    length = len(parent1)
    point1, point2 = sorted(random.sample(range(length), 2))
    child1 = parent1.copy()
    child2 = parent2.copy()
    for rifugiato in parent2:
        if point1 <= rifugiato < point2:
            child1[rifugiato] = parent2[rifugiato]
            child2[rifugiato] = parent1[rifugiato]
    return child1, child2

# Funzione di mutazione forte
def strong_mutation(solution):
    rifugiato = random.choice(list(solution.keys()))
    solution[rifugiato] = random.choice(alloggi_df['ID'].tolist())
    return solution

def genetic_algorithm(generations=10, population_size=10, early_stop_threshold=5):
    population = generate_population(population_size, alloggi_df, lavori_df)
    best_solution = None
    best_score = float('inf')  
    no_improve_count = 0  # Contatore per iterazioni senza miglioramenti

    for generation in range(generations):
        selected = tournament_selection(population)
        new_generation = []

        while len(new_generation) < population_size:
            p1, p2 = random.sample(selected, 2)
            child1, child2 = two_point_crossover(p1, p2)
            child1 = strong_mutation(child1)
            child2 = strong_mutation(child2)
            new_generation.extend([child1, child2])

        population = new_generation
        current_best = max(population, key=lambda x: fitness(x))  # Se fitness va massimizzato
        current_best_score = fitness(current_best)

        if current_best_score < best_score: 
            best_solution = current_best
            best_score = current_best_score
            no_improve_count = 0  # Reset contatore
        else:
            no_improve_count += 1  # Incrementa contatore se non ci sono miglioramenti

        # Stampa il miglior risultato ogni 10 generazioni
        if generation % 10 == 0 or generation == generations - 1:
            logging.info(f"Generazione {generation}, miglior punteggio: {best_score}")
        

        # **Se per 5 generazioni non migliora, interrompi**
        if no_improve_count >= early_stop_threshold:
            logging.info(f"Stop anticipato alla generazione {generation}: nessun miglioramento nelle ultime {early_stop_threshold} generazioni.")
            break



    return best_solution

# Funzione per stampare tutte le distanze e i tempi di viaggio
def print_all_distances_and_times():
    for _, job in lavori_df.iterrows():
        job_address = job['Indirizzo']
        job_coords = (job['latitudine'], job['longitudine'])
        for _, alloggio in alloggi_df.iterrows():
            alloggio_address = alloggio['Indirizzo']
            alloggio_coords = (alloggio['latitudine'], alloggio['longitudine'])
            dist_km, time_min, _ = get_route_info(job_coords, alloggio_coords)
            print(f"Da {job_address} a {alloggio_address}: {dist_km:.2f} km, {time_min} min")

# Esegui l'algoritmo genetico
try:
    best_allocation = genetic_algorithm()
    print(f"Migliore allocazione: {best_allocation}")
except KeyError as e:
    print(e)

# Stampa tutte le distanze e i tempi di viaggio
print_all_distances_and_times()

# Salvataggio dei risultati finali
final_results = []
for rifugiato_id, alloggio_id in best_allocation.items():
    lavoro_info = lavori_df[lavori_df['ID Annuncio'] == rifugiato_id].iloc[0]
    alloggio_info = alloggi_df[alloggi_df['ID'] == alloggio_id].iloc[0]
    dist_km, time_min, _ = get_route_info((lavoro_info['latitudine'], lavoro_info['longitudine']), (alloggio_info['latitudine'], alloggio_info['longitudine']))
    final_results.append({
        "Rifugiato": lavoro_info['Rifugiato'],
        "Lavoro": lavoro_info['Titolo Annuncio'],
        "Indirizzo Lavoro": lavoro_info['Indirizzo'],
        "Alloggio": alloggio_info['Indirizzo'],
        "Distanza (km)": dist_km,
        "Tempo (min)": time_min
    })

final_results_df = pd.DataFrame(final_results)
output_path_final = os.path.join(os.path.dirname(__file__), 'risultati_finali.csv')
final_results_df.to_csv(output_path_final, index=False)
logging.info(f"Risultati finali salvati in {output_path_final}")