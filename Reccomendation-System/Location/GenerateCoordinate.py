import pandas as pd
import requests
import time
import os

# Sostituisci con la tua chiave API
API_KEY = "Zd69mVUjvr-570Ex5u1Nu52-zvYyUzAeD1jKaxLcouk"

def get_coordinates(address):
    geocode_url = f"https://geocode.search.hereapi.com/v1/geocode?q={address}&apiKey={API_KEY}"
    try:
        response = requests.get(geocode_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            location = data["items"][0]["position"]
            return location["lat"], location["lng"]
        else:
            return None, None
    except Exception as e:
        print(f"Errore durante la geocodifica per l'indirizzo {address}: {e}")
        return None, None

# Carica il file CSV esistente (modifica il percorso se necessario)
base_path = os.path.join(os.path.dirname(__file__), '..', 'Dat')
csv_file = os.path.join(base_path, 'alloggi.csv')
df = pd.read_csv(csv_file)

# Ciclo per aggiornare solo le righe con latitudine o longitudine mancanti
for index, row in df.iterrows():
    # Controlla se la latitudine o longitudine sono vuote
    if pd.isnull(row['latitudine']) or pd.isnull(row['longitudine']):
        address = row['Indirizzo']
        print(f"Sto cercando le coordinate per: {address}")
        
        lat, lon = get_coordinates(address)
        
        if lat and lon:
            df.at[index, 'latitudine'] = lat
            df.at[index, 'longitudine'] = lon
            print(f"Coordinate trovate per {address}: ({lat}, {lon})")
        else:
            print(f"Coordinate non trovate per {address}")

# Salva il file CSV aggiornato
df.to_csv(csv_file, index=False)
print(f"File CSV aggiornato salvato in {csv_file}")
