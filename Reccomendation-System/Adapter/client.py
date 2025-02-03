# Codice client per la richiesta dei dati

import requests

API_URL = "http://127.0.0.1:5000"  # URL del tuo server Flask

# 1️⃣ LOGIN - Ottieni il token
login_data = {"email": "rifugiato1@example.com", "password": "Cazzarola69!"}  # Assicurati di includere anche la password
response = requests.post(f"{API_URL}/login", json=login_data)

if response.status_code == 200:
    token = response.json().get("token")
    print("✅ Token ottenuto:", token)
else:
    print("❌ Errore di login:", response.json())
    exit()

# 2️⃣ RICHIESTA DATI - Usa il token per accedere ai dati da Spring Boot
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(f"{API_URL}/get_data", headers=headers)

if response.status_code == 200:
    print("✅ Dati ricevuti da Spring Boot:", response.json())
else:
    print("❌ Errore nell'accesso ai dati:", response.json())