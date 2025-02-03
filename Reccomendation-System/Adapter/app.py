from flask import Flask, request, jsonify
from auth import generate_token, verify_token
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Aggiorna il livello di logging
logging.getLogger().setLevel(logging.DEBUG)

app = Flask(__name__)

SPRING_API_URL = "http://localhost:8080/alloggi/mostra"  # URL dell'API Spring

@app.route("/login", methods=["POST"])
def login():
    """ Endpoint per generare un token dopo il login. """
    data = request.json
    email = data.get("email")
    ruolo = data.get("password")  # Es: "USER" o "ADMIN"

    if not email or not ruolo:
        return jsonify({"error": "Email o password richiesti"}), 400

    token = generate_token(email, ruolo)
    return jsonify({"token": token})

@app.route("/get_data", methods=["GET"])
def get_data():
    """ Recupera dati da Spring Boot usando JWT. """
    token = request.headers.get("Authorization")
    print("Token in arrivo:", token)  # Debugging: mostra il token ricevuto
    logging.info(f"Token in arrivo: {token}")

    if not token:
        print("Token mancante!")
        logging.error("Token mancante!")
        return jsonify({"error": "Token mancante"}), 401

    # Rimuove "Bearer " se presente
    token = token.replace("Bearer ", "")
    decoded = verify_token(token)
    print("Token decodificato:", decoded)  # Debugging: mostra il token decodificato
    logging.info(f"Token decodificato: {decoded}")

    if isinstance(decoded, str):
        print("Token non valido/scaduto!")
        logging.error("Token non valido/scaduto!")
        return jsonify({"error": decoded}), 401  # Token non valido/scaduto

    # Definisci 'headers' con il token
    headers = {"Authorization": f"Bearer {token}"}

    try:
        # Effettua la richiesta GET a Spring Boot
        response = requests.get(SPRING_API_URL, headers=headers)
        print(f"Response status code: {response.status_code}")
        print(f"Response body: {response.text}")  # Mostra il corpo della risposta in formato testo
        logging.info(f"Response status code: {response.status_code}")

        if response.status_code == 200:
            print("Dati ricevuti da Spring Boot:", response.json())
            logging.info(f"Dati ricevuti da Spring Boot: {response.json()}")
            return jsonify(response.json())
        else:
            print("Errore nell'accesso ai dati:", response.json())
            logging.error(f"Errore nell'accesso ai dati: {response.json()}")
            return jsonify({"error": "Errore nell'accesso ai dati", "details": response.json()}), response.status_code

    except Exception as e:
        return jsonify({"error": "Errore durante la richiesta a Spring Boot", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
