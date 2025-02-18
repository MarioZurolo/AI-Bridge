from flask import Flask, request, jsonify
import requests
import logging
from auth import generate_token, verify_token
from flask_cors import CORS # Per risolvere il problema di CORS
from sqlalchemy import create_engine
import pymysql
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RAG')))
from raccomandazioneLavori import match_jobs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','ModuloFia')))

#sys.path.append('/Users/mariozurolo/AI-Bridge/ModuloFia')
from Connector import get_db_connection

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Location')))
from matchLocation import match_housing

app = Flask(__name__)

# Abilita CORS per tutte le origini
# Configurazione CORS: consenti richieste da React (localhost:3000)
CORS(app, resources={r"/*": {"origins": "http://localhost:5174"}})


SPRING_API_URL = "http://localhost:8080/alloggi/mostra"  # URL dell'API Spring


@app.route('/')
def home():
    return 'Server Flask in esecuzione!'


#Endpoint per chiedere i dati a spring boot
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


# Endpoint per inviare dati da Spring Boot
@app.route('/send_data', methods=['POST'])
def receive_data_from_spring():
    try:
        json_data = request.get_json()
        email = json_data.get("email")
        logging.info(f"📩 Email ricevuta dal frontend: {email}")

        if not email:
            return jsonify({"error": "Email non fornita"}), 400

        db_connection = get_db_connection()
        recommendations = match_jobs(email, db_connection)
        best_housing = match_housing(recommendations, db_connection)

        response_data = {
            "recommendations": recommendations,
            "best_housing": best_housing
        }

        logging.info(f"📤 JSON restituito al frontend:\n{json.dumps(response_data, indent=2)}")

        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("❌ Errore nel backend")
        return jsonify({"error": str(e)}), 500

    finally:
        if 'db_connection' in locals():
            db_connection.close()


if __name__ == '__main__':
    app.run(debug=True, port=5000)