import jwt
import base64
import datetime

# Chiave segreta (uguale a quella di Spring, decodificata)
SECRET_KEY = "UmdVa1hwMnI1dTh4L0E/RChHK0tiUGVTaFZtWXEzdDY="
SECRET_KEY_BYTES = base64.b64decode(SECRET_KEY)

def generate_token(email, password):
    """Genera un token JWT valido per 24 ore con tutti i dati necessari."""
    payload = {
        "sub": email,  # email dell'utente
        "password": password,   # id basato su email (può essere anche un altro identificatore)

        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24),  # scadenza del token
    }
    token = jwt.encode(payload, SECRET_KEY_BYTES, algorithm="HS256")
    return token

def verify_token(token):
    """Verifica e decodifica il token JWT."""
    try:
        print("Token in arrivo:", token)  # Debugging: mostra il token ricevuto
        decoded = jwt.decode(token, SECRET_KEY_BYTES, algorithms=["HS256"])
        return decoded
    except jwt.ExpiredSignatureError:
        print("Token scaduto!")
        return "Token scaduto"
    except jwt.InvalidTokenError:
        print("Token non valido!")
        return "Token non valido"

if __name__ == "__main__":
    # Genera un token con dati aggiuntivi
    token = generate_token(
        "rifugiato1@example.com", "Cazzarola69!"
    )
    print("Token generato:", token)
    print("Verifica token:", verify_token(token))
