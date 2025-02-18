from sqlalchemy import create_engine

# Connessione al database con SQLAlchemy
def get_db_connection():
    engine = create_engine('mysql+pymysql://root:root@localhost:3306/bridge')  # Modifica con le tue credenziali
    return engine.connect()