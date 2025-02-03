import random
import csv
from faker import Faker 
import os

# Creiamo un'istanza di Faker
fake = Faker('it_IT')  # Per generare indirizzi italiani

# Percorso della directory corrente
current_dir = os.path.dirname(__file__)

# Funzione per generare un alloggio
def genera_alloggio(id):
    metratura = random.randint(20, 199)
    max_persone = random.randint(1, 10)
    indirizzo = fake.address().replace("\n", ", ")
    
    alloggio = {
        "ID": id,
        "Metratura": metratura,
        "Max persone": max_persone,
        "Indirizzo": indirizzo
    }
    
    return alloggio

# Generiamo 200 alloggi
alloggi = [genera_alloggio(i) for i in range(1, 201)]

# Esportare i dati in CSV
def export_to_csv(filename, data, fieldnames):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Esportazione
export_to_csv(os.path.join(current_dir, "alloggi.csv"), alloggi, alloggi[0].keys())
