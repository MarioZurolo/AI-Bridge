import random
import csv
from faker import Faker 
import os

fake = Faker()

# Configurazione di base
num_rifugiati = 50

# Caricare skill da un file
def carica(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines()]

# Percorso della directory corrente
current_dir = os.path.dirname(__file__)

# Carica le skill da un file esterno
skills = carica(os.path.join(current_dir, "Compentenze.txt"))
nazionalita = carica(os.path.join(current_dir, "Nazionalità.txt"))
lingue = carica(os.path.join(current_dir, "Lingue.txt"))


# Opzioni di esempio
titoli_studio = ["ScuolaPrimaria", "ScuolaSecondariaPrimoGrado", "ScuolaSecondariaSecondoGrado", "Laurea", "Specializzazione"] #, "Laurea", "Dottorato"]
pesi_titoli_studio = [0.65, 0.20, 0.10, 0.03, 0.02]
genere = ["Maschio", "Femmina", "NonSpecificato"]

# Generazione rifugiati
rifugiati = []
email_set = set()
for _ in range(num_rifugiati):
    email = fake.email()
    while email in email_set:
        email = fake.email()
    email_set.add(email)
    rifugiati.append({
        "Email": email,
        "Nome": fake.first_name(),
        "Cognome": fake.last_name(),
        "Data di nascita": fake.date_of_birth(minimum_age=18, maximum_age=60).strftime("%Y-%m-%d"),
        "Nazionalità": random.choice(nazionalita),
        "Lingue parlate": ", ".join(random.sample(lingue, k=random.randint(1, 2))),
        "Genere": random.choice(genere),
        "Skill": ", ".join(random.sample(skills, k=random.randint(1, 2))),
        "Titolo di studio": random.choices(titoli_studio, pesi_titoli_studio)[0],
    })


# Esportare i dati in CSV
def export_to_csv(filename, data, fieldnames):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Esportazione
export_to_csv(os.path.join(current_dir, "rifugiati.csv"), rifugiati, rifugiati[0].keys())