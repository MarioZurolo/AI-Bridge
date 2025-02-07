import pandas as pd
import os

# Percorso della cartella contenente i file CSV
cartella = 'C:\\Users\\benny\\Desktop\\DATASET' 

# Percorsi completi dei file CSV e relativi separatori
file_paths_sep = {
    os.path.join(cartella, 'Lavori_Artigianato_Creatività_Manuale.csv'): ';',
    os.path.join(cartella, 'Lavori_Educazione_Formazione.csv'): ';',
    os.path.join(cartella, 'Lavori_IT_Sviluppo_Software.csv'): ';',
    os.path.join(cartella, 'Lavori_Marketing_Comunicazione.csv'): ';',
    os.path.join(cartella, 'Lavori_Pulizie_Servizi_per_la_Casa.csv'): ';',
    os.path.join(cartella, 'Lavori_Ristorazione_Catering.csv'): ';',
    os.path.join(cartella, 'Lavori_settore_Design&Creativita.csv'): ';',
    os.path.join(cartella, 'Lavori_Tecnico_Manutenzione.csv'): ';'
}

# Unione dei file
all_dataframes = []
for file, sep in file_paths_sep.items():
    df = pd.read_csv(file, sep=sep, on_bad_lines='skip')  # Salta righe problematiche
    all_dataframes.append(df)

# Concatenazione di tutti i DataFrame
merged_df = pd.concat(all_dataframes, ignore_index=True)

# Ricalcolo della colonna 'ID' partendo da 1 fino al numero totale di righe
if 'ID' in merged_df.columns:
    merged_df['ID'] = range(1, len(merged_df) + 1)
else:
    merged_df.insert(0, 'ID', range(1, len(merged_df) + 1))

# Salvataggio del file unito
output_path = os.path.join(cartella, 'Lavori_Uniti.csv')
merged_df.to_csv(output_path, index=False)

print(f"Unione completata! Il file è stato salvato come '{output_path}'.")