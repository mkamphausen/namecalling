import pandas as pd

def count_values_in_column():
    # CSV-Datei einlesen
    df1 = pd.read_csv('data/biden_processed/biden_with_predictions.csv',sep=',', encoding='utf-8', on_bad_lines='skip', low_memory=False, lineterminator='\n')
    df2 = pd.read_csv('data/trump_processed/trump_with_predictions.csv',sep=',', encoding='utf-8', on_bad_lines='skip', low_memory=False, lineterminator='\n')

    # Zählen der 1en und 0en in der angegebenen Spalte
    count_1_df1 = df1['name_calling_pred'].value_counts().get(1, 0)  # Anzahl der 1en
    count_0_df1 = df1['name_calling_pred'].value_counts().get(0, 0)  # Anzahl der 0en
    count_1_df2 = df2['name_calling_pred'].value_counts().get(1, 0)  # Anzahl der 1en
    count_0_df2 = df2['name_calling_pred'].value_counts().get(0, 0)  # Anzahl der 0en
    
    print("-------------------------------------")
    print(f"name calling #Biden: {count_1_df1}")
    print(f"not name calling #Biden: {count_0_df1}")
    print("-------------------------------------")
    print(f"name calling #Trump: {count_1_df2}")
    print(f"not name calling #Trump: {count_0_df2}")
    print("-------------------------------------")
# Beispiel für die Verwendung der Funktion
file_path = 'dein_dateipfad.csv'  # Pfad zur CSV-Datei anpassen
column_name = 'name_calling_pred'      # Name der Spalte anpassen
count_values_in_column()

