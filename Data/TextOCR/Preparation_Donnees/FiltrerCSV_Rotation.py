import pandas as pd
import os

# Chemins des fichiers CSV d'entrée
test_csv_path = '/Data/TextOCR/Test/test-images-with-rotation.csv'
train_csv_path = '/Data/TextOCR/Train/train-images-boxable-with-rotation.csv'

# Chemins des fichiers CSV de sortie
test_filtered_path = '../Test/test-images-with-rotation-filtered.csv'
train_filtered_path = '../Train/train-images-boxable-with-rotation-filtered.csv'

try:
    # Chargement des fichiers CSV
    print(f"Chargement du fichier test depuis: {test_csv_path}")
    test_csv = pd.read_csv(test_csv_path)

    print(f"Chargement du fichier train depuis: {train_csv_path}")
    train_csv = pd.read_csv(train_csv_path)

    # Statistiques sur les fichiers originaux
    print("\n=== Statistiques Originales ===")
    print(f"Fichier test: {test_csv.shape[0]} lignes, {test_csv.shape[1]} colonnes")
    print(f"Fichier train: {train_csv.shape[0]} lignes, {train_csv.shape[1]} colonnes")

    # Filtrage des lignes où Rotation n'est pas 0.0 et n'est pas vide
    test_filtered = test_csv[(test_csv['Rotation'] != 0.0) & (test_csv['Rotation'].notna())]
    train_filtered = train_csv[(train_csv['Rotation'] != 0.0) & (train_csv['Rotation'].notna())]

    # Statistiques sur les données filtrées
    print("\n=== Statistiques Filtrées ===")
    print(f"Fichier test (images avec rotation): {test_filtered.shape[0]} lignes")
    print(f"Pourcentage d'images pivotées dans le jeu test: {len(test_filtered) / len(test_csv) * 100:.2f}%")

    print(f"Fichier train (images avec rotation): {train_filtered.shape[0]} lignes")
    print(f"Pourcentage d'images pivotées dans le jeu train: {len(train_filtered) / len(train_csv) * 100:.2f}%")

    # Sauvegarde des données filtrées
    test_filtered.to_csv(test_filtered_path, index=False)
    train_filtered.to_csv(train_filtered_path, index=False)

    print("\n=== Fichiers enregistrés ===")
    print(f"- CSV test filtré: {test_filtered_path}")
    print(f"- CSV train filtré: {train_filtered_path}")

except FileNotFoundError as e:
    print(f"Erreur: {e}")
    print("\nAssurez-vous que les fichiers CSV sont au bon endroit.")
    print("Vérifiez les chemins suivants:")
    print(f"- {test_csv_path}")
    print(f"- {train_csv_path}")
