import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm


def rotate_images(csv_path, source_folder, destination_folder, image_extension='.jpg'):
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(destination_folder, exist_ok=True)

    # Charger le CSV
    print(f"Chargement du fichier CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Statistiques
    total_images = len(df)
    processed_images = 0
    missing_images = 0

    print(f"Traitement de {total_images} images...")

    # Parcourir chaque ligne du CSV
    for index, row in tqdm(df.iterrows(), total=total_images):
        image_id = row['ImageID']
        rotation = row['Rotation']

        # Construire le chemin de l'image source
        source_image_path = os.path.join(source_folder, f"{image_id}{image_extension}")

        # Vérifier si l'image existe
        if not os.path.exists(source_image_path):
            missing_images += 1
            continue

        # Charger l'image
        image = cv2.imread(source_image_path)

        # Appliquer la rotation
        if rotation == 90.0:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 180.0:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        elif rotation == 270.0:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            # Pour les rotations non standard, utiliser une matrice de rotation
            h, w = image.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            rotated_image = cv2.warpAffine(image, M, (w, h))

        # Enregistrer l'image pivotée
        destination_image_path = os.path.join(destination_folder, f"{image_id}{image_extension}")
        cv2.imwrite(destination_image_path, rotated_image)

        processed_images += 1

    # Afficher les statistiques
    print("\n=== Statistiques ===")
    print(f"Total d'images dans le CSV: {total_images}")
    print(f"Images traitées: {processed_images}")
    print(f"Images manquantes: {missing_images}")
    print(f"Pourcentage de réussite: {processed_images / total_images * 100:.2f}%")

    return processed_images, missing_images


# Traitement des images de test
test_csv_path = '/Data/TextOCR/Test/test-images-with-rotation-filtered.csv'
test_source_folder = '/Users/maxime/Documents/Cours/DeepLearning/Data/TextOCR/Test/Test_Images'
test_destination_folder = '/Users/maxime/Documents/Cours/DeepLearning/Data/TextOCR/Test/Test_Images_Correct'

print("=== Traitement des images de test ===")
test_processed, test_missing = rotate_images(test_csv_path, test_source_folder, test_destination_folder)

# Traitement des images d'entraînement
train_csv_path = '/Data/TextOCR/Train/train-images-boxable-with-rotation-filtered.csv'
train_source_folder = '/Users/maxime/Documents/Cours/DeepLearning/Data/TextOCR/Train/Train_Images'
train_destination_folder = '/Users/maxime/Documents/Cours/DeepLearning/Data/TextOCR/Train/Train_Images_Correct'

print("\n=== Traitement des images d'entraînement ===")
train_processed, train_missing = rotate_images(train_csv_path, train_source_folder, train_destination_folder)

# Statistiques globales
print("\n=== Statistiques globales ===")
print(f"Total d'images traitées: {test_processed + train_processed}")
print(f"Total d'images manquantes: {test_missing + train_missing}")
