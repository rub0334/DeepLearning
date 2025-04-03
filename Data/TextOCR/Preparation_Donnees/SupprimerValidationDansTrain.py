import json
import os
from tqdm import tqdm

# Chemins des dossiers et fichiers
json_path = '/Data/TextOCR/Validation/TextOCR_AnnotationsValidations.json'
train_images_dir = '/Data/TextOCR/Train/Train_Images'
validation_images_dir = '/Data/TextOCR/Validation/Validation_Images'

# Charger le fichier JSON
print(f"Chargement du fichier JSON: {json_path}")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extraire les IDs des images de validation
validation_image_ids = [img_data['id'] for img_id, img_data in data['imgs'].items()
                        if img_data['set'] == 'val']

print(f"Nombre d'images de validation trouvées dans le JSON: {len(validation_image_ids)}")

# Statistiques
total_images = len(validation_image_ids)
deleted_images = 0
missing_images = 0

# Vérifier que les images existent dans le dossier de validation avant de les supprimer
print("Vérification des images dans le dossier de validation...")
validation_images_exist = []
for image_id in validation_image_ids:
    val_path = os.path.join(validation_images_dir, f"{image_id}.jpg")
    if os.path.exists(val_path):
        validation_images_exist.append(image_id)
    else:
        print(f"Attention: Image non trouvée dans le dossier de validation: {image_id}.jpg")

print(f"Images confirmées dans le dossier de validation: {len(validation_images_exist)}/{total_images}")

# Supprimer les images du dossier d'entraînement
print("Suppression des images de validation du dossier d'entraînement...")
for image_id in tqdm(validation_images_exist):
    source_path = os.path.join(train_images_dir, f"{image_id}.jpg")

    if os.path.exists(source_path):
        os.remove(source_path)
        deleted_images += 1
    else:
        missing_images += 1

# Afficher les statistiques
print("\n=== Statistiques ===")
print(f"Total d'images de validation dans le JSON: {total_images}")
print(f"Images supprimées du dossier d'entraînement: {deleted_images}")
print(f"Images non trouvées dans le dossier d'entraînement: {missing_images}")
print(f"Pourcentage de suppression: {deleted_images / len(validation_images_exist) * 100:.2f}%")
