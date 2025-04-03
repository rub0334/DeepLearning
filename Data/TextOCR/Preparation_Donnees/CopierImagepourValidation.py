import json
import os
import shutil
from tqdm import tqdm

# Chemins des dossiers et fichiers
json_path = '/Data/TextOCR/Validation/TextOCR_AnnotationsValidations.json'
train_images_dir = '/Data/TextOCR/Train/Train_Images'
validation_images_dir = '/Data/TextOCR/Validation/Validation_Images'

# Créer le dossier de validation s'il n'existe pas
os.makedirs(validation_images_dir, exist_ok=True)

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
copied_images = 0
missing_images = 0

# Copier les images
print("Copie des images de validation en cours...")
for image_id in tqdm(validation_image_ids):
    source_path = os.path.join(train_images_dir, f"{image_id}.jpg")
    dest_path = os.path.join(validation_images_dir, f"{image_id}.jpg")

    if os.path.exists(source_path):
        shutil.copy2(source_path, dest_path)
        copied_images += 1
    else:
        missing_images += 1
        print(f"Image non trouvée: {image_id}.jpg")

# Afficher les statistiques
print("\n=== Statistiques ===")
print(f"Total d'images de validation dans le JSON: {total_images}")
print(f"Images copiées avec succès: {copied_images}")
print(f"Images manquantes: {missing_images}")
print(f"Pourcentage de réussite: {copied_images / total_images * 100:.2f}%")
print(f"\nLes images de validation ont été copiées dans: {validation_images_dir}")
