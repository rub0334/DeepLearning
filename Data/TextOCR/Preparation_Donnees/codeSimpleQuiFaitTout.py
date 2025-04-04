import os
import pandas as pd
import cv2
import numpy as np
import shutil
import json
from tqdm import tqdm
import sys # Import sys to check for file existence before processing

# ==============================================================================
# Configuration des Chemins
# ==============================================================================
# !!! IMPORTANT : Vérifiez que ce chemin de base est correct !!!
BASE_DIR = r'C:\Users\rubde\Documents\deep_learning'

# ---- Dossiers d'images ----
# Dossier contenant les images d'entraînement originales
TRAIN_IMAGES_DIR_ORIGINAL = os.path.join(BASE_DIR, 'train_images')
# Dossier contenant les images de test originales (supposé, adaptez si nécessaire)
TEST_IMAGES_DIR_ORIGINAL = os.path.join(BASE_DIR, 'test_images') # Supposons que ce dossier existe ou sera créé si besoin
# Dossier où les images de validation seront copiées
VALIDATION_IMAGES_DIR = os.path.join(BASE_DIR, 'validation_images')
# Dossier où TOUTES les images d'entraînement finales (corrigées + non pivotées) seront
TRAIN_IMAGES_DIR_FINAL = os.path.join(BASE_DIR, 'train_images_correct') # Renommé pour la clarté
# Dossier où les images de test corrigées seront sauvegardées
TEST_IMAGES_DIR_CORRECTED = os.path.join(BASE_DIR, 'test_images_correct')

# ---- Fichiers CSV ----
# Fichier CSV original pour le train
TRAIN_CSV_PATH_ORIGINAL = os.path.join(BASE_DIR, 'train-images-boxable-with-rotation.csv')
# Fichier CSV original pour le test
TEST_CSV_PATH_ORIGINAL = os.path.join(BASE_DIR, 'test-images-with-rotation.csv')
# Fichier CSV filtré pour le train (images nécessitant rotation)
TRAIN_CSV_PATH_FILTERED = os.path.join(BASE_DIR, 'train-images-boxable-with-rotation-filtered.csv')
# Fichier CSV filtré pour le test (images nécessitant rotation)
TEST_CSV_PATH_FILTERED = os.path.join(BASE_DIR, 'test-images-with-rotation-filtered.csv')

# ---- Fichiers JSON ----
# Fichier JSON contenant les informations de validation
JSON_VAL_PATH = os.path.join(BASE_DIR, 'TextOCR_0.1_val.json')
# Fichier JSON des annotations train (utilisé pour confirmer la structure, pas pour les IDs ici)
JSON_TRAIN_PATH = os.path.join(BASE_DIR, 'TextOCR_0.1_train.json')

# ---- Paramètres ----
IMAGE_EXTENSION = '.jpg' # Extension des fichiers image

# Listes globales pour stocker les IDs de validation extraits et les images manquantes
validation_image_ids_list = []
missing_files_rotation = []  # Liste des fichiers manquants pendant la rotation
missing_files_non_rotated = [] # Liste des fichiers manquants pendant la copie non-pivotée

# ==============================================================================
# Fonctions Utilitaires (Mêmes fonctions que précédemment)
# ==============================================================================

def check_file_exists(filepath, filename):
    """Vérifie si un fichier existe et quitte si ce n'est pas le cas."""
    if not os.path.exists(filepath):
        print(f"\nERREUR CRITIQUE: Le fichier requis '{filename}' est introuvable à l'emplacement :")
        print(f"  {filepath}")
        print("Veuillez vérifier le chemin et le nom du fichier.")
        print("Arrêt du script.")
        sys.exit(1) # Quitte le script avec un code d'erreur
    else:
        print(f"Fichier trouvé : {filepath}")

def filter_csv_by_rotation(input_csv_path, output_csv_path, csv_type=""):
    """
    Charge un fichier CSV, filtre les lignes où 'Rotation' n'est ni 0.0 ni NaN,
    et sauvegarde le résultat filtré.
    """
    print(f"\n--- Filtrage du fichier CSV {csv_type} ---")
    check_file_exists(input_csv_path, os.path.basename(input_csv_path))

    try:
        print(f"Chargement depuis : {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        print(f"Nombre total de lignes initiales : {len(df)}")

        # Filtrage
        df_filtered = df[(df['Rotation'] != 0.0) & (df['Rotation'].notna())].copy()
        num_filtered = len(df_filtered)
        percentage = (num_filtered / len(df) * 100) if len(df) > 0 else 0
        print(f"Nombre de lignes après filtrage (Rotation != 0.0 et non vide) : {num_filtered}")
        print(f"Pourcentage d'images nécessitant une rotation : {percentage:.2f}%")

        # Sauvegarde
        df_filtered.to_csv(output_csv_path, index=False)
        print(f"Fichier CSV filtré sauvegardé : {output_csv_path}")
        return num_filtered

    except Exception as e:
        print(f"Erreur lors du filtrage du CSV {input_csv_path}: {e}")
        return 0

def extract_validation_ids(json_path):
    """
    Extrait et retourne la liste des IDs d'image de validation depuis le JSON.
    Stocke également les IDs dans la variable globale validation_image_ids_list.
    """
    global validation_image_ids_list # Pour modifier la variable globale
    print(f"\n--- Extraction des IDs de Validation depuis JSON ---")
    check_file_exists(json_path, os.path.basename(json_path))
    validation_ids = []
    try:
        print(f"Chargement du fichier JSON : {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Essayer d'extraire basé sur la structure TextOCR (set='val')
        if 'imgs' in data and isinstance(data['imgs'], dict):
             validation_ids = [img_data['id'] for img_id, img_data in data.get('imgs', {}).items()
                                   if img_data.get('set') == 'val']
             print(f"Structure JSON 'imgs' détectée. IDs extraits avec set='val'.")

        # Si ce n'est pas la structure standard, essayer comme une liste d'objets
        elif isinstance(data, list):
             print(f"Structure JSON de type liste détectée pour {os.path.basename(json_path)}.")
             print("Tentative d'extraction des 'image_id' ou 'id'. Adaptez si nécessaire.")
             count_found = 0
             for item in data:
                 img_id = item.get('image_id', item.get('id'))
                 if img_id:
                     validation_ids.append(str(img_id)) # Assurer que c'est une chaîne
                     count_found += 1
                 # else:
                     # print(f"Avertissement: Élément sans 'image_id' ou 'id' trouvé dans la liste JSON: {item}")
             print(f"{count_found} IDs extraits de la liste.")

        # Si aucune des structures ne correspond
        elif 'imgs' not in data and not isinstance(data, list):
             print(f"Avertissement: La structure du JSON {os.path.basename(json_path)} n'est pas reconnue ('imgs' dict ou liste).")
             # Tenter d'extraire tous les IDs de premier niveau si c'est un dict d'images
             if isinstance(data, dict):
                 potential_ids = list(data.keys())
                 # Heuristique simple: si les clés ressemblent à des IDs d'image
                 if all(isinstance(k, str) for k in potential_ids):
                     print("Tentative d'utiliser les clés du dictionnaire comme IDs.")
                     validation_ids = potential_ids
                 else:
                     print("Impossible d'extraire les IDs de manière fiable.")

        total_ids_in_json = len(validation_ids)
        print(f"Nombre total d'IDs d'images de validation extraits : {total_ids_in_json}")
        validation_image_ids_list = validation_ids # Met à jour la liste globale
        return validation_ids

    except json.JSONDecodeError:
        print(f"Erreur: Impossible de décoder le fichier JSON: {json_path}. Est-ce un JSON valide?")
        validation_image_ids_list = []
        return []
    except KeyError as e:
        print(f"Erreur: Clé manquante probable dans le JSON ({e}). Vérifiez la structure de {os.path.basename(json_path)}.")
        validation_image_ids_list = []
        return []
    except Exception as e:
        print(f"Erreur inattendue lors de l'extraction des IDs de validation : {e}")
        validation_image_ids_list = []
        return []


def copy_validation_images(validation_ids, source_images_dir, dest_images_dir):
    """
    Copie les images de validation (dont les IDs sont fournis) du dossier source
    vers le dossier de destination.
    """
    print("\n--- Copie des Images de Validation ---")
    if not validation_ids:
        print("Aucun ID de validation fourni, copie sautée.")
        return 0

    if not os.path.isdir(source_images_dir):
         print(f"\nERREUR: Le dossier source '{source_images_dir}' n'existe pas pour la copie.")
         return 0

    # Créer le dossier de destination
    os.makedirs(dest_images_dir, exist_ok=True)
    print(f"Dossier source : {source_images_dir}")
    print(f"Dossier de destination : {dest_images_dir}")

    copied_count = 0
    missing_in_source_count = 0
    total_to_copy = len(validation_ids)

    print(f"Tentative de copie de {total_to_copy} images de validation...")
    for image_id in tqdm(validation_ids, desc="Copie Validation"):
        source_path = os.path.join(source_images_dir, f"{image_id}{IMAGE_EXTENSION}")
        dest_path = os.path.join(dest_images_dir, f"{image_id}{IMAGE_EXTENSION}")

        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"\nErreur lors de la copie de {image_id}{IMAGE_EXTENSION}: {e}")
        else:
            missing_in_source_count += 1
            # print(f"\nAttention: Image source non trouvée pour copie : {source_path}")

    print("\nStatistiques de copie Validation :")
    print(f"- Total d'IDs à copier : {total_to_copy}")
    print(f"- Images copiées avec succès : {copied_count}")
    print(f"- Images manquantes dans le dossier source ({os.path.basename(source_images_dir)}) : {missing_in_source_count}")
    if total_to_copy > 0 :
        print(f"- Pourcentage de réussite : {copied_count / total_to_copy * 100:.2f}%")

    return copied_count


def delete_validation_from_train(validation_ids, train_images_dir, validation_images_dir):
    """
    Supprime les images de validation (IDs fournis) du dossier d'entraînement,
    après vérification optionnelle dans le dossier de validation.
    """
    print("\n--- Suppression des Images de Validation du Dossier d'Entraînement ---")
    if not validation_ids:
        print("Aucun ID de validation fourni, suppression sautée.")
        return 0

    if not os.path.isdir(train_images_dir):
         print(f"\nERREUR: Le dossier d'entraînement '{train_images_dir}' n'existe pas. Suppression impossible.")
         return 0

    # Vérifier l'existence dans le dossier de validation (si le dossier existe)
    ids_to_delete = []
    if os.path.isdir(validation_images_dir):
        print("Vérification de l'existence des images dans le dossier de validation...")
        verified_count = 0
        for image_id in tqdm(validation_ids, desc="Vérif Validation"):
            val_path = os.path.join(validation_images_dir, f"{image_id}{IMAGE_EXTENSION}")
            if os.path.exists(val_path):
                ids_to_delete.append(image_id)
                verified_count +=1
            # else:
                # print(f"\nAttention: Image {image_id} non trouvée dans {validation_images_dir}, ne sera pas supprimée de {train_images_dir}.")
        print(f"Images confirmées dans le dossier validation : {verified_count}/{len(validation_ids)}")
        if verified_count != len(validation_ids):
             print("ATTENTION : Certaines images listées pour validation n'ont pas pu être trouvées dans le dossier de validation après copie.")
    else:
         print(f"Attention: Dossier de validation {validation_images_dir} non trouvé. Suppression basée uniquement sur la liste d'IDs fournie.")
         ids_to_delete = validation_ids # Supprimer tous les IDs de la liste

    # Statistiques Suppression
    deleted_count = 0
    missing_in_train_count = 0
    total_to_delete = len(ids_to_delete)

    if total_to_delete == 0:
        print("Aucune image à supprimer du dossier d'entraînement (soit liste vide, soit échec de vérification).")
        return 0

    print(f"Tentative de suppression de {total_to_delete} images du dossier d'entraînement ({os.path.basename(train_images_dir)})...")
    for image_id in tqdm(ids_to_delete, desc="Suppression Train"):
        train_path = os.path.join(train_images_dir, f"{image_id}{IMAGE_EXTENSION}")

        if os.path.exists(train_path):
            try:
                os.remove(train_path)
                deleted_count += 1
            except Exception as e:
                print(f"\nErreur lors de la suppression de {train_path}: {e}")
        else:
            missing_in_train_count += 1
            # print(f"\nImage {image_id} non trouvée dans {train_images_dir} (peut-être déjà supprimée ou jamais existé).")

    print("\nStatistiques de suppression du dossier Train :")
    print(f"- Total d'IDs à supprimer (après vérif si possible) : {total_to_delete}")
    print(f"- Images supprimées avec succès : {deleted_count}")
    print(f"- Images non trouvées dans le dossier train lors de la suppression : {missing_in_train_count}")
    if total_to_delete > 0:
         print(f"- Pourcentage de suppression : {deleted_count / total_to_delete * 100:.2f}%")

    return deleted_count


def rotate_images_from_csv(csv_path, source_folder, destination_folder, csv_type=""):
    """
    Lit un fichier CSV (filtré), charge les images correspondantes depuis le
    dossier source, applique la rotation spécifiée et sauvegarde dans le
    dossier de destination.
    """
    global missing_files_rotation # Pour modifier la variable globale

    print(f"\n--- Rotation des Images {csv_type} ---")
    check_file_exists(csv_path, os.path.basename(csv_path))

    if not os.path.isdir(source_folder):
         print(f"\nERREUR: Le dossier source '{source_folder}' n'existe pas pour la rotation.")
         return 0, 0

    # Créer le dossier de destination
    os.makedirs(destination_folder, exist_ok=True)
    print(f"Dossier source : {source_folder}")
    print(f"Dossier de destination : {destination_folder}")

    try:
        print(f"Chargement du fichier CSV filtré : {csv_path}")
        df = pd.read_csv(csv_path)
        # Assurer que ImageID est une chaîne pour la comparaison future avec les sets
        df['ImageID'] = df['ImageID'].astype(str)
        total_images_in_csv = len(df)
        print(f"Traitement de {total_images_in_csv} images basé sur le CSV filtré...")

        processed_count = 0
        missing_in_source_count = 0
        error_count = 0

        for index, row in tqdm(df.iterrows(), total=total_images_in_csv, desc=f"Rotation {csv_type}"):
            image_id = str(row['ImageID']) # Assurer str
            rotation = row['Rotation']

            source_image_path = os.path.join(source_folder, f"{image_id}{IMAGE_EXTENSION}")
            destination_image_path = os.path.join(destination_folder, f"{image_id}{IMAGE_EXTENSION}")

            if not os.path.exists(source_image_path):
                missing_in_source_count += 1
                # Stocker le nom du fichier manquant
                missing_files_rotation.append(source_image_path)
                # print(f"\nImage source non trouvée pour rotation : {source_image_path}")
                continue

            try:
                image = cv2.imread(source_image_path)
                if image is None:
                    print(f"\nErreur: Impossible de charger l'image {source_image_path}")
                    error_count += 1
                    continue

                rotated_image = None
                if rotation == 90.0:
                    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == 180.0:
                    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
                elif rotation == 270.0:
                    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                else: # Gérer les rotations non standard si présentes
                    h, w = image.shape[:2]
                    center = (w / 2, h / 2)
                    M = cv2.getRotationMatrix2D(center, float(rotation), 1.0)
                    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                if rotated_image is not None:
                    success = cv2.imwrite(destination_image_path, rotated_image)
                    if success:
                        processed_count += 1
                    else:
                         print(f"\nErreur: Échec de l'écriture de l'image {destination_image_path}")
                         error_count += 1
                else:
                     print(f"\nErreur: `rotated_image` est None après tentative de rotation pour {image_id}")
                     error_count += 1

            except Exception as e:
                print(f"\nErreur lors du traitement/rotation de l'image {image_id}: {e}")
                error_count += 1

        print("\nStatistiques de Rotation :")
        print(f"- Total d'images dans le CSV filtré : {total_images_in_csv}")
        print(f"- Images traitées (pivotées) et sauvegardées : {processed_count}")
        print(f"- Images sources manquantes lors de la rotation : {missing_in_source_count}")
        print(f"- Erreurs pendant le traitement/sauvegarde : {error_count}")
        if total_images_in_csv > 0:
             print(f"- Pourcentage de réussite (basé sur CSV filtré) : {processed_count / total_images_in_csv * 100:.2f}%")

        return processed_count, total_images_in_csv

    except Exception as e:
        print(f"Erreur majeure lors de la rotation des images {csv_type}: {e}")
        return 0, 0

# ==============================================================================
# NOUVELLE FONCTION : Copier les images non pivotées et non validation
# ==============================================================================
def copy_non_rotated_train_images(original_csv_path, filtered_csv_path, validation_ids, source_images_dir, destination_images_dir):
    """
    Identifie les images d'entraînement qui n'ont pas été pivotées et ne sont pas
    dans l'ensemble de validation, puis les copie du dossier source vers le dossier final.
    """
    global missing_files_non_rotated  # Pour modifier la variable globale

    print("\n--- Copie des Images d'Entraînement Non Pivotées (et non validation) ---")

    check_file_exists(original_csv_path, os.path.basename(original_csv_path))
    # Le fichier filtré peut ne pas exister s'il n'y avait rien à filtrer, on gère ça
    # check_file_exists(filtered_csv_path, os.path.basename(filtered_csv_path))

    if not os.path.isdir(source_images_dir):
         print(f"\nERREUR: Le dossier source '{source_images_dir}' n'existe pas.")
         return 0

    os.makedirs(destination_images_dir, exist_ok=True) # Assure que le dossier destination existe
    print(f"Dossier source : {source_images_dir}")
    print(f"Dossier de destination : {destination_images_dir}")

    try:
        # Charger tous les IDs du CSV original
        print(f"Chargement du CSV original : {original_csv_path}")
        df_original = pd.read_csv(original_csv_path)
        # Assurer que les IDs sont des strings pour la comparaison avec les sets
        original_ids = set(df_original['ImageID'].astype(str).tolist())
        print(f"Total IDs dans le CSV original : {len(original_ids)}")

        # Charger les IDs du CSV filtré (ceux qui ont été pivotés)
        rotated_ids = set()
        if os.path.exists(filtered_csv_path):
            print(f"Chargement du CSV filtré : {filtered_csv_path}")
            df_filtered = pd.read_csv(filtered_csv_path)
            rotated_ids = set(df_filtered['ImageID'].astype(str).tolist())
            print(f"Total IDs dans le CSV filtré (pivotés) : {len(rotated_ids)}")
        else:
            print(f"Info : Le fichier CSV filtré {filtered_csv_path} n'existe pas (aucune image à pivoter).")


        # IDs de validation (déjà extraits et convertis en set pour efficacité)
        validation_ids_set = set(validation_ids)
        print(f"Total IDs de validation : {len(validation_ids_set)}")

        # Trouver les IDs à copier : ceux de l'original qui ne sont NI pivotés NI validation
        ids_to_copy = original_ids - rotated_ids - validation_ids_set
        num_to_copy = len(ids_to_copy)
        print(f"Nombre d'images à copier (Original - Pivotées - Validation) : {num_to_copy}")

        if num_to_copy == 0:
            print("Aucune image supplémentaire à copier.")
            return 0

        # Copier les fichiers
        copied_count = 0
        missing_count = 0
        error_count = 0
        print("Copie des images non pivotées en cours...")
        for image_id in tqdm(ids_to_copy, desc="Copie Non-Pivotées"):
            source_path = os.path.join(source_images_dir, f"{image_id}{IMAGE_EXTENSION}")
            dest_path = os.path.join(destination_images_dir, f"{image_id}{IMAGE_EXTENSION}")

            # Vérifier si la destination existe déjà (au cas où, même si la logique l'exclut)
            if os.path.exists(dest_path):
                 # print(f"Avertissement: L'image {dest_path} existe déjà dans la destination. Copie sautée.")
                 continue # Ne pas écraser ou recompter

            if os.path.exists(source_path):
                try:
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                except Exception as e:
                    print(f"\nErreur lors de la copie de {source_path} vers {dest_path}: {e}")
                    error_count += 1
            else:
                # Cela peut arriver si l'image était listée dans le CSV original mais manquait déjà
                # OU si elle a été supprimée comme validation mais qu'il y a eu un souci avec les listes d'IDs.
                print(f"\nAttention: Image source non trouvée pour copie : {source_path}")
                # Stocker le nom du fichier manquant
                missing_files_non_rotated.append(source_path)
                missing_count += 1

        print("\nStatistiques de copie des non-pivotées :")
        print(f"- Images identifiées pour copie : {num_to_copy}")
        print(f"- Images copiées avec succès : {copied_count}")
        print(f"- Images sources manquantes lors de la copie : {missing_count}")
        print(f"- Erreurs de copie : {error_count}")

        return copied_count

    except Exception as e:
        print(f"Erreur majeure lors de la copie des images non pivotées : {e}")
        return 0

# ==============================================================================
# Exécution Principale du Workflow
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print(" DÉBUT DU WORKFLOW DE PRÉPARATION DES DONNÉES TEXTOCR (v2)")
    print("="*70)
    print(f"Utilisation du dossier de base : {BASE_DIR}")

    # --- Vérification initiale ---
    print("\n--- Vérification des fichiers/dossiers initiaux ---")
    check_file_exists(TRAIN_CSV_PATH_ORIGINAL, os.path.basename(TRAIN_CSV_PATH_ORIGINAL))
    check_file_exists(TEST_CSV_PATH_ORIGINAL, os.path.basename(TEST_CSV_PATH_ORIGINAL))
    check_file_exists(JSON_VAL_PATH, os.path.basename(JSON_VAL_PATH))
    # check_file_exists(JSON_TRAIN_PATH, os.path.basename(JSON_TRAIN_PATH)) # Vérifier si nécessaire
    if not os.path.isdir(TRAIN_IMAGES_DIR_ORIGINAL):
        print(f"\nERREUR CRITIQUE: Le dossier d'images d'entraînement '{TRAIN_IMAGES_DIR_ORIGINAL}' est introuvable.")
        sys.exit(1)
    if not os.path.isdir(TEST_IMAGES_DIR_ORIGINAL):
        print(f"\nATTENTION: Le dossier d'images de test '{TEST_IMAGES_DIR_ORIGINAL}' est introuvable.")
        print("La rotation et le traitement des images de test seront sautés si le dossier n'est pas créé.")


    # --- Étape 1: Filtrer les CSV pour identifier les rotations ---
    print("\n" + "="*20 + " ÉTAPE 1: FILTRAGE DES CSV " + "="*20)
    filtered_train_count = filter_csv_by_rotation(TRAIN_CSV_PATH_ORIGINAL, TRAIN_CSV_PATH_FILTERED, "Train")
    filtered_test_count = filter_csv_by_rotation(TEST_CSV_PATH_ORIGINAL, TEST_CSV_PATH_FILTERED, "Test")

    # --- Étape 2: Identifier les images de validation ---
    # On extrait les IDs ici pour les réutiliser
    print("\n" + "="*20 + " ÉTAPE 2: IDENTIFICATION VALIDATION " + "="*20)
    val_ids = extract_validation_ids(JSON_VAL_PATH)
    total_val_ids = len(val_ids) # Garder le compte total

    # --- Étape 3: Séparer les images de validation (Copie) ---
    print("\n" + "="*20 + " ÉTAPE 3: SÉPARATION VALIDATION (COPIE) " + "="*20)
    copied_val_count = copy_validation_images(val_ids, TRAIN_IMAGES_DIR_ORIGINAL, VALIDATION_IMAGES_DIR)

    # --- Étape 4: Séparer les images de validation (Suppression de Train) ---
    print("\n" + "="*20 + " ÉTAPE 4: SÉPARATION VALIDATION (SUPPRESSION) " + "="*20)
    # Utilise les IDs extraits 'val_ids' et vérifie contre VALIDATION_IMAGES_DIR
    deleted_val_count = delete_validation_from_train(val_ids, TRAIN_IMAGES_DIR_ORIGINAL, VALIDATION_IMAGES_DIR)

    # --- Étape 5: Appliquer la rotation aux images identifiées ---
    print("\n" + "="*20 + " ÉTAPE 5: ROTATION DES IMAGES " + "="*20)
    processed_train_rot = 0
    if filtered_train_count > 0:
         # Sauvegarde dans le dossier final/correct
         processed_train_rot, _ = rotate_images_from_csv(TRAIN_CSV_PATH_FILTERED, TRAIN_IMAGES_DIR_ORIGINAL, TRAIN_IMAGES_DIR_FINAL, "Train")
    else:
         print("\nAucune image d'entraînement à pivoter selon le CSV filtré.")
         # Assurer que le dossier destination existe même si rien n'est pivoté
         os.makedirs(TRAIN_IMAGES_DIR_FINAL, exist_ok=True)


    processed_test_rot = 0
    if filtered_test_count > 0 and os.path.isdir(TEST_IMAGES_DIR_ORIGINAL):
        processed_test_rot, _ = rotate_images_from_csv(TEST_CSV_PATH_FILTERED, TEST_IMAGES_DIR_ORIGINAL, TEST_IMAGES_DIR_CORRECTED, "Test")
    elif not os.path.isdir(TEST_IMAGES_DIR_ORIGINAL):
        print(f"\nDossier source test '{TEST_IMAGES_DIR_ORIGINAL}' non trouvé. Rotation des images test sautée.")
    else:
        print("\nAucune image de test à pivoter selon le CSV filtré.")
         # Assurer que le dossier destination existe même si rien n'est pivoté
        os.makedirs(TEST_IMAGES_DIR_CORRECTED, exist_ok=True)


    # --- Étape 6: Copier les images d'entraînement restantes (non pivotées, non validation) ---
    print("\n" + "="*20 + " ÉTAPE 6: COPIE DES IMAGES TRAIN RESTANTES " + "="*20)
    copied_non_rotated_count = copy_non_rotated_train_images(
        TRAIN_CSV_PATH_ORIGINAL,
        TRAIN_CSV_PATH_FILTERED,
        val_ids, # Utilise la liste des IDs de validation extraite plus tôt
        TRAIN_IMAGES_DIR_ORIGINAL,
        TRAIN_IMAGES_DIR_FINAL # Copie dans le même dossier final que les images pivotées
    )

    # --- Résumé Final ---
    print("\n" + "="*70)
    print(" RÉSUMÉ DU WORKFLOW DE PRÉPARATION DES DONNÉES (v2)")
    print("="*70)
    print(f"- Dossier Images Train Originales: {TRAIN_IMAGES_DIR_ORIGINAL}")
    print(f"- Dossier Images Test Originales: {TEST_IMAGES_DIR_ORIGINAL}")
    print(f"- Dossier Images Validation Finales: {VALIDATION_IMAGES_DIR}")
    print(f"- Dossier Images Train Finales (Pivotées + Non-Pivotées): {TRAIN_IMAGES_DIR_FINAL}")
    print(f"- Dossier Images Test Corrigées: {TEST_IMAGES_DIR_CORRECTED}")
    print("-" * 70)
    print(f"- CSV Train filtré : {filtered_train_count} images identifiées pour rotation.")
    print(f"- CSV Test filtré : {filtered_test_count} images identifiées pour rotation.")
    print(f"- IDs de validation extraits du JSON : {total_val_ids}")
    print(f"- Images de validation copiées vers '{os.path.basename(VALIDATION_IMAGES_DIR)}' : {copied_val_count}")
    print(f"- Images de validation supprimées de '{os.path.basename(TRAIN_IMAGES_DIR_ORIGINAL)}' : {deleted_val_count}")
    print(f"- Images d'entraînement pivotées vers '{os.path.basename(TRAIN_IMAGES_DIR_FINAL)}' : {processed_train_rot}")
    print(f"- Images de test pivotées vers '{os.path.basename(TEST_IMAGES_DIR_CORRECTED)}' : {processed_test_rot}")
    print(f"- Images d'entraînement non-pivotées/non-validation copiées vers '{os.path.basename(TRAIN_IMAGES_DIR_FINAL)}' : {copied_non_rotated_count}")
    print("-" * 70)
    try:
        final_train_count = len(os.listdir(TRAIN_IMAGES_DIR_FINAL))
        print(f"- Nombre total d'images dans le dossier train final '{os.path.basename(TRAIN_IMAGES_DIR_FINAL)}': {final_train_count}")
        # Vérification rapide : final_train_count devrait être proche de (processed_train_rot + copied_non_rotated_count)
        # Les petites différences peuvent venir d'erreurs de copie, de fichiers manquants, etc.
        expected_train_count = processed_train_rot + copied_non_rotated_count
        if final_train_count != expected_train_count:
             print(f"  (Attendu basé sur traitement: {expected_train_count}. La différence peut indiquer des pbs de copie/source.)")
    except FileNotFoundError:
        print(f"- Dossier train final '{os.path.basename(TRAIN_IMAGES_DIR_FINAL)}' non trouvé ou vide.")

    try:
        final_val_count = len(os.listdir(VALIDATION_IMAGES_DIR))
        print(f"- Nombre total d'images dans le dossier validation final '{os.path.basename(VALIDATION_IMAGES_DIR)}': {final_val_count}")
    except FileNotFoundError:
        print(f"- Dossier validation final '{os.path.basename(VALIDATION_IMAGES_DIR)}' non trouvé ou vide.")

    # --- Affichage des Images Manquantes ---
    print("\n" + "="*20 + " IMAGES MANQUANTES " + "="*20)
    total_missing = len(missing_files_rotation) + len(missing_files_non_rotated)
    print(f"Nombre total d'images sources manquantes : {total_missing}")

    if missing_files_rotation:
        print("\n--- Images manquantes lors de la rotation : ---")
        for file_path in missing_files_rotation:
            print(f"- {file_path}")

    if missing_files_non_rotated:
        print("\n--- Images manquantes lors de la copie des non-pivotées : ---")
        for file_path in missing_files_non_rotated:
            print(f"- {file_path}")

    print("\n--- Workflow terminé ---")