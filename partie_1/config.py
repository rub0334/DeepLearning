import torch
import os

# --- Configuration Générale ---
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PROJECT_DIR = r"C:\Users\rubde\Documents\GitHub\Deep_Learning\DeepLearning\partie_1" # Adaptez si nécessaire

# --- Configuration des Données (TextOCR Localisé) ---
DATA_BASE_PATH = r"C:\Users\rubde\Documents\deep_learning" # Chemin racine des données
TRAIN_JSON = os.path.join(DATA_BASE_PATH, "TextOCR_0.1_train.json")
VAL_JSON = os.path.join(DATA_BASE_PATH, "TextOCR_0.1_val.json")
TRAIN_IMG_DIR = os.path.join(DATA_BASE_PATH, "train_images")
VAL_IMG_DIR = os.path.join(DATA_BASE_PATH, "validation_images")

# --- Hyperparamètres d'Entraînement (Détection) ---
IMG_SIZE = (640, 640) # Taille cible pour le redimensionnement des images
BATCH_SIZE = 4       # Ajustez selon la mémoire GPU disponible
NUM_WORKERS = 2       # Nombre de processus pour le chargement des données (ajustez)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 22       # Nombre d'époques (à ajuster)
WEIGHT_DECAY = 1e-5
USE_AMP = True        # Utiliser la précision mixte (Automatic Mixed Precision)

# --- Configuration Checkpointing ---
CHECKPOINT_DIR = os.path.join(BASE_PROJECT_DIR, "checkpoints")
SAVE_BEST_ONLY = True # Sauvegarder uniquement le meilleur modèle basé sur la validation

# --- Configuration Modèle Détection ---
# Pour un modèle "from scratch", nous aurons besoin de définir les détails architecturaux ici si nécessaire
# Par exemple, nombre de classes (1 pour texte + 1 pour background implicite)
NUM_DETECTION_CLASSES = 2 # 1: Texte, 0: Background (implicite géré par certains modèles)

# --- Autres ---
# Pourraient être ajoutés: paramètres d'optimiseur spécifiques, scheduler, etc.
# config.py
# ...
# --- Hyperparamètres Perte Détection FPN ---
FPN_POS_WEIGHT_CLS = 10.0 # Poids pour les positifs dans la perte de classification
FPN_LAMBDA_BBOX = 1.0    # Poids pour la perte de régression
GRAD_CLIP_NORM = 1.0     # Valeur max pour la norme du gradient