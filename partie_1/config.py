# config.py
import torch
import os

# --- Configuration Générale ---
PROJECT_NAME = "TextOCR_Detection_From_Scratch"
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Répertoire racine du projet partie_1/
OUTPUT_DIR = os.path.join(BASE_DIR) # Dossier de sortie principal

# --- Configuration des Données ---
# !! IMPORTANT: Adaptez ce chemin à votre système !!
DATA_BASE_PATH = r"C:\Users\rubde\Documents\deep_learning" # Chemin racine fourni

TRAIN_ANNOTATION_FILE = os.path.join(DATA_BASE_PATH, "TextOCR_0.1_train.json")
VAL_ANNOTATION_FILE = os.path.join(DATA_BASE_PATH, "TextOCR_0.1_val.json")
TRAIN_IMAGE_DIR = os.path.join(DATA_BASE_PATH, "train_images")
VAL_IMAGE_DIR = os.path.join(DATA_BASE_PATH, "validation_images") # Corrigé pour correspondre aux instructions

# --- Configuration de l'Entraînement ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 25 # À ajuster en fonction de la convergence
BATCH_SIZE = 2    # À ajuster en fonction de la mémoire GPU
LEARNING_RATE = 0.001 # Taux d'apprentissage initial
WEIGHT_DECAY = 0.0005 # Régularisation L2
LR_STEP_SIZE = 8     # Réduire le LR toutes les N époques
LR_GAMMA = 0.1       # Facteur de réduction du LR

# --- Configuration du Modèle ---
# Nous utilisons une seule classe pour la détection : "text"
NUM_CLASSES = 2 # 1 classe ('text') + 1 classe ('background')

# --- Configuration de la Reproductibilité ---
RANDOM_SEED = 42

# --- Configuration du Checkpointing ---
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAVE_FREQ = 1 # Sauvegarder le checkpoint après chaque époque

# --- Configuration des Visualisations (Optionnel) ---
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# --- Configuration de l'Évaluation ---
# Seuil de confiance : Prédictions avec un score inférieur seront ignorées
EVAL_CONFIDENCE_THRESHOLD = 0.5
# Seuil IoU : Une prédiction doit avoir un IoU >= à ce seuil avec une GT pour être un TP potentiel
EVAL_IOU_THRESHOLD = 0.5

# Crée les dossiers nécessaires s'ils n'existent pas
# Ces opérations sont idempotentes et peuvent rester ici sans problème
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# --- SUPPRESSION DU BLOC PRINT CI-DESSOUS ---
# print(f"Configuration chargée:")
# print(f"  Device: {DEVICE}")
# print(f"  Data Base Path: {DATA_BASE_PATH}")
# print(f"  Checkpoint Dir: {CHECKPOINT_DIR}")
# print(f"  Num Epochs: {NUM_EPOCHS}")
# print(f"  Batch Size: {BATCH_SIZE}")
# print(f"  Learning Rate: {LEARNING_RATE}")
# --- FIN SUPPRESSION ---