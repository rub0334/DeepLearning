# config.py

import torch

# --- Paths ---
ANNOTATION_FILE = "data/TextOCR_0.1_train.json"
IMAGE_DIR = "data/train_val_images/train_images" # Ajuster si nécessaire
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "output_visualizations"

# --- Data Preprocessing ---
IMG_HEIGHT = 64

# --- Model Architecture ---
CNN_OUTPUT_CHANNELS = 512
RNN_HIDDEN_SIZE = 256
RNN_LAYERS = 2
# NUM_CLASSES sera défini dynamiquement

# --- Training ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 16 # Ajuster selon la mémoire GPU
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
CLIP_GRAD_NORM = 5.0
VALIDATION_SPLIT = 0.1 # % pour la validation (calculé sur le dataset COMPLET)
SEED = 42

# --- Progressive Data Increase Parameters ---
# Activer ou désactiver l'augmentation progressive
ENABLE_PROGRESSIVE_LOADING = True # Mettre à False pour utiliser tout le jeu d'entraînement dès le début
# Nombre initial d'échantillons d'entraînement
INITIAL_TRAIN_SAMPLES = 50000
# Augmentation du nombre d'échantillons à chaque étape
SAMPLE_INCREASE_STEP = 10000
# Augmenter la taille toutes les N époques
INCREASE_EVERY_N_EPOCHS = 2
# Maximum (sera plafonné à la taille réelle du jeu d'entraînement de toute façon)
MAX_TRAIN_SAMPLES_CAP = 1000000 # Mettre une très grande valeur ou None pour ne pas limiter artificiellement

# --- CTC ---
BLANK_TOKEN = '-'

# --- Evaluation ---
EVAL_BATCH_SIZE = 32 # Utiliser une taille de batch potentiellement plus grande pour l'évaluation