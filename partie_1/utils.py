# utils.py
import torch
import random
import numpy as np
import os
import config # Importer notre configuration
from collections import OrderedDict # Ajout pour DataParallel

def set_seed(seed=config.RANDOM_SEED):
    """Fixe les graines aléatoires pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Attention: Peut ralentir l'entraînement mais améliore la reproductibilité
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Benchmark à False améliore la reproductibilité
    print(f"Random seed set to {seed}")

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Sauvegarde l'état de l'entraînement.

    Args:
        state (dict): Dictionnaire contenant l'état à sauvegarder (epoch, state_dict, optimizer,
                      best_f1_score, lr_scheduler, etc.).
        filename (str): Nom du fichier de checkpoint.
    """
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    print(f"=> Saving checkpoint to {filepath}")
    # Utiliser cpu() pour éviter les problèmes de chargement entre devices si nécessaire,
    # mais généralement .pth.tar peut être chargé sur n'importe quel device avec map_location.
    torch.save(state, filepath)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Charge un checkpoint et retourne l'époque de départ et la meilleure métrique (F1 si disponible).

    Args:
        checkpoint_path (str): Chemin vers le fichier de checkpoint.
        model (torch.nn.Module): Modèle dans lequel charger les poids.
        optimizer (torch.optim.Optimizer, optional): Optimiseur dont charger l'état.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler dont charger l'état.

    Returns:
        tuple: (start_epoch, best_metric) où
               - start_epoch (int): L'époque à laquelle reprendre l'entraînement (+1 par rapport à celle sauvegardée).
               - best_metric (float): La meilleure métrique F1-Score enregistrée (ou 0.0 si non trouvée).
    """
    if not os.path.exists(checkpoint_path):
        print(f"=> No checkpoint found at '{checkpoint_path}'")
        # Retourner 0.0 comme meilleur F1 score par défaut si on commence de zéro
        return 0, 0.0

    print(f"=> Loading checkpoint '{checkpoint_path}'")
    # Charger sur le device configuré
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

    # --- Chargement de l'état du modèle ---
    state_dict = checkpoint.get('state_dict')
    if not state_dict:
        print("ERREUR: Checkpoint ne contient pas 'state_dict'. Impossible de charger le modèle.")
        return 0, 0.0

    # Gérer le préfixe 'module.' ajouté par DataParallel ou DDP
    if list(state_dict.keys())[0].startswith('module.'):
        print("   Detected model saved with DataParallel/DDP, removing 'module.' prefix.")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    print("   Model state loaded successfully.")

    # --- Chargement de l'état de l'optimiseur ---
    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("   Optimizer state loaded successfully.")
            # Déplacer l'état de l'optimiseur sur le bon device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(config.DEVICE)
        except Exception as e:
            print(f"   WARNING: Could not load optimizer state: {e}. Optimizer starts from scratch.")
    elif optimizer:
         print("   Optimizer state not found in checkpoint. Optimizer starts from scratch.")

    # --- Chargement de l'état du Scheduler ---
    if scheduler and 'lr_scheduler' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("   Scheduler state loaded successfully.")
        except Exception as e:
             print(f"   WARNING: Could not load scheduler state: {e}. Scheduler starts from scratch.")
    elif scheduler:
        print("   Scheduler state not found in checkpoint. Scheduler starts from scratch.")

    # --- Récupération de l'époque et de la meilleure métrique ---
    start_epoch = checkpoint.get('epoch', -1) + 1 # +1 pour commencer l'époque *suivante*
    # Prioriser la nouvelle métrique 'best_f1_score'
    # Mettre 0.0 comme défaut pour F1 score
    best_f1_score = checkpoint.get('best_f1_score', 0.0)

    # (Optionnel: Si vous voulez aussi gérer l'ancienne métrique 'best_metric' pour la compatibilité)
    # if 'best_f1_score' not in checkpoint and 'best_metric' in checkpoint:
    #    print("   WARNING: Using old 'best_metric' from checkpoint as F1 score was not found.")
    #    best_f1_score = checkpoint.get('best_metric', 0.0) # Assumer que l'ancienne métrique était un score (pas une perte)

    print(f"=> Loaded checkpoint '{checkpoint_path}' (resume from epoch {start_epoch}, best F1 score recorded: {best_f1_score:.4f})")
    return start_epoch, best_f1_score # Retourne l'époque de début et le meilleur F1


# --- Fonctions spécifiques aux données TextOCR (inchangées) ---

def get_bounding_box_from_points(points):
    """
    Calcule la boîte englobante horizontale (xmin, ymin, xmax, ymax)
    à partir d'une liste de points [x1, y1, x2, y2, ...].
    """
    if not points or len(points) < 2:
        return None

    # Convertir en numpy array et trouver min/max pour chaque axe
    pts_array = np.array(points).reshape(-1, 2)
    x_min, y_min = pts_array.min(axis=0)
    x_max, y_max = pts_array.max(axis=0)

    # Vérifier que la boîte a une aire > 0 (pas dégénérée)
    if x_max <= x_min or y_max <= y_min:
       # Commenté pour éviter trop de logs si ça arrive souvent
       # print(f"Warning: Degenerate bounding box calculated from points {points}. Min/Max: ({x_min}, {y_min}), ({x_max}, {y_max})")
       return None # Retourner None pour indiquer une boîte invalide

    return [float(x_min), float(y_min), float(x_max), float(y_max)] # Assurer float pour compatibilité tenseurs

def get_horizontal_bbox_from_textocr_bbox(textocr_bbox):
    """
    Convertit le format TextOCR [xmin, ymin, width, height]
    en [xmin, ymin, xmax, ymax].
    """
    x_min, y_min, width, height = textocr_bbox
    # Vérifier validité
    if width <= 0 or height <= 0:
       # print(f"Warning: Degenerate bbox from TextOCR format {textocr_bbox}")
       return None # Retourner None pour indiquer une boîte invalide
    x_max = x_min + width
    y_max = y_min + height
    # Assurer float pour compatibilité tenseurs
    return [float(x_min), float(y_min), float(x_max), float(y_max)]

# On pourrait ajouter ici des fonctions de visualisation plus tard