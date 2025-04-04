import torch
import random
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def set_seed(seed=42):
    """Fixe les graines aléatoires pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Les opérations cudnn peuvent être non déterministes, ces lignes aident mais ne garantissent pas à 100%
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar', best_filename='model_best_detection.pth.tar'):
    """Sauvegarde l'état de l'entraînement."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)
        print(f"Best model saved to {best_filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, device='cpu'):
    """Charge un checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None, 0, float('inf') # Renvoie None pour l'état, époque 0, et perte infinie

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Gestion des modèles DataParallel/DistributedDataParallel
    state_dict = checkpoint['state_dict']
    # Supprimer le préfixe 'module.' si le modèle a été sauvegardé avec DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("Model weights loaded successfully.")

    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer state loaded successfully.")
        except Exception as e:
             print(f"Could not load optimizer state: {e}. Starting optimizer from scratch.")


    if scaler and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
        try:
            scaler.load_state_dict(checkpoint['scaler'])
            print("AMP Scaler state loaded successfully.")
        except Exception as e:
            print(f"Could not load AMP Scaler state: {e}. Initializing new scaler.")


    print(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}, Best validation loss so far: {best_val_loss:.4f}")
    return model, optimizer, scaler, start_epoch, best_val_loss


def polygon_to_bbox(points):
    """Convertit une liste de points [x1, y1, x2, y2, ...] en une bbox [xmin, ymin, xmax, ymax]."""
    if not points or len(points) < 2:
        return None
    xs = points[0::2]
    ys = points[1::2]
    if not xs or not ys: # Au cas où il n'y aurait qu'un seul point
        return None
    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)
    # Vérification de validité (largeur et hauteur > 0)
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax, ymax]

def visualize_detection(image_path, targets, predictions=None, output_path=None, score_threshold=0.5):
    """Visualise les boîtes englobantes (ground truth et/ou prédictions) sur une image."""
    try:
        img = Image.open(image_path).convert("RGB")
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img)
        plt.axis('off')

        # Afficher les boîtes Ground Truth (en vert)
        if targets and 'boxes' in targets:
            for box in targets['boxes']:
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)

        # Afficher les boîtes Prédites (en rouge)
        if predictions and 'boxes' in predictions:
             scores = predictions.get('scores', [1.0] * len(predictions['boxes'])) # Default score 1 if not provided
             for i, box in enumerate(predictions['boxes']):
                 if scores[i] >= score_threshold:
                     xmin, ymin, xmax, ymax = box
                     rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                              linewidth=2, edgecolor='r', facecolor='none')
                     ax.add_patch(rect)
                     if 'labels' in predictions: # Optionnel: afficher label/score
                          label = predictions['labels'][i]
                          score = scores[i]
                          ax.text(xmin, ymin - 5, f'{label}: {score:.2f}', color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5, pad=0))


        if output_path:
            if not os.path.exists(os.path.dirname(output_path)):
                 os.makedirs(os.path.dirname(output_path))
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        plt.close(fig)

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"Error during visualization: {e}")

# --- Fonctions de Métriques (Simplifiées pour le début) ---
# Pour la détection, une métrique clé est l'Intersection over Union (IoU)
def calculate_iou(box1, box2):
    """Calcule l'Intersection over Union (IoU) entre deux boîtes [xmin, ymin, xmax, ymax]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

# Note: Une évaluation complète (mAP) nécessiterait une logique plus complexe
# pour apparier prédictions et ground truths basées sur l'IoU et calculer Precision/Recall.