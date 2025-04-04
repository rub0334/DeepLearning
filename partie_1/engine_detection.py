# engine_detection.py
import torch
import math
import sys
from tqdm import tqdm
import config # Importer notre configuration
from torchvision.ops import box_iou # Outil essentiel pour calculer l'IoU

# Retirer ou commenter l'ancienne fonction evaluate basée sur la perte
def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    """
    Exécute une époque d'entraînement.
    (Copiez le reste de la fonction depuis la réponse précédente si elle manque)
    """
    model.train() # Mettre le modèle en mode entraînement
    total_loss = 0.0
    num_batches = len(data_loader)

    # Barre de progression
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]", unit="batch")

    for i, (images, targets) in enumerate(pbar):
        # Déplacer les images et les cibles vers le bon device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # --- Calcul avec Précision Mixte (si activé) ---
        if scaler:
            # Utiliser la nouvelle API torch.amp
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Faster R-CNN retourne un dictionnaire de pertes en mode entraînement
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Vérifier si la perte est valide
            if not math.isfinite(losses.item()):
                print(f"\nERREUR: Perte infinie/NaN détectée à l'itération {i} (AMP). Arrêt.")
                print(f"Loss dict: {loss_dict}")
                for k, v in loss_dict.items():
                    print(f"  {k}: {v.item()}")
                sys.exit(1) # Arrêter l'entraînement

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses).backward()
            # scaler.unscale_(optimizer) # Décommenter si utilisation de clip_grad_norm_
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        # --- Calcul sans Précision Mixte ---
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not math.isfinite(losses.item()):
                print(f"\nERREUR: Perte infinie détectée à l'itération {i}. Arrêt.")
                print(f"Loss dict: {loss_dict}")
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Décommenter si clip_grad_norm_
            optimizer.step()

        batch_loss = losses.item()
        total_loss += batch_loss

        # Mise à jour de la barre de progression
        pbar.set_postfix(loss=f"{batch_loss:.4f}", avg_loss=f"{total_loss / (i + 1):.4f}")

    avg_epoch_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} [Train] - Average Loss: {avg_epoch_loss:.4f}") # Correction affichage ici
    return avg_epoch_loss


@torch.no_grad() # TRÈS IMPORTANT: Désactiver les gradients pour l'évaluation
def evaluate_metrics(model, data_loader, device, epoch):
    """
    Évalue le modèle sur le set de validation et calcule les métriques de détection.

    Args:
        model (torch.nn.Module): Le modèle à évaluer.
        data_loader (DataLoader): DataLoader pour les données de validation.
        device (torch.device): Device ('cuda' ou 'cpu').
        epoch (int): Numéro de l'époque actuelle (pour l'affichage).

    Returns:
        dict: Dictionnaire contenant les métriques calculées:
              {'precision': float, 'recall': float, 'f1_score': float,
               'TP': int, 'FP': int, 'FN': int}
    """
    model.eval() # Mettre le modèle en mode ÉVALUATION pour obtenir les prédictions

    total_tp = 0
    total_fp = 0
    total_fn = 0
    n_batches = len(data_loader)

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Evaluate Metrics]", unit="batch")

    for i, (images, targets) in enumerate(pbar):
        images = list(image.to(device) for image in images)
        # Les cibles restent sur CPU pour comparaison après prédiction, ou déplacer si besoin
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets] # Pas nécessaire pour l'instant

        # Obtenir les prédictions du modèle
        # Utiliser AMP pour l'inférence si activé (peut accélérer)
        # Note: l'inférence AMP est moins critique que l'entraînement AMP
        # if device == torch.device("cuda"):
        #      with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        #          outputs = model(images)
        # else:
        #      outputs = model(images) # Sortie est une liste de dict par image dans le batch
        outputs = model(images) # Sortie est une liste de dict par image dans le batch


        # Comparer prédictions et vérité terrain pour chaque image du batch
        for j in range(len(outputs)):
            preds = outputs[j] # Dict: {'boxes': Tensor[N,4], 'labels': Tensor[N], 'scores': Tensor[N]}
            target = targets[j] # Dict: {'boxes': Tensor[M,4], 'labels': Tensor[M], ...}

            # Extraire les prédictions au-dessus du seuil de confiance
            pred_scores = preds['scores']
            conf_mask = pred_scores >= config.EVAL_CONFIDENCE_THRESHOLD
            pred_boxes_filtered = preds['boxes'][conf_mask]
            # pred_labels_filtered = preds['labels'][conf_mask] # On n'a qu'une classe (1), mais utile si plusieurs classes
            pred_scores_filtered = pred_scores[conf_mask]
            num_preds = pred_boxes_filtered.shape[0]

            # Extraire les boîtes de vérité terrain (ground truth)
            gt_boxes = target['boxes'].to(device) # Déplacer GT sur GPU pour calcul IoU
            # gt_labels = target['labels'] # Utile si plusieurs classes
            num_gt = gt_boxes.shape[0]

            image_tp = 0
            image_fp = 0
            image_fn = 0

            if num_preds == 0:
                # Aucune prédiction détectée, toutes les GT sont des Faux Négatifs
                image_fn = num_gt
            elif num_gt == 0:
                # Aucune vérité terrain, toutes les prédictions sont des Faux Positifs
                image_fp = num_preds
            else:
                # Calculer la matrice IoU entre prédictions filtrées et GT
                # Shape: [num_preds, num_gt]
                iou_matrix = box_iou(pred_boxes_filtered, gt_boxes)

                # --- Logique de Matching (Greedy basée sur le score) ---
                # Marqueurs pour savoir si une GT ou une prédiction a déjà été associée
                gt_matched = torch.zeros(num_gt, dtype=torch.bool, device=device)
                pred_matched = torch.zeros(num_preds, dtype=torch.bool, device=device)

                # Trier les prédictions par score descendant
                indices = torch.argsort(pred_scores_filtered, descending=True)

                for pred_idx in indices:
                    if pred_matched[pred_idx]: # Si déjà matchée (ne devrait pas arriver avec ce tri mais sécurité)
                        continue

                    # Trouver la meilleure GT pour cette prédiction
                    overlaps = iou_matrix[pred_idx] # IoUs de cette prédiction avec toutes les GTs
                    best_gt_match_iou, best_gt_match_idx = overlaps.max(dim=0)

                    # Vérifier si le match est valide (IoU > seuil ET la GT n'est pas déjà matchée)
                    if best_gt_match_iou >= config.EVAL_IOU_THRESHOLD and not gt_matched[best_gt_match_idx]:
                        # C'est un Vrai Positif (TP)
                        image_tp += 1
                        pred_matched[pred_idx] = True
                        gt_matched[best_gt_match_idx] = True
                    # Sinon (IoU trop bas ou GT déjà prise), cette prédiction est un Faux Positif (FP)
                    # On ne l'incrémente pas ici, on le fera à la fin

                # Calculer FP: nombre de prédictions non matchées
                image_fp = (~pred_matched).sum().item()
                # Calculer FN: nombre de GT non matchées
                image_fn = (~gt_matched).sum().item()
                # --- Fin Logique de Matching ---

            # Accumuler les comptes pour tout le dataset
            total_tp += image_tp
            total_fp += image_fp
            total_fn += image_fn

    # Calculer les métriques globales à la fin de l'époque
    # Gérer les divisions par zéro
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\nEpoch {epoch+1} [Evaluate Metrics Results]:")
    print(f"  Confidence Threshold: {config.EVAL_CONFIDENCE_THRESHOLD}, IoU Threshold: {config.EVAL_IOU_THRESHOLD}")
    print(f"  True Positives (TP): {total_tp}")
    print(f"  False Positives (FP): {total_fp}")
    print(f"  False Negatives (FN): {total_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1_score:.4f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn
    }

# Ne pas oublier de commenter ou supprimer l'ancienne fonction 'evaluate' si elle existe encore.