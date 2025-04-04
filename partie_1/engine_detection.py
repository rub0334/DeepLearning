# engine_detection.py

import torch
import math
import sys
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional # Added typing

# Import pour la perte
import torch.nn.functional as F
# from torchvision.ops import box_iou # Keep if needed for advanced matching later

import config
# from utils import calculate_iou # Keep if needed for detailed eval later

# --- Configuration (ajout pour clarté, valeurs à définir dans config.py) ---
FPN_POS_WEIGHT_CLS = getattr(config, 'FPN_POS_WEIGHT_CLS', 10.0)
FPN_LAMBDA_BBOX = getattr(config, 'FPN_LAMBDA_BBOX', 1.0)
GRAD_CLIP_NORM = getattr(config, 'GRAD_CLIP_NORM', 1.0) # Ajout de Grad clip norm ici

# =============================================================================
# == NOUVELLES FONCTIONS DE PERTE POUR LE MODÈLE FPN ==========================
# =============================================================================

def get_grid_centers(grid_size: Tuple[int, int], stride: int, device: torch.device) -> torch.Tensor:
    """Crée les coordonnées des centres des cellules de la grille (format x, y)."""
    H_grid, W_grid = grid_size
    shifts_y = torch.arange(0, H_grid, dtype=torch.float32, device=device) * stride
    shifts_x = torch.arange(0, W_grid, dtype=torch.float32, device=device) * stride
    # Créer la grille. Note: + stride / 2 pour obtenir le CENTRE de la cellule
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    # Aplatir et ajouter le décalage du centre
    centers_x = (shift_x + stride / 2).flatten() # Shape [H*W]
    centers_y = (shift_y + stride / 2).flatten() # Shape [H*W]
    # Retourner les centres [H*W, 2] format (x, y)
    grid_centers = torch.stack((centers_x, centers_y), dim=1)
    return grid_centers

# MODIFIED SIGNATURE: Added device argument
def map_targets_to_fpn_grids_simple(targets: List[Dict[str, torch.Tensor]],
                                    fpn_strides: List[int],
                                    feature_map_shapes: List[Tuple[int, int, int, int]], # NCHW shapes from preds
                                    device: torch.device # <<< AJOUTÉ : device cible pour les tenseurs créés
                                    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    ÉBAUCHE TRES SIMPLIFIÉE de mapping pour FPN.
    Assign TOUS les GT boxes à la cellule contenant leur centre sur CHAQUE niveau FPN.
    Calcule les cibles de régression (dx, dy, dw, dh) relatives au centre de cette cellule.

    Args:
        targets: Liste de dictionnaires de cibles GT (doivent être déjà sur le bon device).
        fpn_strides: Strides de chaque niveau FPN (ex: [8, 16, 32]).
        feature_map_shapes: Formes (N, C, H, W) des logits/bbox_preds de chaque niveau.
        device: Le device sur lequel créer les tenseurs cibles.

    Returns:
        Tuple[List[Tensor], List[Tensor]]:
            - all_level_target_cls: Liste de tenseurs cibles de classe [B, H*W] par niveau.
            - all_level_target_reg: Liste de tenseurs cibles de régression [B, H*W, 4] par niveau.

    LIMITATIONS MAJEURES:
        - Ne fait pas d'assignation basée sur la taille (un GT devrait idéalement être assigné
          à un seul niveau FPN).
        - Ne gère pas les GT dont le centre tombe sur la même cellule (le dernier écrase).
        - Méthode d'assignation non robuste. À REMPLACER par une méthode FCOS, ATSS, ou basée sur ancre/IoU.
    """
    all_level_target_cls = []
    all_level_target_reg = []
    # <<< SUPPRIMÉ: Inférence du device depuis targets. Utilisation du 'device' passé en argument.

    batch_size = len(targets)

    for level_idx, stride in enumerate(fpn_strides):
        # Obtenir H, W de la feature map à partir des shapes fournies
        N, _, H_grid, W_grid = feature_map_shapes[level_idx]
        grid_size = (H_grid, W_grid)
        num_locs = H_grid * W_grid

        # Initialiser les cibles pour ce niveau SUR LE BON DEVICE
        level_target_cls = torch.zeros((batch_size, num_locs), dtype=torch.float32, device=device)
        level_target_reg = torch.zeros((batch_size, num_locs, 4), dtype=torch.float32, device=device)

        # Obtenir les centres de grille pour ce niveau SUR LE BON DEVICE
        grid_centers = get_grid_centers(grid_size, stride, device) # [NumLocs, 2] (x, y)

        for i in range(batch_size):
            # On suppose que les targets[i]['boxes'] sont déjà sur le bon 'device'
            # grâce au code appelant (train_one_epoch/evaluate)
            gt_boxes = targets[i]['boxes'] # [N_boxes, 4] en xmin, ymin, xmax, ymax
            if gt_boxes.shape[0] == 0:
                continue

            # Coordonnées et dimensions GT (seront sur le même device que gt_boxes)
            gt_centers_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
            gt_centers_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
            gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
            gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]

            # Trouver les indices de grille correspondants aux centres GT
            # Ces calculs se feront sur le device de gt_centers_x/y
            grid_x_indices = (gt_centers_x / stride).long().clamp(0, W_grid - 1)
            grid_y_indices = (gt_centers_y / stride).long().clamp(0, H_grid - 1)
            # Convertir en indice plat
            flat_indices = grid_y_indices * W_grid + grid_x_indices # Shape [N_boxes]

            # Marquer ces cellules comme positives (cible classe = 1.0)
            # Indexation de level_target_cls (sur device) avec flat_indices (sur device)
            level_target_cls[i, flat_indices] = 1.0

            # Calculer les cibles de régression (dx, dy, dw, dh) relatives au centre de la cellule
            # Convention type FCOS/YOLO:
            # dx = (gt_center_x - cell_center_x) / stride
            # dy = (gt_center_y - cell_center_y) / stride
            # dw = log(gt_width / stride)
            # dh = log(gt_height / stride)

            # Indexation de grid_centers (sur device) avec flat_indices (sur device)
            assigned_cell_centers = grid_centers[flat_indices] # [N_boxes, 2]

            delta_x = (gt_centers_x - assigned_cell_centers[:, 0]) / stride
            delta_y = (gt_centers_y - assigned_cell_centers[:, 1]) / stride
            # Utiliser log. Clamp pour éviter log(0) ou log(neg) si largeur/hauteur est invalide
            delta_w = torch.log((gt_widths.clamp(min=1e-6)) / stride)
            delta_h = torch.log((gt_heights.clamp(min=1e-6)) / stride)

            target_deltas = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=1) # [N_boxes, 4]

            # Assigner les deltas cibles aux cellules correspondantes
            # Indexation de level_target_reg (sur device) avec flat_indices (sur device)
            # Attention: Si plusieurs GT tombent dans la même cellule, le dernier écrase !
            level_target_reg[i, flat_indices, :] = target_deltas

        all_level_target_cls.append(level_target_cls) # [Batch, NumLocs_level]
        all_level_target_reg.append(level_target_reg) # [Batch, NumLocs_level, 4]

    return all_level_target_cls, all_level_target_reg

# MODIFIED SIGNATURE: Added device argument
def compute_fpn_detection_loss(all_cls_logits: List[torch.Tensor], # Liste [N, HxW, NumCls]
                               all_bbox_pred: List[torch.Tensor],  # Liste [N, HxW, 4] (dx,dy,dw,dh)
                               targets: List[Dict[str, torch.Tensor]],
                               fpn_strides: List[int],
                               device: torch.device, # <<< AJOUTÉ : device principal pour les calculs
                               pos_weight: float = 10.0,
                               lambda_bbox: float = 1.0) -> Optional[Dict[str, torch.Tensor]]:
    """
    Calcule la perte pour le détecteur FPN en utilisant un mapping SIMPLIFIÉ.

    Args:
        all_cls_logits: Sorties de classification de chaque niveau FPN (sur 'device').
        all_bbox_pred: Sorties de régression de boîte de chaque niveau FPN (sur 'device').
        targets: Liste de cibles GT (sur 'device').
        fpn_strides: Strides de chaque niveau FPN.
        device: Le device sur lequel les calculs principaux doivent être effectués.
        pos_weight: Poids pour les positifs dans la perte BCE.
        lambda_bbox: Poids pour la perte de régression.

    Returns:
        Dictionnaire contenant les pertes calculées et le nombre de positifs,
        ou None si une erreur se produit (ex: pas de cibles). Les pertes retournées
        seront sur le 'device' spécifié.
    """
    if not targets: # Gérer le cas d'un batch sans cibles valides
        print("Warning: compute_fpn_detection_loss received empty targets list.")
        # Retourner des pertes nulles sur le bon device, SANS grad_fn
        return {
            "loss_classifier": torch.tensor(0.0, device=device),
            "loss_box_reg": torch.tensor(0.0, device=device),
            "loss_total": torch.tensor(0.0, device=device),
            "num_positives": 0.0
        }

    # Les prédictions (all_cls_logits, all_bbox_pred) et les targets sont supposées être sur 'device'
    # On va déduire les shapes et appeler le mapping en lui passant le 'device'

    feature_map_shapes_approx = []
    # Utilisation du device passé en argument pour les tenseurs créés ici
    batch_size_pred = all_cls_logits[0].shape[0] # Taille de batch déduite des prédictions

    # Vérifier la cohérence de la taille de batch avec les cibles
    if batch_size_pred != len(targets):
        print(f"Warning: Mismatch between prediction batch size ({batch_size_pred}) and targets length ({len(targets)}). Skipping loss calculation.")
        # Retourner des pertes nulles sur le bon device, SANS grad_fn
        return {
            "loss_classifier": torch.tensor(0.0, device=device),
            "loss_box_reg": torch.tensor(0.0, device=device),
            "loss_total": torch.tensor(0.0, device=device),
            "num_positives": 0.0
        }


    for level_logits in all_cls_logits:
        N, HxW, NumCls = level_logits.shape
        # Approximation de H et W (suppose un ratio proche de 1)
        H_approx = W_approx = int(math.sqrt(HxW))
        # On simule la shape NCHW pour map_targets
        feature_map_shapes_approx.append((N, NumCls, H_approx, W_approx))


    # --- Étape 1: Mapper les cibles aux grilles FPN (Fonction simplifiée) ---
    try:
        # Passer le device explicitement pour la création des tenseurs cibles
        target_cls_levels, target_reg_levels = map_targets_to_fpn_grids_simple(
            targets, fpn_strides, feature_map_shapes_approx, device # <<< Passé ici
        )
    except Exception as e:
        # Cette erreur ne devrait plus être un device mismatch si les inputs sont corrects
        print(f"Error during target mapping: {e}. Skipping loss calculation for this batch.")
        # import traceback; traceback.print_exc() # Pour débogage avancé
        # Retourner des pertes nulles sur le bon device, SANS grad_fn
        return {
            "loss_classifier": torch.tensor(0.0, device=device),
            "loss_box_reg": torch.tensor(0.0, device=device),
            "loss_total": torch.tensor(0.0, device=device),
            "num_positives": 0.0
        }

    # Les tenseurs target_cls_levels et target_reg_levels sont maintenant garantis d'être sur 'device'

    # Utiliser le device passé en argument pour les tenseurs de perte cumulée
    total_loss_cls = torch.tensor(0.0, device=device)
    total_loss_reg = torch.tensor(0.0, device=device)
    num_positives_total = 0.0 # Utiliser float pour division future

    # --- Étape 2: Calculer la perte pour chaque niveau FPN ---
    for level_idx in range(len(all_cls_logits)):
        pred_cls_logits = all_cls_logits[level_idx] # [N, HxW, NumCls=1] (sur device)
        pred_bbox_deltas = all_bbox_pred[level_idx] # [N, HxW, 4] (sur device)

        target_cls = target_cls_levels[level_idx] # [N, HxW] (sur device)
        target_reg = target_reg_levels[level_idx] # [N, HxW, 4] (sur device)

        # Masque des positifs pour ce niveau [N, HxW] (sur device)
        pos_mask = target_cls > 0.5

        num_positives_level = pos_mask.sum().float() # Convertir en float
        num_positives_total += num_positives_level

        # --- Perte de Classification (BCE sur tous les emplacements) ---
        # Utiliser .squeeze(-1) si NumCls=1
        current_cls_logits = pred_cls_logits.squeeze(-1) # Shape: [N, HxW] (sur device)

        # Le pos_weight doit aussi être sur le bon device
        pos_weight_tensor = torch.tensor([pos_weight], device=device)

        loss_cls_level = F.binary_cross_entropy_with_logits(
            current_cls_logits, # sur device
            target_cls,         # sur device
            reduction='sum',
            pos_weight=pos_weight_tensor # sur device
        ) # loss_cls_level sera sur device

        # --- Perte de Régression (Smooth L1 uniquement sur les positifs) ---
        loss_reg_level = torch.tensor(0.0, device=device) # Initialiser sur device
        if num_positives_level > 0:
            # Indexation avec pos_mask (sur device) donnera des tenseurs sur device
            pred_deltas_pos = pred_bbox_deltas[pos_mask] # [NumPositives_level, 4] (sur device)
            target_deltas_pos = target_reg[pos_mask]     # [NumPositives_level, 4] (sur device)

            # Vérifier les valeurs avant la perte (débogage NaN)
            if torch.isnan(pred_deltas_pos).any() or torch.isinf(pred_deltas_pos).any():
                print(f"Warning: NaN/Inf detected in pred_deltas_pos at level {level_idx}")
            if torch.isnan(target_deltas_pos).any() or torch.isinf(target_deltas_pos).any():
                print(f"Warning: NaN/Inf detected in target_deltas_pos at level {level_idx}")


            loss_reg_level = F.smooth_l1_loss(
                pred_deltas_pos,     # sur device
                target_deltas_pos, # sur device
                reduction='sum',
                beta=1.0 / 9.0 # Beta = 1/9 est commun (vu dans Faster R-CNN)
            ) # loss_reg_level sera sur device

            if torch.isnan(loss_reg_level) or torch.isinf(loss_reg_level):
                 print(f"Warning: loss_reg_level became {loss_reg_level} at level {level_idx}")


        total_loss_cls += loss_cls_level # Somme de tenseurs sur device
        total_loss_reg += loss_reg_level # Somme de tenseurs sur device

    # --- Étape 3: Normaliser les pertes ---
    if num_positives_total > 0:
         # Normaliser cls par total locations OU total positifs ? Normalisons par positifs ici.
         # Normaliser reg par total positifs.
         avg_loss_cls = total_loss_cls / num_positives_total
         avg_loss_reg = total_loss_reg / num_positives_total
    else:
         # Gérer le cas où il n'y a aucun positif sur aucun niveau
         avg_loss_cls = total_loss_cls * 0.0 # Perte nulle (tenseurs sur device * 0.0)
         avg_loss_reg = total_loss_reg * 0.0

    # Combinaison finale (tenseurs sur device)
    total_loss = avg_loss_cls + lambda_bbox * avg_loss_reg

    # Vérifications finales de NaN/Inf
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"\nWarning: Final total_loss is {total_loss} (cls={avg_loss_cls.item()}, reg={avg_loss_reg.item()}, num_pos={num_positives_total})")
        # Option: retourner des pertes nulles pour éviter de crasher ?
        # Retourner des pertes nulles sur le bon device, SANS grad_fn
        return {
            "loss_classifier": torch.tensor(0.0, device=device),
            "loss_box_reg": torch.tensor(0.0, device=device),
            "loss_total": torch.tensor(0.0, device=device), # Retourne une perte nulle pour ce batch
            "num_positives": num_positives_total # Garde le compte
        }

    # Retourner les pertes. 'total_loss' AURA un grad_fn si les inputs en avaient un.
    # Les autres sont détachées pour le logging.
    return {
        "loss_classifier": avg_loss_cls.detach(),
        "loss_box_reg": avg_loss_reg.detach(),
        "loss_total": total_loss, # <<< Important: cette perte a le graphe de calcul attaché
        "num_positives": num_positives_total # Retourner en float
    }


# =============================================================================
# == ANCIENNES FONCTIONS DE PERTE (À SUPPRIMER ou GARDER COMMENTÉES) ==========
# =============================================================================
# def map_targets_to_grid(...): ...
# def compute_detection_loss(...): ... # Cette fonction est maintenant obsolète


# =============================================================================
# == BOUCLES D'ENTRAÎNEMENT ET VALIDATION MISES À JOUR ========================
# =============================================================================

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, grad_clip_norm=GRAD_CLIP_NORM): # Utilisation de scaler et grad_clip
    model.train()
    total_loss = 0.0
    loss_cls_total = 0.0
    loss_reg_total = 0.0
    total_positives = 0.0 # Compter les positifs pour la moyenne
    start_time = time.time()

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    # Récupérer les strides FPN du modèle (suppose qu'elles sont accessibles)
    try:
        # Essayer d'accéder aux strides. S'assurer que c'est bien une liste/tuple de nombres.
        fpn_strides = list(model.strides.values()) # Ex: si model.strides est un dict {level: stride}
        if not all(isinstance(s, (int, float)) for s in fpn_strides):
             raise TypeError("Model strides must be numeric values.")
    except AttributeError:
        print("\nCRITICAL ERROR: Model does not have a 'strides' attribute.")
        print("Please ensure your model instance has an attribute 'strides' (e.g., a dict like {'p3': 8, 'p4': 16, ...} or a list/tuple [8, 16, ...]).")
        print("Stopping training.")
        sys.exit(1) # Arrêter l'entraînement proprement
    except TypeError as e:
        print(f"\nCRITICAL ERROR: Problem with model strides format: {e}")
        print("Please ensure model.strides contains numeric stride values.")
        print("Stopping training.")
        sys.exit(1) # Arrêter l'entraînement proprement


    for batch_idx, batch_data in enumerate(progress_bar):
        # Gérer les batches potentiellement vides retournés par collate_fn
        if batch_data is None:
            print(f"Warning: Skipping None batch returned by collate_fn at index {batch_idx}")
            continue
        images, targets = batch_data
        if images is None or targets is None:
             print(f"Warning: Skipping empty batch content at index {batch_idx}")
             continue

        images = images.to(device)
        # <<< Important: Mettre les cibles sur le bon device AVANT de les passer à la perte
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # --- Calcul de la perte avec la nouvelle fonction FPN ---
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            predictions = model(images) # output = {"cls_logits": List[N,HxW,C], "bbox_pred": List[N,HxW,4]}

            # <<< Passer le 'device' explicitement à la fonction de perte
            losses = compute_fpn_detection_loss(
                all_cls_logits=predictions['cls_logits'],
                all_bbox_pred=predictions['bbox_pred'],
                targets=targets,
                fpn_strides=fpn_strides,
                device=device, # <<< Passé ici
                pos_weight=FPN_POS_WEIGHT_CLS,
                lambda_bbox=FPN_LAMBDA_BBOX
            )

            # Gérer le cas où la perte retourne None ou un dict avec perte nulle (erreur interne)
            if losses is None:
                 print(f"Warning: Loss computation failed unexpectedly for batch {batch_idx}. Skipping batch.")
                 continue

            loss = losses['loss_total'] # Cette perte DOIT avoir un grad_fn si valide

            # Vérifier si la perte est valide (tensor 0.0 peut être retourné en cas d'erreur gérée)
            # On ne backprop que si la perte est > 0 et finie
            # Note: loss peut être 0.0 légitimement (ex: 0 positifs et prédictions parfaites sur négatifs)
            # La vérification isfinite est la plus importante ici.
            is_loss_valid_for_backward = torch.isfinite(loss) and loss.requires_grad

        # --- Vérification de la perte avant Backprop ---
        # Fait maintenant dans le bloc autocast pour utiliser is_loss_valid_for_backward
        if not is_loss_valid_for_backward:
             # loss.item() pourrait échouer si loss n'est pas un scalaire (ne devrait pas arriver ici)
             try:
                 loss_val = loss.item()
             except:
                 loss_val = "N/A (non-scalar or invalid)"
             if not torch.isfinite(loss):
                 print(f"\nWarning: Loss is {loss_val} BEFORE backward at Epoch {epoch+1}, Batch {batch_idx} (NumPos: {losses.get('num_positives', 0)}). Skipping batch.")
             # else: loss is 0.0 or does not require grad, skip backward silently or log if needed
             # Optionnel: Sauvegarder état pour débogage
             # torch.save({'model': model.state_dict(), 'preds': predictions, 'targets': targets, 'batch_idx': batch_idx}, 'debug_nan_state.pth')
             continue # Skip backprop and optimizer step

        # --- Backpropagation ---
        optimizer.zero_grad(set_to_none=True) # set_to_none=True peut être plus rapide
        scaler.scale(loss).backward() # Ceci ne devrait plus causer d'erreur device

        # --- Clipping de Gradient ---
        # Important: unscale avant de clipper si AMP est utilisé
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        # --- Mise à jour de l'Optimiseur ---
        # Ne step que si le gradient est valide (ce qui devrait être le cas si on est arrivé ici)
        scaler.step(optimizer)
        scaler.update()

        # --- Enregistrement des métriques (seulement si perte calculée) ---
        batch_loss = loss.item() # Maintenant on est sûr que loss est finie
        total_loss += batch_loss
        # Utiliser .item() seulement pour le logging, pas pour les cumuls
        loss_cls_total += losses['loss_classifier'].item() # Déjà détaché dans compute_loss
        loss_reg_total += losses['loss_box_reg'].item()    # Déjà détaché dans compute_loss
        num_positives = losses.get('num_positives', 0) # Utiliser get avec défaut
        total_positives += num_positives

        # Mise à jour de la barre de progression
        progress_bar.set_postfix({
            'Loss': f"{batch_loss:.4f}",
            'Avg Loss': f"{total_loss / (batch_idx + 1):.4f}",
            'Cls': f"{losses['loss_classifier'].item():.4f}",
            'Reg': f"{losses['loss_box_reg'].item():.4f}",
            'NPos': f"{num_positives:.1f}", # Afficher nombre de positifs
            'LR': f"{optimizer.param_groups[0]['lr']:.1e}"
        })

    # Calcul des moyennes à la fin de l'époque
    num_batches_processed = progress_bar.n # Nombre de batches réellement traités
    if num_batches_processed > 0 :
        avg_train_loss = total_loss / num_batches_processed
        avg_cls_loss = loss_cls_total / num_batches_processed
        avg_reg_loss = loss_reg_total / num_batches_processed
        avg_positives = total_positives / num_batches_processed
    else:
        avg_train_loss = 0.0
        avg_cls_loss = 0.0
        avg_reg_loss = 0.0
        avg_positives = 0.0

    epoch_time = time.time() - start_time

    print(f"Epoch {epoch+1} [Train] Completed in {epoch_time:.2f}s: Avg Loss: {avg_train_loss:.4f} (Cls: {avg_cls_loss:.4f}, Reg: {avg_reg_loss:.4f}), Avg Pos: {avg_positives:.2f}")
    return avg_train_loss


@torch.no_grad() # Désactive le calcul de gradient pour l'évaluation
def evaluate(model, data_loader, device): # img_size n'est plus directement utilisé ici si perte utilise shapes
    model.eval()
    total_loss = 0.0
    loss_cls_total = 0.0
    loss_reg_total = 0.0
    total_positives = 0.0
    start_time = time.time()

    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

    # Récupérer les strides FPN du modèle
    try:
        fpn_strides = list(model.strides.values())
        if not all(isinstance(s, (int, float)) for s in fpn_strides):
             raise TypeError("Model strides must be numeric values.")
    except AttributeError:
        print("\nCRITICAL ERROR: Model does not have 'strides' attribute during evaluation.")
        print("Cannot compute FPN loss. Returning infinite loss.")
        return float('inf') # Retourner une perte infinie pour indiquer l'échec
    except TypeError as e:
        print(f"\nCRITICAL ERROR: Problem with model strides format during evaluation: {e}")
        print("Returning infinite loss.")
        return float('inf') # Retourner une perte infinie pour indiquer l'échec

    for batch_idx, batch_data in enumerate(progress_bar):
        if batch_data is None:
             print(f"Warning: Skipping None validation batch at index {batch_idx}")
             continue
        images, targets = batch_data
        if images is None or targets is None:
             print(f"Warning: Skipping empty validation batch content at index {batch_idx}")
             continue

        images = images.to(device)
        # <<< Important: Mettre les cibles sur le bon device AVANT de les passer à la perte
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # --- Inférence et Calcul de Perte ---
        # Utiliser autocast si AMP est activé globalement et qu'on est sur CUDA
        with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP and device.type=='cuda'):
            predictions = model(images)

            # <<< Passer le 'device' explicitement à la fonction de perte
            losses = compute_fpn_detection_loss(
                all_cls_logits=predictions['cls_logits'],
                all_bbox_pred=predictions['bbox_pred'],
                targets=targets,
                fpn_strides=fpn_strides,
                device=device, # <<< Passé ici
                pos_weight=FPN_POS_WEIGHT_CLS,
                lambda_bbox=FPN_LAMBDA_BBOX
            )

            # Gérer le cas où la perte retourne None ou dict avec perte nulle (erreur interne)
            if losses is None:
                 print(f"Warning: Loss computation failed unexpectedly for validation batch {batch_idx}. Skipping loss for this batch.")
                 continue # Skip to next batch

            loss = losses['loss_total'] # Ici loss n'aura pas de grad_fn (@torch.no_grad)

        # Vérifier la perte de validation
        if not math.isfinite(loss.item()):
             print(f"\nWarning: Validation loss is {loss.item()} at Batch {batch_idx} (NumPos: {losses.get('num_positives', 0)}). Skipping loss calculation for this batch.")
             # Ne pas ajouter la perte si non finie
        else:
             # Accumuler les pertes valides
             total_loss += loss.item()
             loss_cls_total += losses['loss_classifier'].item() # Déjà détaché
             loss_reg_total += losses['loss_box_reg'].item()    # Déjà détaché
             total_positives += losses.get('num_positives', 0)


        # --- Section Post-traitement (NMS, etc.) ---
        # !! CETTE SECTION EST SUPPRIMÉE POUR L'INSTANT !!
        # ... (commentaire inchangé) ...
        # --------------------------------------------

        progress_bar.set_postfix({
            'Avg Loss': f"{total_loss / (batch_idx + 1):.4f}"
         })

    # Calcul des moyennes à la fin de l'évaluation
    num_batches_processed = progress_bar.n # Nombre de batches réellement traités
    if num_batches_processed > 0 :
        avg_val_loss = total_loss / num_batches_processed
        avg_cls_loss = loss_cls_total / num_batches_processed
        avg_reg_loss = loss_reg_total / num_batches_processed
        avg_positives = total_positives / num_batches_processed
    else:
        avg_val_loss = 0.0
        avg_cls_loss = 0.0
        avg_reg_loss = 0.0
        avg_positives = 0.0

    eval_time = time.time() - start_time

    print(f"Evaluation Completed in {eval_time:.2f}s: Avg Loss: {avg_val_loss:.4f} (Cls: {avg_cls_loss:.4f}, Reg: {avg_reg_loss:.4f}), Avg Pos: {avg_positives:.2f}")

    # Pour l'instant, on retourne juste la perte moyenne.
    # Le calcul de mAP nécessiterait l'implémentation du post-traitement.
    return avg_val_loss