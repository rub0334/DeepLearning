# train.py

import torch
import torch.nn as nn
import torch.optim as optim
# Importer Subset pour créer des sous-ensembles de datasets
from torch.utils.data import DataLoader, Subset
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import difflib
import math # Pour math.ceil

# --- Importer les modules du projet ---
import config
from dataset import TextOCRDataset
from model import CRNN
from utils import CTCLabelConverter, get_character_set, save_checkpoint, load_checkpoint, collate_fn

# --- Gestion de l'import optionnel de jiwer ---
try:
    import jiwer
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip()
    ])
    print("jiwer library found. Will use it for WER/CER calculation.")
except ImportError:
    print("Warning: jiwer library not found. CER calculation will use difflib (WER not calculated).")
    print("Install jiwer for better metrics: pip install jiwer")
    jiwer = None
    transformation = None

def main():
    # --- Configuration initiale ---
    print(f"Using device: {config.DEVICE}")
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU.")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- Préparation des données ---
    print("Loading annotations to determine character set...")
    try:
        with open(config.ANNOTATION_FILE, 'r', encoding='utf-8') as f:
            annotations_data = json.load(f).get('anns', {})
        if not annotations_data: raise ValueError("'anns' key not found or empty.")
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return

    character_set = get_character_set(annotations_data)
    if not character_set: raise ValueError("Could not determine character set.")
    label_converter = CTCLabelConverter(character_set)
    num_classes_for_model = label_converter.num_classes

    print("Preparing FULL Datasets (Train + Val)...")
    try:
        # Créer les datasets COMPLETS une seule fois
        full_train_dataset = TextOCRDataset(
            config.ANNOTATION_FILE, config.IMAGE_DIR, label_converter,
            is_train=True, validation_split=config.VALIDATION_SPLIT, seed=config.SEED
        )
        # Le dataset de validation reste constant
        val_dataset = TextOCRDataset(
            config.ANNOTATION_FILE, config.IMAGE_DIR, label_converter,
            is_train=False, validation_split=config.VALIDATION_SPLIT, seed=config.SEED
        )
    except Exception as e:
        print(f"Error creating Datasets: {e}")
        return

    if len(full_train_dataset) == 0:
        print("Error: Full training dataset is empty.")
        return
    print(f"Full training dataset size: {len(full_train_dataset)}")
    if len(val_dataset) == 0:
        print("Warning: Validation dataset is empty.")

    # --- Configuration DataLoader (Workers, Pin Memory) ---
    num_workers = min(4, os.cpu_count()) if os.name != 'nt' else 0
    print(f"Using {num_workers} workers for DataLoaders.")
    pin_memory = True if config.DEVICE != torch.device('cpu') else False

    # Créer le DataLoader de validation (il ne change pas)
    val_loader = DataLoader(
        val_dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    ) if len(val_dataset) > 0 else None # Créer seulement si val_dataset n'est pas vide

    # --- Initialisation Modèle, Loss, Optimiseur ---
    print(f"Initializing model with {num_classes_for_model} classes...")
    model = CRNN(num_classes=num_classes_for_model).to(config.DEVICE)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_metric = float('inf')

    # --- Reprise depuis Checkpoint ---
    # ... (code inchangé pour charger le checkpoint) ...
    latest_checkpoint = None
    if os.path.exists(config.CHECKPOINT_DIR):
         checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.startswith('crnn_epoch_') and f.endswith('.pth')]
         if checkpoints:
              try:
                  checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                  latest_checkpoint = os.path.join(config.CHECKPOINT_DIR, checkpoints[-1])
              except Exception as e:
                  print(f"Could not determine latest checkpoint automatically: {e}")

    if latest_checkpoint and os.path.exists(latest_checkpoint):
         print(f"Resuming training from {latest_checkpoint}")
         try:
             start_epoch = load_checkpoint(latest_checkpoint, model, optimizer)
         except Exception as e:
             print(f"Error loading checkpoint {latest_checkpoint}: {e}. Starting from scratch.")
             start_epoch = 0
    else:
        print("No checkpoint found or specified. Starting training from scratch.")


    # --- Boucle d'Entraînement ---
    print("Starting training...")
    current_train_samples = config.INITIAL_TRAIN_SAMPLES # Initialiser

    for epoch in range(start_epoch, config.EPOCHS):
        print("-" * 50)
        print(f"Starting Epoch {epoch+1}/{config.EPOCHS}")

        # === Mise à jour de la taille du jeu d'entraînement (si activé) ===
        if config.ENABLE_PROGRESSIVE_LOADING:
            # Calculer le nombre d'étapes d'augmentation passées
            increase_steps_done = math.floor(epoch / config.INCREASE_EVERY_N_EPOCHS)
            # Calculer la taille cible pour cette époque
            target_samples = config.INITIAL_TRAIN_SAMPLES + increase_steps_done * config.SAMPLE_INCREASE_STEP
            # Plafonner à la taille max configurée ET à la taille réelle du dataset
            max_possible_samples = len(full_train_dataset)
            if config.MAX_TRAIN_SAMPLES_CAP is not None:
                 max_possible_samples = min(max_possible_samples, config.MAX_TRAIN_SAMPLES_CAP)
            current_train_samples = min(target_samples, max_possible_samples)

            print(f"Progressive Loading: Using {current_train_samples} / {len(full_train_dataset)} training samples for this epoch.")
            # Créer le sous-ensemble pour cette époque
            indices = list(range(current_train_samples))
            epoch_train_dataset = Subset(full_train_dataset, indices)
        else:
            # Utiliser le jeu d'entraînement complet si non activé
            if epoch == start_epoch: # Afficher une seule fois
                 print(f"Progressive Loading Disabled. Using all {len(full_train_dataset)} training samples.")
            epoch_train_dataset = full_train_dataset # Utiliser le dataset complet directement

        if len(epoch_train_dataset) == 0:
            print("Error: Current training subset is empty. Stopping.")
            break

        # === Créer le DataLoader d'entraînement pour CETTE époque ===
        train_loader = DataLoader(
            epoch_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )

        # === Phase d'Entraînement ===
        model.train()
        epoch_train_loss = 0.0
        # Utiliser leave=False pour que la barre se nettoie après la fin de la boucle interne
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]", leave=False)

        for batch_idx, batch_data in enumerate(progress_bar):
            # ... (Gestion des batches vides/None - code inchangé) ...
            if batch_data is None: continue
            images, labels_padded, label_lengths = batch_data
            if images.nelement() == 0: continue

            images = images.to(config.DEVICE, non_blocking=True)
            optimizer.zero_grad()
            log_probs = model(images)

            # ... (Préparation input_lengths, label_lengths_squeezed - code inchangé) ...
            input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.long)
            label_lengths_squeezed = label_lengths.squeeze(-1).long()

            # ... (Filtrage des labels de longueur 0 - code inchangé) ...
            valid_indices = label_lengths_squeezed > 0
            if not valid_indices.all():
                log_probs_filtered = log_probs[:, valid_indices, :]
                labels_padded_filtered = labels_padded[valid_indices, :]
                input_lengths_filtered = input_lengths[valid_indices]
                label_lengths_filtered = label_lengths_squeezed[valid_indices]
                if log_probs_filtered.nelement() == 0: continue
                log_probs_to_use, labels_to_use, input_lengths_to_use, label_lengths_to_use = \
                    log_probs_filtered, labels_padded_filtered, input_lengths_filtered, label_lengths_filtered
            else:
                 log_probs_to_use, labels_to_use, input_lengths_to_use, label_lengths_to_use = \
                    log_probs, labels_padded, input_lengths, label_lengths_squeezed

            loss = criterion(log_probs_to_use, labels_to_use, input_lengths_to_use, label_lengths_to_use)

            if torch.isinf(loss) or torch.isnan(loss):
                 print(f"\nWarning: Invalid loss detected (inf or nan) at epoch {epoch+1}, batch {batch_idx}. Skipping backward pass.")
                 continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
            optimizer.step()

            epoch_train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_losses.append(avg_train_loss)
        # Affichage loss train après la barre de progression
        print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")

        # === Phase de Validation ===
        # Utiliser le val_loader qui a été créé une seule fois avec le dataset de validation complet
        if val_loader: # Vérifier si le loader de validation existe
            model.eval()
            epoch_val_loss = 0.0
            all_preds = []
            all_gts = []
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]", leave=False)

            with torch.no_grad():
                for batch_data in progress_bar_val:
                    # ... (Gestion batches vides/None - code inchangé) ...
                    if batch_data is None: continue
                    images, labels_padded, label_lengths = batch_data
                    if images.nelement() == 0: continue

                    images = images.to(config.DEVICE, non_blocking=True)
                    log_probs = model(images) # (T, N, C)

                    # ... (Calcul loss validation avec filtrage - code inchangé) ...
                    input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.long)
                    label_lengths_squeezed = label_lengths.squeeze(-1).long()
                    valid_indices = label_lengths_squeezed > 0
                    if not valid_indices.all():
                        log_probs_filtered = log_probs[:, valid_indices, :]
                        labels_padded_filtered = labels_padded[valid_indices, :]
                        input_lengths_filtered = input_lengths[valid_indices]
                        label_lengths_filtered = label_lengths_squeezed[valid_indices]
                        if log_probs_filtered.nelement() > 0:
                             loss = criterion(log_probs_filtered, labels_padded_filtered, input_lengths_filtered, label_lengths_filtered)
                             epoch_val_loss += loss.item()
                    else:
                         loss = criterion(log_probs, labels_padded, input_lengths, label_lengths_squeezed)
                         epoch_val_loss += loss.item()

                    # ... (Décodage pour WER/CER - code inchangé) ...
                    preds = log_probs.argmax(dim=2).permute(1, 0)
                    for i in range(preds.size(0)):
                        pred_text = label_converter.decode(preds[i])
                        gt_text_indices = labels_padded[i]
                        gt_text_cleaned = ''.join([label_converter.int_to_char.get(idx.item(), '') for idx in gt_text_indices if idx.item() != 0])
                        all_preds.append(pred_text)
                        all_gts.append(gt_text_cleaned)

            avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_losses.append(avg_val_loss)

            # --- Calcul des métriques (WER/CER avec filtrage GT vides) ---
            current_metric = avg_val_loss
            if not all_gts:
                 print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f} | No validation samples for metrics.")
            else:
                filtered_preds, filtered_gts, empty_gt_count = [], [], 0
                for pred, gt in zip(all_preds, all_gts):
                    if gt: filtered_preds.append(pred); filtered_gts.append(gt)
                    else: empty_gt_count += 1
                if empty_gt_count > 0: print(f"Info: Filtered out {empty_gt_count} samples with empty GT for metrics.")

                if not filtered_gts: print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f} | No valid non-empty GT samples after filtering.")
                elif jiwer and transformation:
                    try:
                        wer = jiwer.wer(filtered_gts, filtered_preds, truth_transform=transformation, hypothesis_transform=transformation)
                        cer = jiwer.cer(filtered_gts, filtered_preds, truth_transform=transformation, hypothesis_transform=transformation)
                        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f} | WER: {wer*100:.2f}% | CER: {cer*100:.2f}% (jiwer on {len(filtered_gts)} samples)")
                        current_metric = cer
                    except Exception as e: print(f"Error calculating jiwer metrics: {e}")
                else: # Fallback difflib
                    if filtered_gts:
                        # ... (calcul difflib sur filtered_preds/filtered_gts - code inchangé) ...
                        total_edit_distance = 0; total_gt_length = 0
                        for pred, gt in zip(filtered_preds, filtered_gts):
                            sm = difflib.SequenceMatcher(None, pred, gt); edit_distance = 0
                            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                                if tag == 'replace': edit_distance += max(i2 - i1, j2 - j1)
                                elif tag == 'delete': edit_distance += i2 - i1
                                elif tag == 'insert': edit_distance += j2 - j1
                            total_edit_distance += edit_distance; total_gt_length += len(gt)
                        if total_gt_length > 0:
                            cer_approx = (total_edit_distance / total_gt_length) * 100
                            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f} | Approx CER: {cer_approx:.2f}% (difflib on {len(filtered_gts)} samples)")
                            current_metric = cer_approx
                        else: print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f} | Total GT length 0 for difflib.")


            # --- Sauvegarde du meilleur modèle ---
            if current_metric < best_val_metric:
                best_val_metric = current_metric
                best_model_path = os.path.join(config.CHECKPOINT_DIR, "crnn_best_model.pth")
                save_checkpoint(model, optimizer, epoch + 1, best_model_path)
                print(f"*** New best model saved (metric: {best_val_metric:.4f}) ***")

            # ... (scheduler.step(current_metric) si utilisé) ...

        else: # Si val_loader n'existe pas
            val_losses.append(None)
            print(f"Epoch {epoch+1} completed. No validation performed.")

        # --- Sauvegarde Checkpoint Époque ---
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"crnn_epoch_{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)
        print("-" * 50) # Séparateur pour la prochaine époque

    print("Training finished.")

    # --- Affichage Courbes Loss ---
    # ... (code inchangé) ...
    if train_losses:
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, train_losses, label='Train Loss', marker='o', linestyle='-')
        valid_val_epochs = [i+1 for i, v in enumerate(val_losses) if v is not None]
        valid_val_losses = [v for v in val_losses if v is not None]
        if valid_val_losses: plt.plot(valid_val_epochs, valid_val_losses, label='Validatio Loss', marker='x', linestyle='--')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training and Validation Loss'); plt.legend(); plt.grid(True)
        loss_curve_path = os.path.join(config.OUTPUT_DIR, 'loss_curves.png')
        try: plt.savefig(loss_curve_path); print(f"Loss curves saved to {loss_curve_path}")
        except Exception as e: print(f"Could not save loss curves plot: {e}")


if __name__ == "__main__":
    main()
    