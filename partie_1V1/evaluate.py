import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import json
from tqdm import tqdm
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms

import config
from dataset import TextOCRDataset
from utils import collate_fn
from model import CRNN
from utils import CTCLabelConverter, get_character_set, load_checkpoint
try:
    import jiwer
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip()
    ])
except ImportError:
    print("Warning: jiwer library not found. WER/CER calculation will be approximate or skipped.")
    jiwer = None

def visualize_predictions(model, dataloader, label_converter, device, num_samples=10, output_dir=config.OUTPUT_DIR):
    model.eval()
    samples_shown = 0
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualization samples to {output_dir}")

    with torch.no_grad():
        for batch_idx, (images, labels_padded, label_lengths) in enumerate(dataloader):
            if samples_shown >= num_samples:
                break

            images = images.to(device)

            log_probs = model(images) # (T, N, C)
            preds = log_probs.argmax(dim=2).permute(1, 0) # (N, T)

            for i in range(images.size(0)):
                if samples_shown >= num_samples:
                    break

                # Récupérer l'image originale (ou au moins la version avant normalisation)
                # Ceci nécessite d'adapter le Dataset pour retourner aussi l'image originale recadrée
                # Ou de recharger l'image basée sur l'index (moins efficace)
                # Pour la simplicité ici, on dé-normalise l'image tensorielle
                img_tensor = images[i].cpu().detach()
                # Denormalize: (tensor * std) + mean -> puis convertir en PIL
                img_tensor = img_tensor * 0.5 + 0.5 # Inverse de Normalize(mean=[0.5], std=[0.5])
                img_tensor = torch.clamp(img_tensor, 0, 1)
                img_pil = transforms.ToPILImage()(img_tensor.squeeze(0)) # Squeeze channel dim for grayscale

                # Convertir en RGB pour dessiner en couleur si besoin
                if img_pil.mode == 'L':
                    img_pil = img_pil.convert('RGB')

                # Obtenir prédiction et vérité terrain
                pred_text = label_converter.decode(preds[i])
                gt_text_indices = labels_padded[i]
                gt_text = ''.join([label_converter.int_to_char.get(idx.item(), '?') for idx in gt_text_indices if idx.item() != 0]) # Nettoyer GT

                # Dessiner sur l'image
                draw = ImageDraw.Draw(img_pil)
                # Utiliser une police par défaut ou spécifier un chemin .ttf
                try:
                     font = ImageFont.truetype("arial.ttf", 15) # Peut nécessiter l'installation de la police
                except IOError:
                     font = ImageFont.load_default()

                text_to_display = f"GT:  {gt_text}\nPred:{pred_text}"
                # Position du texte (en haut à gauche)
                text_position = (5, 5)
                 # Boîte pour le fond du texte
                bbox = draw.textbbox(text_position, text_to_display, font=font)
                # Ajouter une marge
                margin = 5
                bbox = (bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin)
                draw.rectangle(bbox, fill="white")
                draw.text(text_position, text_to_display, fill="black", font=font)

                # Sauvegarder l'image annotée
                img_pil.save(os.path.join(output_dir, f"sample_{samples_shown}_pred.png"))
                samples_shown += 1

def evaluate(model, dataloader, criterion, label_converter, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_gts = []
    progress_bar = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for images, labels_padded, label_lengths in progress_bar:
            if images is None or labels_padded is None or label_lengths is None or images.nelement() == 0: continue

            images = images.to(device)

            log_probs = model(images) # (T, N, C)
            input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.long)

            # Calcul de la loss (filtrer les labels de longueur 0)
            valid_indices = label_lengths.squeeze(-1) > 0
            if not valid_indices.all():
                 log_probs_filtered = log_probs[:, valid_indices, :]
                 labels_padded_filtered = labels_padded[valid_indices, :]
                 input_lengths_filtered = input_lengths[valid_indices]
                 label_lengths_squeezed_filtered = label_lengths.squeeze(-1)[valid_indices]
                 if log_probs_filtered.nelement() > 0:
                     loss = criterion(log_probs_filtered, labels_padded_filtered, input_lengths_filtered, label_lengths_squeezed_filtered)
                     total_loss += loss.item() * log_probs_filtered.size(1) # Pondérer par le nombre d'échantillons valides
            else:
                label_lengths_squeezed = label_lengths.squeeze(-1)
                loss = criterion(log_probs, labels_padded, input_lengths, label_lengths_squeezed)
                total_loss += loss.item() * log_probs.size(1) # Pondérer par la taille du batch

            # Décodage pour WER/CER
            preds = log_probs.argmax(dim=2).permute(1, 0) # (N, T)
            for i in range(preds.size(0)):
                pred_text = label_converter.decode(preds[i])
                gt_text_indices = labels_padded[i]
                gt_text_cleaned = ''.join([label_converter.int_to_char.get(idx.item(), '?') for idx in gt_text_indices if idx.item() != 0])
                all_preds.append(pred_text)
                all_gts.append(gt_text_cleaned)

    avg_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    print(f"Average Evaluation Loss: {avg_loss:.4f}")

    # Calculer WER/CER
    if jiwer:
        try:
            wer = jiwer.wer(all_gts, all_preds, truth_transform=transformation, hypothesis_transform=transformation)
            cer = jiwer.cer(all_gts, all_preds, truth_transform=transformation, hypothesis_transform=transformation)
            print(f"Word Error Rate (WER): {wer*100:.2f}%")
            print(f"Character Error Rate (CER): {cer*100:.2f}%")
            return avg_loss, wer, cer
        except Exception as e:
            print(f"Error calculating WER/CER with jiwer: {e}. Metrics might be inaccurate.")
            return avg_loss, -1.0, -1.0 # Retourner des valeurs invalides
    else:
        # Calcul approximatif si jiwer n'est pas disponible (moins fiable)
        print("jiwer not available, cannot calculate accurate WER/CER.")
        return avg_loss, -1.0, -1.0


def main_eval(checkpoint_path):
    print(f"Evaluating model from checkpoint: {checkpoint_path}")

    # --- Recharger la configuration nécessaire (vocabulaire) ---
    # Il est crucial que le vocabulaire soit le même qu'à l'entraînement
    print("Loading annotations to determine character set...")
    with open(config.ANNOTATION_FILE, 'r', encoding='utf-8') as f:
        annotations_data = json.load(f)['anns']
    character_set = get_character_set(annotations_data)
    label_converter = CTCLabelConverter(character_set)
    num_classes = label_converter.num_classes

    # --- Préparer le DataLoader (utiliser le set de validation ici) ---
    print("Preparing Validation DataLoader...")
    # Utiliser les mêmes transformations que pendant l'entraînement/validation
    # Note: Pas d'augmentation pour l'évaluation finale.
    eval_dataset = TextOCRDataset(config.ANNOTATION_FILE, config.IMAGE_DIR, label_converter, is_train=False) # is_train=False pour utiliser le split de validation
    if len(eval_dataset) == 0:
        print("Error: Validation dataset is empty. Cannot evaluate.")
        return

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False, # Pas besoin de mélanger pour l'évaluation
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # --- Charger le Modèle ---
    print(f"Loading model with {num_classes} classes...")
    model = CRNN(num_classes=num_classes).to(config.DEVICE)
    try:
        load_checkpoint(checkpoint_path, model) # Charger uniquement les poids du modèle
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # --- Définir la Loss (pour le calcul de la loss d'évaluation) ---
    criterion = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True).to(config.DEVICE) # Utiliser 'sum' puis diviser par N total

    # --- Lancer l'Évaluation ---
    evaluate(model, eval_loader, criterion, label_converter, config.DEVICE)

    # --- Générer des Visualisations ---
    print("\nGenerating some prediction visualizations...")
    from torchvision import transforms # Assurez-vous que transforms est importé
    visualize_predictions(model, eval_loader, label_converter, config.DEVICE, num_samples=20)


if __name__ == "__main__":
    # --- Trouver le dernier checkpoint entraîné ---
    latest_epoch = -1
    checkpoint_to_eval = None
    if os.path.exists(config.CHECKPOINT_DIR):
         checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.startswith('crnn_epoch_') and f.endswith('.pth')]
         if checkpoints:
              try:
                   checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                   checkpoint_to_eval = os.path.join(config.CHECKPOINT_DIR, checkpoints[-1]) # Prendre le dernier
                   latest_epoch = int(checkpoints[-1].split('_')[-1].split('.')[0])
              except Exception as e:
                   print(f"Could not determine latest checkpoint automatically: {e}")

    if checkpoint_to_eval:
        main_eval(checkpoint_to_eval)
    else:
        print("No checkpoint found in", config.CHECKPOINT_DIR)
        print("Please train the model first using train.py or specify a checkpoint path.")
        # Exemple si vous voulez évaluer un checkpoint spécifique:
        # specific_checkpoint = "checkpoints/crnn_epoch_10.pth" # Mettre le chemin désiré
        # if os.path.exists(specific_checkpoint):
        #    main_eval(specific_checkpoint)