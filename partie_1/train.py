# train.py
"""
Main training script for the end-to-end OCR model.
Handles dataset loading, model initialization, training loop,
validation, logging, and checkpointing.
"""

import os
import time
import argparse
import json
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import numpy as np

from typing import Dict, List, Tuple, Optional # <<<=== AJOUTEZ CETTE LIGNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # Use PyTorch's built-in TensorBoard
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Import project modules
import config
import utils
from dataset import TextOCRDataset, collate_fn
from model import OCREndToEndModel

# --- Helper Functions ---

def train_one_epoch(model, loader, criterion_det, criterion_rec, optimizer, scaler, device, epoch, writer):
    """Runs one training epoch."""
    model.train()
    running_loss = 0.0
    running_det_loss = 0.0
    running_rec_loss = 0.0
    num_samples = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")

    for i, batch in enumerate(progress_bar):
        images = batch['image'].to(device, non_blocking=True)
        det_targets = batch['det_target'].to(device, non_blocking=True)
        rec_targets_flat = batch['rec_targets_flat'].to(device, non_blocking=True)
        rec_target_lengths = batch['rec_target_lengths'].to(device, non_blocking=True)

        batch_size = images.size(0)
        num_samples += batch_size
        optimizer.zero_grad(set_to_none=True) # More efficient potentially

        with autocast(enabled=config.USE_AMP):
            # Forward pass
            outputs = model(images)
            det_logits = outputs['detection']      # (B, H', W')
            rec_logits = outputs['recognition']    # (SeqLen, B, VocabSize)

            # --- Calculate Losses ---
            # Detection Loss (e.g., BCE + Dice)
            loss_det_bce = criterion_det['bce'](det_logits, det_targets)
            loss_det_dice = criterion_det['dice'](det_logits, det_targets)
            loss_det = loss_det_bce + loss_det_dice # Combine detection losses

            # Recognition Loss (CTC)
            # Prepare for CTCLoss: Need log_probs, flat targets, input lengths, target lengths
            rec_log_probs = F.log_softmax(rec_logits, dim=2) # (SeqLen, B, VocabSize)
            # Input lengths: Length of the sequence produced by the RNN (width of feature map)
            seq_len = rec_log_probs.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long).to(device)

            # Check if there are any recognition targets in the batch
            if rec_targets_flat.numel() > 0 and rec_target_lengths.numel() > 0:
                 loss_rec = criterion_rec(rec_log_probs, rec_targets_flat, input_lengths, rec_target_lengths)
                 # Handle potential inf loss from CTC (can happen with short sequences/targets)
                 if torch.isinf(loss_rec):
                      print("Warning: Infinite CTC loss detected. Setting to 0 for this batch.")
                      loss_rec = torch.tensor(0.0, device=device, requires_grad=True) # Keep grad flowing
                 elif torch.isnan(loss_rec):
                      print("Warning: NaN CTC loss detected. Setting to 0 for this batch.")
                      loss_rec = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                 # No text instances in this batch, set rec loss to 0
                 loss_rec = torch.tensor(0.0, device=device, requires_grad=True) # Still need grad if other losses exist


            # Total Weighted Loss
            total_loss = (config.DETECTION_LOSS_WEIGHT * loss_det +
                          config.RECOGNITION_LOSS_WEIGHT * loss_rec)

        # Backward pass & Optimization
        scaler.scale(total_loss).backward()

        # Optional Gradient Clipping (unscale first)
        if config.GRAD_CLIP_NORM is not None and config.GRAD_CLIP_NORM > 0:
            scaler.unscale_(optimizer) # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

        # --- Logging ---
        current_loss = total_loss.item()
        running_loss += current_loss * batch_size
        running_det_loss += loss_det.item() * batch_size
        running_rec_loss += loss_rec.item() * batch_size if isinstance(loss_rec, torch.Tensor) else loss_rec * batch_size

        # Update progress bar description
        progress_bar.set_postfix(loss=f"{current_loss:.4f}",
                                 det_loss=f"{loss_det.item():.4f}",
                                 rec_loss=f"{loss_rec.item() if isinstance(loss_rec, torch.Tensor) else loss_rec:.4f}")

        # Log batch losses to TensorBoard (optional, can be noisy)
        # global_step = epoch * len(loader) + i
        # writer.add_scalar('Loss/train_batch', current_loss, global_step)
        # writer.add_scalar('Loss/train_det_batch', loss_det.item(), global_step)
        # writer.add_scalar('Loss/train_rec_batch', loss_rec.item() if isinstance(loss_rec, torch.Tensor) else loss_rec, global_step)


    # Calculate average epoch losses
    epoch_loss = running_loss / num_samples if num_samples > 0 else 0
    epoch_det_loss = running_det_loss / num_samples if num_samples > 0 else 0
    epoch_rec_loss = running_rec_loss / num_samples if num_samples > 0 else 0

    return epoch_loss, epoch_det_loss, epoch_rec_loss


def validate(model, loader, criterion_det, criterion_rec, device, vocab_data, epoch, writer):
    """Runs validation."""
    model.eval()
    total_val_loss = 0.0
    total_det_loss = 0.0
    total_rec_loss = 0.0
    num_samples = 0

    # For Metrics
    all_pred_polygons = [] # List[List[List[Tuple]]] - Batch x Polys x Points x 2
    all_gt_polygons = []   # List[List[List[Tuple]]]
    all_pred_texts = []    # List[str] - Flat list of all predicted texts in validation set
    all_gt_texts = []      # List[str] - Flat list of all GT texts in validation set

    idx_to_char = vocab_data['idx_to_char']
    blank_idx = vocab_data['char_to_idx'].get(utils.CTC_BLANK_TOKEN, 0)

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Validate]")

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            images = batch['image'].to(device, non_blocking=True)
            det_targets = batch['det_target'].to(device, non_blocking=True)
            rec_targets_flat = batch['rec_targets_flat'].to(device, non_blocking=True)
            rec_target_lengths = batch['rec_target_lengths'].to(device, non_blocking=True)
            gt_polys_batch = batch['gt_polygons'] # Keep on CPU for post-processing/metrics
            image_ids = batch['image_ids']
            original_sizes = batch['original_sizes']

            batch_size = images.size(0)
            num_samples += batch_size

            with autocast(enabled=config.USE_AMP): # Can use AMP even in eval if desired
                outputs = model(images)
                det_logits = outputs['detection']
                rec_logits = outputs['recognition'] # (SeqLen, B, VocabSize)

                # --- Calculate Losses --- (Same as training, just for monitoring)
                loss_det_bce = criterion_det['bce'](det_logits, det_targets)
                loss_det_dice = criterion_det['dice'](det_logits, det_targets)
                loss_det = loss_det_bce + loss_det_dice

                rec_log_probs = F.log_softmax(rec_logits, dim=2)
                seq_len = rec_log_probs.size(0)
                input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long).to(device)

                if rec_targets_flat.numel() > 0 and rec_target_lengths.numel() > 0:
                    loss_rec = criterion_rec(rec_log_probs, rec_targets_flat, input_lengths, rec_target_lengths)
                    if torch.isinf(loss_rec) or torch.isnan(loss_rec):
                         loss_rec = torch.tensor(0.0, device=device)
                else:
                    loss_rec = torch.tensor(0.0, device=device)

                total_loss = (config.DETECTION_LOSS_WEIGHT * loss_det +
                              config.RECOGNITION_LOSS_WEIGHT * loss_rec)

            total_val_loss += total_loss.item() * batch_size
            total_det_loss += loss_det.item() * batch_size
            total_rec_loss += loss_rec.item() * batch_size

            # --- Calculate Metrics ---
            # 1. Post-process Detection Output
            det_probs = torch.sigmoid(det_logits).cpu().numpy() # (B, H', W')
            batch_pred_polygons = []
            # TODO: Implement polygon extraction from probability map
            # This is non-trivial. Common approaches:
            #   a) Thresholding + findContours (like in basic example)
            #   b) More advanced methods like Pixel Aggregation (PAN++) or DBNet postprocessing.
            # Placeholder: Assume we get polygons somehow (e.g., just using GT for now for testing pipeline)
            # Example using findContours (very basic):
            for b_idx in range(batch_size):
                 prob_map = (det_probs[b_idx] * 255).astype(np.uint8)
                 thresh_map = cv2.threshold(prob_map, 128, 255, cv2.THRESH_BINARY)[1] # Simple threshold
                 contours, _ = cv2.findContours(thresh_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                 img_pred_polygons = []
                 for cnt in contours:
                      # Approximate polygon, scale back to original image size coords (or target size)
                      epsilon = 0.01 * cv2.arcLength(cnt, True)
                      approx = cv2.approxPolyDP(cnt, epsilon, True)
                      if len(approx) >= 3 and cv2.contourArea(approx) > 10: # Filter small/invalid
                           # Scale points from feature map size back to input image size
                           scaled_poly = [(pt[0][0] * config.FEATURE_MAP_STRIDE, pt[0][1] * config.FEATURE_MAP_STRIDE) for pt in approx]
                           img_pred_polygons.append(scaled_poly)
                 batch_pred_polygons.append(img_pred_polygons)

            all_pred_polygons.extend(batch_pred_polygons)
            all_gt_polygons.extend(gt_polys_batch) # These are already scaled to target image size from dataset

            # 2. Decode Recognition Output (Greedy for simplicity)
            decoded_batch_preds = utils.ctc_greedy_decode(rec_log_probs.cpu(), idx_to_char, blank_idx)
            all_pred_texts.extend(decoded_batch_preds)

            # 3. Decode Ground Truth Recognition Targets
            current_pos = 0
            for b_idx in range(batch_size):
                # Find how many text instances belong to this image
                num_texts_in_image = batch['rec_target_lengths'].shape[0] if 'rec_target_lengths' in batch else 0 # Needs adjustment based on actual collate output structure
                # This part is tricky with the flat structure. We need to know how many texts belonged to image 'b_idx'.
                # Let's assume for now the rec_target_lengths corresponds correctly to the flatten order.
                # We need a way to reconstruct GT texts per image if doing image-level matching.
                # For now, compare flat pred vs flat GT decode.

                # Decode GT targets associated with this batch
                if rec_target_lengths.numel() > 0:
                     start_idx = 0 # This assumes flat structure needs more careful indexing
                     # Correct way: Need length info per image passed from collate or dataset
                     # Let's decode all GTs flat for now for overall CER/WER
                     pass # Deferring detailed GT decoding until structure is clearer

            # --- Visualization (Optional) ---
            if i < config.NUM_VISUALIZATION_BATCHES and epoch % config.VALIDATE_EPOCH_FREQ == 0: # Visualize first few batches
                 img_vis = batch['image'][0].cpu().numpy().transpose(1, 2, 0) # CHW -> HWC
                 # Unnormalize for visualization
                 mean = np.array(config.MEAN)
                 std = np.array(config.STD)
                 img_vis = std * img_vis + mean
                 img_vis = np.clip(img_vis, 0, 1) * 255
                 img_vis = img_vis.astype(np.uint8)
                 img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV

                 vis_gt_polys = all_gt_polygons[-batch_size:][0] # Get GT polys for the first image of this batch
                 vis_pred_polys = all_pred_polygons[-batch_size:][0] # Get Pred polys for first image

                 # TODO: Get GT/Pred text corresponding *only* to the first image
                 vis_gt_texts = ["GT_placeholder"] * len(vis_gt_polys) # Placeholder
                 vis_pred_texts = ["Pred_placeholder"] * len(vis_pred_polys)# Placeholder

                 vis_path = os.path.join(config.CURRENT_VISUALIZATION_DIR, f"epoch_{epoch+1}_batch_{i}_img_{image_ids[0]}.png")

                 # Need original image path? Or draw on transformed image? Drawing on transformed is easier here.
                 # utils.visualize_predictions(original_image_path, vis_pred_polys, vis_pred_texts, vis_gt_polys, vis_gt_texts, vis_path)
                 # Draw directly on img_vis (transformed, unnormalized image)
                 vis_img_out = utils.draw_polygons_and_text(img_vis.copy(), vis_gt_polys, vis_gt_texts, color=(0, 0, 255)) # Red for GT
                 vis_img_out = utils.draw_polygons_and_text(vis_img_out, vis_pred_polys, vis_pred_texts, color=(0, 255, 0)) # Green for Pred
                 cv2.imwrite(vis_path, vis_img_out)


    # --- Decode all Ground Truth Texts ---
    # This needs careful handling of the flattened structure. We need the lengths corresponding
    # to each original text instance. Let's assume rec_target_lengths holds this flat list.
    current_pos_gt = 0
    flat_gt_cpu = rec_targets_flat.cpu()
    lengths_gt_cpu = rec_target_lengths.cpu()
    if lengths_gt_cpu.numel() > 0:
        for length in lengths_gt_cpu:
            target_indices = flat_gt_cpu[current_pos_gt : current_pos_gt + length.item()]
            decoded_gt = "".join([idx_to_char.get(idx.item(), '?') for idx in target_indices])
            all_gt_texts.append(decoded_gt)
            current_pos_gt += length.item()


    # Calculate overall metrics
    avg_val_loss = total_val_loss / num_samples if num_samples > 0 else 0
    avg_det_loss = total_det_loss / num_samples if num_samples > 0 else 0
    avg_rec_loss = total_rec_loss / num_samples if num_samples > 0 else 0

    det_metrics = utils.calculate_detection_metrics(all_pred_polygons, all_gt_polygons, config.DETECTION_IOU_THRESHOLD)
    # Note: CER/WER calculated on *all* decoded sequences vs *all* GT sequences.
    # This measures overall text quality but doesn't perfectly isolate recognition accuracy
    # if detection is poor (e.g., predicts text where there is none).
    rec_metrics = utils.calculate_cer_wer(all_pred_texts, all_gt_texts)

    metrics = {
        'val_loss': avg_val_loss,
        'val_det_loss': avg_det_loss,
        'val_rec_loss': avg_rec_loss,
        'val_det_precision': det_metrics['precision'],
        'val_det_recall': det_metrics['recall'],
        'val_det_f1': det_metrics['f1_score'],
        'val_rec_cer': rec_metrics['cer'],
        'val_rec_wer': rec_metrics['wer']
    }

    return metrics


# --- Main Training Function ---

def main():
    parser = argparse.ArgumentParser(description='Train End-to-End OCR Model')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., latest_checkpoint.pth)')
    args = parser.parse_args()

    config.print_config() # Print loaded configuration

    # --- Setup ---
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    writer = SummaryWriter(log_dir=config.CURRENT_LOG_DIR) # Tensorboard logger

    # --- Data ---
    print("Loading vocabulary...")
    if not os.path.exists(config.VOCAB_PATH):
        print("Vocabulary file not found. Attempting to create...")
        try:
            utils.create_vocab(config.TRAIN_JSON, config.VOCAB_PATH)
        except Exception as e:
            print(f"Error creating vocabulary: {e}. Exiting.")
            return
    vocab_data = utils.load_vocab(config.VOCAB_PATH)
    vocab_size = len(vocab_data['char_to_idx'])
    pad_idx = vocab_data['char_to_idx'].get(utils.PAD_TOKEN, 0) # Check if PAD exists
    blank_idx = vocab_data['char_to_idx'].get(utils.CTC_BLANK_TOKEN, 0)
    print(f"Vocabulary loaded with {vocab_size} tokens. Blank index: {blank_idx}")

    print("Loading datasets...")
    try:
        # Create datasets (using default transforms from dataset.py based on mode)
        train_dataset = TextOCRDataset(config.TRAIN_JSON, config.TRAIN_IMG_DIR, vocab_data, mode='train')
        val_dataset = TextOCRDataset(config.VAL_JSON, config.VAL_IMG_DIR, vocab_data, mode='val')
    except Exception as e:
        print(f"Error creating datasets: {e}. Exiting.")
        return

    # Create dataloaders
    collate_with_padding = lambda batch: collate_fn(batch, pad_idx=pad_idx)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, collate_fn=collate_with_padding,
                              pin_memory=config.PIN_MEMORY, persistent_workers=bool(config.NUM_WORKERS > 0))
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, collate_fn=collate_with_padding,
                            pin_memory=config.PIN_MEMORY, persistent_workers=bool(config.NUM_WORKERS > 0))
    print(f"Train loader: {len(train_loader)} batches. Val loader: {len(val_loader)} batches.")


    # --- Model ---
    print("Initializing model...")
    model = OCREndToEndModel(vocab_size=vocab_size,
                             input_channels=1 if config.CONVERT_GRAYSCALE else 3).to(device)

    # --- Loss Functions ---
    print("Initializing loss functions...")
    criterion_det = {
        'bce': nn.BCEWithLogitsLoss().to(device),
        'dice': utils.DiceLoss().to(device) # Assumes DiceLoss is in utils
    }
    # zero_infinity=True helps handle potential issues in CTC
    criterion_rec = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True).to(device)

    # --- Optimizer & Scheduler ---
    print(f"Initializing optimizer: {config.OPTIMIZER}")
    if config.OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'Adam':
         optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else: # Default or add more options (SGD etc.)
         print(f"Warning: Optimizer {config.OPTIMIZER} not explicitly handled. Using AdamW.")
         optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    scheduler = None
    if config.LR_SCHEDULER == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)
        print("Using StepLR scheduler.")
    elif config.LR_SCHEDULER == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_MAX, eta_min=1e-6) # Add eta_min
        print("Using CosineAnnealingLR scheduler.")
    else:
        print("No LR scheduler selected.")

    # --- AMP Scaler ---
    scaler = GradScaler(enabled=config.USE_AMP)
    print(f"Automatic Mixed Precision (AMP) enabled: {config.USE_AMP}")

    # --- Checkpointing ---
    start_epoch = 0
    best_metric_val = float('inf') if config.BEST_MODEL_METRIC_MODE == 'min' else float('-inf')

    if args.resume:
        checkpoint_path = os.path.join(config.CURRENT_CHECKPOINT_DIR, args.resume)
        if os.path.isfile(checkpoint_path):
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler and 'scheduler_state_dict' in checkpoint:
                     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
                best_metric_val = checkpoint.get('best_metric_val', best_metric_val) # Load best score if saved
                print(f"Checkpoint loaded. Resuming from epoch {start_epoch}, Best {config.BEST_MODEL_METRIC}: {best_metric_val:.4f}")
            except Exception as e:
                 print(f"Error loading checkpoint: {e}. Starting training from scratch.")
                 start_epoch = 0
                 best_metric_val = float('inf') if config.BEST_MODEL_METRIC_MODE == 'min' else float('-inf')
        else:
            print(f"Warning: Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    start_time = time.time()

    for epoch in range(start_epoch, config.EPOCHS):
        epoch_start_time = time.time()

        # Train
        train_loss, train_det_loss, train_rec_loss = train_one_epoch(
            model, train_loader, criterion_det, criterion_rec, optimizer, scaler, device, epoch, writer
        )

        # Log training metrics
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/train_det_epoch', train_det_loss, epoch)
        writer.add_scalar('Loss/train_rec_epoch', train_rec_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch+1}/{config.EPOCHS} [Train] Avg Loss: {train_loss:.4f} (Det: {train_det_loss:.4f}, Rec: {train_rec_loss:.4f})")


        # Validate
        validation_metrics = {}
        if (epoch + 1) % config.VALIDATE_EPOCH_FREQ == 0 or epoch == config.EPOCHS - 1:
             validation_metrics = validate(
                 model, val_loader, criterion_det, criterion_rec, device, vocab_data, epoch, writer
             )

             # Log validation metrics
             for key, value in validation_metrics.items():
                 writer.add_scalar(f"Metrics/{key}", value, epoch)

             print(f"Epoch {epoch+1}/{config.EPOCHS} [Val] Loss: {validation_metrics['val_loss']:.4f} | "
                   f"Det F1: {validation_metrics['val_det_f1']:.4f} | "
                   f"Rec CER: {validation_metrics['val_rec_cer']:.4f} | "
                   f"Rec WER: {validation_metrics['val_rec_wer']:.4f}")
        else:
             print(f"Epoch {epoch+1}/{config.EPOCHS} - Skipping validation.")

        # Update Learning Rate Scheduler (if used)
        if scheduler:
             # Special case: some schedulers like ReduceLROnPlateau need metrics
             if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 if validation_metrics: # Only step if validation was run
                      scheduler.step(validation_metrics[config.BEST_MODEL_METRIC])
             else:
                  scheduler.step() # Most schedulers step per epoch


        # --- Checkpointing ---
        current_metric_val = validation_metrics.get(config.BEST_MODEL_METRIC, None)
        is_best = False
        if current_metric_val is not None:
             if config.BEST_MODEL_METRIC_MODE == 'min' and current_metric_val < best_metric_val:
                 best_metric_val = current_metric_val
                 is_best = True
             elif config.BEST_MODEL_METRIC_MODE == 'max' and current_metric_val > best_metric_val:
                 best_metric_val = current_metric_val
                 is_best = True

        # Prepare checkpoint data
        checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_metric_val': best_metric_val,
                'config': config.__dict__ # Save config for reference (optional)
        }
        if scheduler:
             checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

        # Save latest checkpoint
        if (epoch + 1) % config.SAVE_CHECKPOINT_EPOCH_FREQ == 0:
            latest_checkpoint_path = os.path.join(config.CURRENT_CHECKPOINT_DIR, 'latest_checkpoint.pth')
            torch.save(checkpoint_data, latest_checkpoint_path)
            print(f"Saved latest checkpoint to {latest_checkpoint_path}")

        # Save best checkpoint
        if is_best and config.SAVE_BEST_CHECKPOINT:
             best_checkpoint_path = os.path.join(config.CURRENT_CHECKPOINT_DIR, 'best_checkpoint.pth')
             torch.save(checkpoint_data, best_checkpoint_path)
             print(f"*** Saved BEST checkpoint (Metric {config.BEST_MODEL_METRIC}: {best_metric_val:.4f}) to {best_checkpoint_path} ***")


        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")
        print("-" * 50)


    # --- End of Training ---
    total_training_time = time.time() - start_time
    print(f"\n--- Training Finished ---")
    print(f"Total Training Time: {total_training_time / 3600:.2f} hours")
    print(f"Best Validation {config.BEST_MODEL_METRIC}: {best_metric_val:.4f}")
    print(f"Checkpoints and logs saved in: {config.CURRENT_CHECKPOINT_DIR} and {config.CURRENT_LOG_DIR}")
    writer.close()


if __name__ == '__main__':
    main()