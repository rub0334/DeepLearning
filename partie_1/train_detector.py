import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import time
import argparse # Utilisé pour les options de ligne de commande
import numpy as np
from PIL import Image

# Importer depuis nos modules locaux
import config
from utils import set_seed, save_checkpoint, load_checkpoint, visualize_detection
from data_loader import create_detection_dataloaders
from model_detection import build_detection_model
from engine_detection import train_one_epoch, evaluate

def main():
    # --- Configuration et Initialisation ---
    parser = argparse.ArgumentParser(description="Train Text Detection Model 'From Scratch' on TextOCR")
    # Arguments pour remplacer ou compléter config.py
    parser.add_argument('--config_lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--config_epochs', type=int, default=config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--config_batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--resume', type=str, default=None, help='Path to specific checkpoint to resume training from')
    parser.add_argument('--resume-best', action='store_true', # <<< NOUVEL ARGUMENT
                        help='Resume training from the best saved model (model_best_detection.pth.tar) if it exists')
    parser.add_argument('--vis_freq', type=int, default=5, help='Frequency (in epochs) to save visualization samples')
    # Suppression de l'ancien argument '--resume_best' qui était redondant avec '--resume-best' action='store_true'
    args = parser.parse_args()

    # Appliquer les arguments de la ligne de commande
    learning_rate = args.config_lr
    num_epochs = args.config_epochs
    batch_size = args.config_batch_size

    # Fixer les graines pour la reproductibilité
    set_seed(config.SEED)

    # Sélection du device (GPU si disponible)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    if config.DEVICE == "cuda":
        try:
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        except Exception as e:
             print(f"Could not get CUDA device details: {e}")


    # Créer les DataLoaders
    print("Creating DataLoaders...")
    train_loader, val_loader = create_detection_dataloaders(
        train_json=config.TRAIN_JSON,
        val_json=config.VAL_JSON,
        train_img_dir=config.TRAIN_IMG_DIR,
        val_img_dir=config.VAL_IMG_DIR,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        img_size=config.IMG_SIZE # Assurez-vous que IMG_SIZE est défini dans config.py
    )
    if not train_loader or not val_loader:
        print("Error creating dataloaders. Exiting.")
        return # Stop execution if dataloaders fail

    # Construire le modèle de détection
    print("Building detection model...")
    # Utiliser NUM_DETECTION_CLASSES de config.py (ex: 1 pour 'texte', le background est implicite dans la perte BCE)
    model = build_detection_model(num_classes=config.NUM_DETECTION_CLASSES) # Assurez-vous que NUM_DETECTION_CLASSES=1
    model.to(device)

    # Optimiseur
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)

    # Scheduler de taux d'apprentissage
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Exemple

    # Précision mixte (AMP)
    use_amp_actual = config.USE_AMP and device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp_actual)
    if config.USE_AMP and not use_amp_actual:
        print("Warning: AMP is enabled in config but no CUDA device found or CUDA unavailable. Running without AMP.")
    elif not config.USE_AMP:
        print("AMP not enabled in config.")
    print(f"Automatic Mixed Precision (AMP) active: {scaler.is_enabled()}")

    # --- Logique de reprise depuis un checkpoint ---
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_to_load = None

    # 1. Priorité à --resume s'il est spécifié
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Attempting to resume from specified checkpoint: {args.resume}")
            checkpoint_to_load = args.resume
        else:
            print(f"Warning: Specified resume checkpoint not found at '{args.resume}'.")

    # 2. Sinon (pas de --resume ou fichier non trouvé), essayer --resume-best si demandé
    if checkpoint_to_load is None and args.resume_best:
        best_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'model_best_detection.pth.tar')
        if os.path.isfile(best_checkpoint_path):
            print(f"Attempting to resume from best checkpoint: {best_checkpoint_path}")
            checkpoint_to_load = best_checkpoint_path
        else:
            print("Warning: --resume-best specified, but best checkpoint ('model_best_detection.pth.tar') not found.")

    # 3. Charger le checkpoint sélectionné (s'il y en a un)
    if checkpoint_to_load:
        try:
            model, optimizer, scaler, start_epoch, best_val_loss = load_checkpoint(
                checkpoint_to_load, model, optimizer, scaler, device
            )
            print(f"Successfully resumed training from epoch {start_epoch + 1}. Best loss recorded: {best_val_loss:.4f}")

            # !!! IMPORTANT: Reprise de l'état du scheduler !!!
            # L'utilitaire load_checkpoint actuel ne charge PAS l'état du scheduler.
            # Pour une reprise parfaite, il faudrait sauvegarder et charger scheduler.state_dict().
            # Workaround pour StepLR/Cosine etc.: Appeler step() le bon nombre de fois.
            print(f"Attempting to restore scheduler state for epoch {start_epoch}...")
            # S'assurer que le scheduler existe avant de le step
            if scheduler:
                for _ in range(start_epoch):
                    scheduler.step()
                print(f"Scheduler stepped {start_epoch} times. Current LR: {optimizer.param_groups[0]['lr']:.1e}")
            else:
                print("No scheduler defined, skipping scheduler state restoration.")

        except Exception as e:
            print(f"Error loading checkpoint '{checkpoint_to_load}': {e}")
            print("Starting training from scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
            # Réinitialiser scaler au cas où le chargement a échoué partiellement
            scaler = torch.amp.GradScaler(enabled=use_amp_actual)
    else:
        print("No valid checkpoint specified or found. Starting training from scratch.")
        start_epoch = 0
        best_val_loss = float('inf')
    # --- Fin de la logique de reprise ---


    # Créer les répertoires si nécessaire
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    vis_dir = os.path.join(config.BASE_PROJECT_DIR, "visualizations_detection") # Nom spécifique
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {config.CHECKPOINT_DIR}")
    print(f"Visualizations will be saved in: {vis_dir}")


    # --- Boucle d'Entraînement ---
    print(f"\nStarting training from epoch {start_epoch + 1}...")
    start_training_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Entraînement
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, scaler
            # grad_clip_norm est lu depuis config via engine_detection
        )

        # Validation
        val_loss = evaluate(
            model, val_loader, device
        )

        # Mise à jour du scheduler
        current_lr = optimizer.param_groups[0]['lr'] # Get LR avant step
        if scheduler:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                 scheduler.step(val_loss)
            else:
                 scheduler.step() # Pour StepLR, CosineAnnealingLR, etc.
        new_lr = optimizer.param_groups[0]['lr'] # Get LR après step

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} finished in {epoch_duration:.2f}s. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {new_lr:.1e}")

        if new_lr != current_lr:
            print(f"Learning rate updated to {new_lr:.1e}")

        # Sauvegarde du checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"🎉 New best model found with validation loss: {best_val_loss:.4f}")

        # Préparer l'état à sauvegarder (inclure scaler si AMP est utilisé)
        checkpoint_state = {
            'epoch': epoch + 1, # Sauvegarder l'époque *suivante* à démarrer
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'scaler': scaler.state_dict() if scaler and scaler.is_enabled() else None, # Sauvegarder scaler si activé
            # Optionnel: Sauvegarder l'état du scheduler pour une reprise parfaite
            # 'scheduler': scheduler.state_dict() if scheduler else None,
            'config': { # Sauvegarder les hyperparams utilisés peut être utile
                 'lr': learning_rate,
                 'batch_size': batch_size,
                 'img_size': config.IMG_SIZE,
                 'num_epochs': num_epochs,
                 'seed': config.SEED,
                 'num_det_classes': config.NUM_DETECTION_CLASSES,
                 'weight_decay': config.WEIGHT_DECAY
            }
        }

        # Sauvegarder le checkpoint
        save_last = True # Toujours sauvegarder le dernier checkpoint
        save_best = is_best # Sauvegarder si c'est le meilleur

        if save_last:
            save_checkpoint(
                checkpoint_state,
                is_best=False, # Ne pas écraser le meilleur avec celui-ci
                checkpoint_dir=config.CHECKPOINT_DIR,
                filename=f'checkpoint_last.pth.tar'
            )

        if save_best:
             save_checkpoint(
                 checkpoint_state,
                 is_best=True, # Va sauvegarder comme model_best_detection.pth.tar
                 checkpoint_dir=config.CHECKPOINT_DIR
                 # Le nom de fichier 'model_best...' est géré dans save_checkpoint si is_best=True
             )

        # Visualisation (décommentée mais nécessite une fonction visualize_sample_predictions adaptée)
        # Attention: la fonction visualize_sample_predictions fournie précédemment
        # avait un post-traitement simplifié qui n'est PAS compatible avec la sortie FPN
        # (cls_logits: List[N,HxW,C], bbox_pred: List[N,HxW,4]).
        # Il faudrait la réécrire complètement pour gérer la sortie FPN (décodage, NMS multi-niveau).
        # Pour l'instant, on la garde commentée.
        # if (epoch + 1) % args.vis_freq == 0 or is_best:
        #    print(f"Generating visualization samples for epoch {epoch+1}...")
        #    # visualize_sample_predictions_fpn(model, val_loader, device, vis_dir, epoch + 1) # <-- FONCTION A CREER/ADAPTER


    # --- Fin de l'Entraînement ---
    total_training_time = time.time() - start_training_time
    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_training_time / 3600:.2f} hours")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")
    print(f"Best model saved at: {os.path.join(config.CHECKPOINT_DIR, 'model_best_detection.pth.tar')}")


# La fonction visualize_sample_predictions doit être adaptée pour FPN
# La version précédente n'est pas compatible.
# def visualize_sample_predictions_fpn(model, data_loader, device, output_dir, epoch_num, num_samples=3):
#     """
#     Sauvegarde des images de validation avec les boîtes GT et les prédictions FPN.
#     NECESSITE UNE IMPLEMENTATION COMPLETE DU POST-TRAITEMENT FPN (DECODAGE + NMS)
#     """
#     print("Visualization function for FPN output needs implementation (decoding + NMS). Skipping visualization.")
#     # ... Implementation future ...
#     pass


if __name__ == "__main__":
    main()