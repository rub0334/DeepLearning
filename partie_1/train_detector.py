# train_detector.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import os
import sys # Ajout pour sys.exit en cas d'erreur critique

# Importations des modules locaux
import config       # Notre configuration
import utils        # Nos utilitaires (seed, checkpointing)
import data_loader  # Notre chargement de données
import model_detection # Notre définition de modèle
import engine_detection # Nos boucles train/eval

def main():
    # 1. Initialisation et Configuration
    print("--- Démarrage de l'entraînement du détecteur de texte ---")
    utils.set_seed(config.RANDOM_SEED) # Fixer les graines pour la reproductibilité
    device = config.DEVICE

    # ---> AJOUT DU BLOC PRINT DE CONFIGURATION ICI <---
    print("--- Configuration Utilisée ---")
    print(f"  Device: {device}") # Utilise la variable locale 'device'
    print(f"  Data Base Path: {config.DATA_BASE_PATH}")
    print(f"  Checkpoint Dir: {config.CHECKPOINT_DIR}")
    print(f"  Num Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Weight Decay: {config.WEIGHT_DECAY}")
    print(f"  LR Step Size: {config.LR_STEP_SIZE}")
    print(f"  LR Gamma: {config.LR_GAMMA}")
    print(f"  Eval Confidence Threshold: {config.EVAL_CONFIDENCE_THRESHOLD}")
    print(f"  Eval IoU Threshold: {config.EVAL_IOU_THRESHOLD}")
    print("-" * 30) # Séparateur pour la clarté
    # ---> FIN AJOUT <---

    print(f"Utilisation du device: {device}") # Peut être redondant maintenant, mais informatif

    # 2. Chargement des Données
    print("Chargement des données...")
    try:
        train_loader, val_loader = data_loader.create_dataloaders(config.BATCH_SIZE)
    except ValueError as e:
        print(f"Erreur critique lors de la création des DataLoaders: {e}")
        sys.exit(1) # Arrêter si les datasets sont vides
    except ImportError as e_imp: # Attraper l'erreur d'import potentiel (ex: cv2)
        print(f"Erreur d'importation lors de la création des DataLoaders: {e_imp}")
        sys.exit(1)

    # 3. Initialisation du Modèle
    print("Initialisation du modèle...")
    model = model_detection.get_detection_model(num_classes=config.NUM_CLASSES)
    model.to(device) # Déplacer le modèle sur le bon device

    # 4. Initialisation de l'Optimiseur et du Scheduler
    print("Initialisation de l'optimiseur et du scheduler...")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    lr_scheduler = StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)

    # 5. Initialisation pour la Précision Mixte
    use_amp = torch.cuda.is_available() and config.DEVICE == torch.device("cuda")
    scaler = None
    if use_amp:
        scaler = torch.amp.GradScaler()
        print("Utilisation de la Précision Mixte Automatique (AMP) via torch.amp.")
    else:
        print("Précision Mixte non utilisée.")

    # 6. Chargement d'un Checkpoint (si existant)
    start_epoch = 0
    best_f1_score = 0.0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "last_checkpoint.pth.tar")
    scheduler_to_load = lr_scheduler # Garder une référence au scheduler

    if os.path.exists(checkpoint_path):
        print(f"Reprise depuis le checkpoint: {checkpoint_path}")
        start_epoch, best_f1_score = utils.load_checkpoint(checkpoint_path, model, optimizer, scheduler=scheduler_to_load)
        print(f"Reprise à l'époque {start_epoch}. Meilleur F1 Score précédent: {best_f1_score:.4f}")

    # 7. Boucle d'Entraînement Principale
    print("\n--- Début de la boucle d'entraînement ---")
    start_time = time.time()

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()

        # --- Phase d'Entraînement ---
        try:
            if not hasattr(engine_detection, 'train_one_epoch'):
                 print("ERREUR CRITIQUE: La fonction 'train_one_epoch' est manquante dans engine_detection.py!")
                 sys.exit(1)
            train_loss = engine_detection.train_one_epoch(
                model, optimizer, train_loader, device, epoch, scaler
            )
        except Exception as e_train:
             print(f"\n--- ERREUR PENDANT L'ENTRAINEMENT (Epoch {epoch+1}) ---")
             print(f"{type(e_train).__name__}: {e_train}")
             import traceback
             traceback.print_exc() # Afficher la trace complète pour le débogage
             # Optionnel: Sauvegarder l'état actuel pour débogage
             # utils.save_checkpoint({...}, filename=f"error_state_epoch_{epoch+1}.pth.tar")
             sys.exit(1) # Arrêter en cas d'erreur d'entraînement


        # --- Phase d'Évaluation des Métriques ---
        try:
            if not hasattr(engine_detection, 'evaluate_metrics'):
                 print("ERREUR CRITIQUE: La fonction 'evaluate_metrics' est manquante dans engine_detection.py!")
                 sys.exit(1)
            eval_metrics = engine_detection.evaluate_metrics(
                model, val_loader, device, epoch
            )
            current_f1 = eval_metrics['f1_score']
        except Exception as e_eval:
            print(f"\n--- ERREUR PENDANT L'EVALUATION (Epoch {epoch+1}) ---")
            print(f"{type(e_eval).__name__}: {e_eval}")
            import traceback
            traceback.print_exc()
            # On continue l'entraînement, mais on ne met pas à jour le meilleur modèle
            current_f1 = -1.0 # Mettre une valeur qui n'est jamais la meilleure

        # --- Mise à jour du Scheduler ---
        if lr_scheduler is not None:
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Taux d'apprentissage pour la prochaine époque: {current_lr:.6f}")

        # --- Sauvegarde du Checkpoint ---
        # Sauvegarder seulement si l'évaluation n'a pas échoué (current_f1 != -1.0)
        if current_f1 >= 0: # >= 0 car F1 peut être 0
            is_best = current_f1 > best_f1_score
            if is_best:
                best_f1_score = current_f1
                print(f"** Nouveau meilleur F1-Score de validation: {best_f1_score:.4f} **")
                utils.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_f1_score': best_f1_score,
                    'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None
                }, filename="best_model.pth.tar")

        # Toujours sauvegarder le dernier état (même si l'évaluation a échoué)
        if (epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == config.NUM_EPOCHS:
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_f1_score': best_f1_score, # Sauvegarde le meilleur F1 connu
                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None
            }, filename="last_checkpoint.pth.tar")

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} terminée en {epoch_duration:.2f} secondes.")
        print("-" * 50)

    # 8. Fin de l'entraînement
    total_training_time = time.time() - start_time
    print("--- Entraînement Terminé ---")
    print(f"Durée totale de l'entraînement: {total_training_time / 3600:.2f} heures")
    print(f"Meilleur F1-Score de validation obtenu: {best_f1_score:.4f}")
    print(f"Le meilleur modèle a été sauvegardé dans: {os.path.join(config.CHECKPOINT_DIR, 'best_model.pth.tar')}")
    print(f"Le dernier checkpoint a été sauvegardé dans: {os.path.join(config.CHECKPOINT_DIR, 'last_checkpoint.pth.tar')}")


# --- Protection Multiprocessing pour Windows ---
if __name__ == "__main__":
    # Cette protection est essentielle sous Windows lors de l'utilisation
    # de multiprocessing (DataLoader avec num_workers > 0).
    # Elle garantit que le code de la fonction main() n'est exécuté
    # que par le processus principal et non par les workers.

    # Vérifications initiales des chemins
    paths_to_check = [
        config.TRAIN_ANNOTATION_FILE,
        config.VAL_ANNOTATION_FILE,
        config.TRAIN_IMAGE_DIR,
        config.VAL_IMAGE_DIR
    ]
    paths_ok = True
    for path in paths_to_check:
        if not os.path.exists(path):
            print(f"ERREUR CRITIQUE: Chemin non trouvé : {path}")
            print("Vérifiez le chemin DATA_BASE_PATH dans config.py et la structure de vos dossiers.")
            paths_ok = False

    if not paths_ok:
         print("Arrêt du script à cause de chemins manquants.")
         sys.exit(1)

    # Vérification CUDA
    if config.DEVICE == torch.device("cuda") and not torch.cuda.is_available():
        print("ERREUR CRITIQUE: CUDA est sélectionné dans la config mais n'est pas disponible !")
        print("Vérifiez votre installation PyTorch et les drivers NVIDIA.")
        sys.exit(1)
    elif config.DEVICE == torch.device("cpu"):
        print("Attention: Entraînement sur CPU. Cela sera TRES lent.")

    # Lancer l'entraînement principal
    main()