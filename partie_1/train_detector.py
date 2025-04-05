# train_detector.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import os
import sys # Ajout pour sys.exit en cas d'erreur critique

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
    print(f"Utilisation du device: {device}")

    # 2. Chargement des Données
    print("Chargement des données...")
    try:
        train_loader, val_loader = data_loader.create_dataloaders(config.BATCH_SIZE)
    except ValueError as e:
        print(f"Erreur critique lors de la création des DataLoaders: {e}")
        sys.exit(1) # Arrêter si les datasets sont vides

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
        # Passer le scheduler à load_checkpoint
        start_epoch, best_f1_score = utils.load_checkpoint(checkpoint_path, model, optimizer, scheduler=scheduler_to_load)
        print(f"Reprise à l'époque {start_epoch}. Meilleur F1 Score précédent: {best_f1_score:.4f}")
        # Important: S'assurer que le lr_scheduler a été avancé si non chargé depuis checkpoint
        # La fonction load_checkpoint modifiée devrait gérer ça via scheduler.load_state_dict()

    # 7. Boucle d'Entraînement Principale
    print("\n--- Début de la boucle d'entraînement ---")
    start_time = time.time() # S'assurer qu'elle est définie ici

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()

        # --- Phase d'Entraînement ---
        try:
            # Vérifier si la fonction existe avant de l'appeler
            if not hasattr(engine_detection, 'train_one_epoch'):
                 print("ERREUR CRITIQUE: La fonction 'train_one_epoch' est manquante dans engine_detection.py!")
                 sys.exit(1)

            train_loss = engine_detection.train_one_epoch(
                model, optimizer, train_loader, device, epoch, scaler
            )
        except Exception as e_train:
             print(f"\n--- ERREUR PENDANT L'ENTRAINEMENT (Epoch {epoch+1}) ---")
             print(f"{e_train}")
             # Optionnel: Sauvegarder l'état actuel pour débogage
             # utils.save_checkpoint({...}, filename="error_state.pth.tar")
             raise e_train # Relancer l'erreur pour arrêter proprement


        # --- Phase d'Évaluation des Métriques ---
        try:
             # Vérifier si la fonction existe avant de l'appeler
            if not hasattr(engine_detection, 'evaluate_metrics'):
                 print("ERREUR CRITIQUE: La fonction 'evaluate_metrics' est manquante dans engine_detection.py!")
                 sys.exit(1)

            eval_metrics = engine_detection.evaluate_metrics(
                model, val_loader, device, epoch
            )
            current_f1 = eval_metrics['f1_score']
        except Exception as e_eval:
            print(f"\n--- ERREUR PENDANT L'EVALUATION (Epoch {epoch+1}) ---")
            print(f"{e_eval}")
            # Décider si on continue ou arrête
            # On pourrait juste logguer l'erreur et continuer l'entraînement suivant
            current_f1 = 0.0 # Mettre une valeur par défaut pour éviter erreur plus loin
            # raise e_eval # Décommenter pour arrêter en cas d'erreur d'évaluation

        # --- Mise à jour du Scheduler ---
        if lr_scheduler is not None:
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Taux d'apprentissage pour la prochaine époque: {current_lr:.6f}")

        # --- Sauvegarde du Checkpoint ---
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

        if (epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == config.NUM_EPOCHS:
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_f1_score': best_f1_score,
                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None
            }, filename="last_checkpoint.pth.tar")

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} terminée en {epoch_duration:.2f} secondes.")
        print("-" * 50)

    # 8. Fin de l'entraînement
    # Cette partie ne sera atteinte que si la boucle se termine normalement
    total_training_time = time.time() - start_time # `start_time` est définie avant la boucle
    print("--- Entraînement Terminé ---")
    print(f"Durée totale de l'entraînement: {total_training_time / 3600:.2f} heures")
    print(f"Meilleur F1-Score de validation obtenu: {best_f1_score:.4f}")
    print(f"Le meilleur modèle a été sauvegardé dans: {os.path.join(config.CHECKPOINT_DIR, 'best_model.pth.tar')}")
    print(f"Le dernier checkpoint a été sauvegardé dans: {os.path.join(config.CHECKPOINT_DIR, 'last_checkpoint.pth.tar')}")

if __name__ == "__main__":
    # ... (vérifications initiales inchangées) ...
    main()