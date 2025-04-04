import torch
import torch.nn.functional as F  # Pour F.pad
import string
import config  # Importer la configuration
import os  # Pour load/save checkpoint


class CTCLabelConverter:
    """ Convertit entre le texte et les indices pour la CTC Loss. """

    def __init__(self, character_set):
        """
        Initialise le convertisseur.
        Args:
            character_set (set ou str): L'ensemble des caractères uniques présents dans le dataset.
        """
        # Trier les caractères pour une correspondance cohérente
        list_character = sorted(list(character_set))

        # Créer les dictionnaires de mapping
        # Index 0 est réservé pour le 'blank' token de CTC
        self.char_to_int = {char: i + 1 for i, char in enumerate(list_character)}
        self.int_to_char = {i + 1: char for i, char in enumerate(list_character)}

        # Ajouter le blank token (utiliser celui défini dans config)
        self.blank_token_idx = 0
        self.char_to_int[config.BLANK_TOKEN] = self.blank_token_idx
        self.int_to_char[self.blank_token_idx] = config.BLANK_TOKEN

        # Nombre total de classes = nombre de caractères + 1 (pour le blank)
        self.num_classes = len(self.char_to_int)

        print(f"Character set size (including blank): {self.num_classes}")
        # Afficher seulement une partie du set s'il est trop long
        char_preview = "".join(list_character)
        if len(char_preview) > 200:
            char_preview = char_preview[:100] + "..." + char_preview[-100:]
        print(f"Character set preview: {char_preview}")

    def encode(self, text):
        """
        Encode une chaîne de caractères en une séquence d'indices.
        Ignore les caractères non présents dans le character_set.
        Args:
            text (str): La chaîne de caractères à encoder.
        Returns:
            torch.IntTensor: Le tenseur d'indices correspondant.
        """
        # Mapper chaque caractère à son indice, ignorer les inconnus
        encoded = [self.char_to_int.get(char, -1) for char in text]
        # Filtrer les caractères non trouvés (ceux mappés à -1)
        encoded = [idx for idx in encoded if idx != -1]
        return torch.IntTensor(encoded)

    def decode(self, indices, raw=False):
        """
        Décode une séquence d'indices en une chaîne de caractères.
        Utilise un décodage 'best path' simplifié par défaut (supprime répétitions et blanks).
        Args:
            indices (torch.Tensor ou list): Séquence d'indices à décoder.
            raw (bool): Si True, ne supprime ni les répétitions ni les blanks (utile pour voir la sortie brute du modèle).
        Returns:
            str: La chaîne de caractères décodée.
        """
        # Assurer que les indices sont sur CPU et sont des entiers pour le traitement
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().tolist()  # Convertir Tensor en liste d'entiers Python

        if not raw:
            # 1. Supprimer les répétitions consécutives
            merged_indices = []
            if len(indices) > 0:
                merged_indices.append(indices[0])
                for i in range(1, len(indices)):
                    if indices[i] != indices[i - 1]:
                        merged_indices.append(indices[i])

            # 2. Supprimer les blank tokens (index 0)
            decoded_indices = [idx for idx in merged_indices if idx != self.blank_token_idx]
        else:
            # Garder la séquence brute si raw=True
            decoded_indices = indices

        # 3. Convertir les indices restants en caractères
        # Utiliser .get(idx, '?') pour gérer les indices invalides potentiels
        text = ''.join([self.int_to_char.get(idx, '?') for idx in decoded_indices])
        return text


# Fonction pour créer le vocabulaire à partir des annotations
def get_character_set(annotations_data):
    """
    Extrait l'ensemble des caractères uniques des annotations fournies.
    Args:
        annotations_data (dict): Le dictionnaire 'anns' du fichier JSON.
    Returns:
        set: L'ensemble des caractères uniques trouvés.
    """
    all_chars = set()
    print("Extracting character set from annotations...")
    # Utiliser tqdm si le dictionnaire est très grand (optionnel)
    # from tqdm import tqdm
    # for ann_id, ann_details in tqdm(annotations_data.items()):
    for ann_id, ann_details in annotations_data.items():
        text = ann_details.get('utf8_string', '')
        # Mettre à jour l'ensemble avec les caractères de la chaîne actuelle
        all_chars.update(list(text))
    # Optionnel: Ajouter des caractères de base s'ils manquent ?
    # all_chars.update(list(string.printable)) # Non recommandé si on veut se limiter au dataset
    if not all_chars:
        print("Warning: Character set is empty after processing annotations!")
    return all_chars


# Fonction de padding pour le DataLoader (gère les séquences de longueur variable)
def collate_fn(batch):
    """
    Fonction de collation personnalisée pour DataLoader.
    Prend une liste de tuples (image, label, label_length) et les assemble en batches.
    Gère le padding des labels ET des images de largeur variable.
    """
    # Filtrer les échantillons None qui pourraient provenir d'erreurs dans __getitem__
    valid_batch = [item for item in batch if item is not None and item[0] is not None and item[0].nelement() > 0]

    if not valid_batch:
        # Si le batch est vide après filtrage, retourner des tenseurs vides pour éviter un crash immédiat
        print("Warning: collate_fn received an empty or invalid batch.")
        images_stacked = torch.empty((0, 1, config.IMG_HEIGHT, 1), dtype=torch.float)
        labels_padded_out = torch.empty((0, 1), dtype=torch.long)
        label_lengths_out = torch.empty((0,), dtype=torch.long)
        return images_stacked, labels_padded_out, label_lengths_out

    # Sépare les images, labels et longueurs de labels du batch valide
    images, labels, label_lengths_list = zip(*valid_batch)

    # --- Padding des Images ---
    # 1. Trouver la largeur maximale dans le batch valide
    max_width = 0
    for img in images:
        if img.shape[2] > max_width:  # img shape is (C, H, W)
            max_width = img.shape[2]

    # 2. Padder chaque image pour atteindre max_width
    padded_images = []
    num_channels = images[0].shape[0]  # Obtenir le nombre de canaux (ex: 1 pour grayscale)
    img_height = images[0].shape[1]  # Obtenir la hauteur (devrait être config.IMG_HEIGHT)

    for img in images:
        c, h, w = img.shape
        if w < max_width:
            padding_width = max_width - w
            # Utiliser F.pad. Tuple de padding: (pad_gauche, pad_droite, pad_haut, pad_bas)
            # On padde seulement à droite (dimension 2)
            # Valeur de padding 0: correspond à la moyenne 0.5 si normalisé entre [0, 1]
            # ou 0 si normalisé entre [-1, 1] (ce qui est notre cas) -> fond gris/moyen
            try:
                img_padded = F.pad(img, (0, padding_width, 0, 0), mode='constant', value=0)
                padded_images.append(img_padded)
            except Exception as e:
                print(f"Error during image padding for image with shape {img.shape}: {e}. Skipping image.")
                # On pourrait ajouter une image dummy ici, mais il est plus simple de skipper
                # et de gérer la potentielle disparité de taille plus tard.
                # Pour l'instant, on ne l'ajoute pas à padded_images.
                pass  # Ne pas ajouter l'image qui a causé l'erreur
        else:
            padded_images.append(img)  # Pas besoin de padding

    # Si aucune image n'a pu être paddée correctement
    if not padded_images:
        print("Warning: No images left in batch after padding attempt in collate_fn.")
        images_stacked = torch.empty((0, num_channels, img_height, 1), dtype=torch.float)
        labels_padded_out = torch.empty((0, 1), dtype=torch.long)
        label_lengths_out = torch.empty((0,), dtype=torch.long)
        return images_stacked, labels_padded_out, label_lengths_out

    # 3. Empiler les images paddées
    try:
        images_stacked = torch.stack(padded_images, 0)
    except RuntimeError as e:
        print(f"Critical Error during torch.stack in collate_fn after padding: {e}")
        print(f"Number of images attempted to stack: {len(padded_images)}")
        # Tenter d'afficher les formes pour aider au débogage
        # for i, p_img in enumerate(padded_images):
        #     print(f" Shape of image {i} to stack: {p_img.shape}")
        raise e  # Renvoyer l'erreur car c'est un problème critique

    # --- Padding des Labels ---
    # Récupérer les labels et longueurs correspondant aux images qui ont été stackées
    # (Cela suppose une correspondance 1:1 entre images et labels dans valid_batch)
    # Si le padding d'image a échoué et skippé des images, cette partie doit être adaptée.
    # Pour l'instant, on suppose que le padding réussit pour toutes les images dans valid_batch.

    # Concatène les tenseurs de longueur
    label_lengths = torch.cat(label_lengths_list)

    # Padde les séquences de labels
    labels_int = [torch.IntTensor(lbl) if not isinstance(lbl, torch.Tensor) else lbl.int() for lbl in labels]
    try:
        labels_padded_out = torch.nn.utils.rnn.pad_sequence(labels_int, batch_first=True,
                                                            padding_value=0)  # 0 est le blank index
    except Exception as e:
        print(f"Error during label padding: {e}")
        # Gérer l'erreur, par exemple en retournant des tenseurs vides ou en levant l'erreur
        raise e

    label_lengths_out = label_lengths

    # --- Vérification Finale des Dimensions ---
    # Assurer que les tailles de batch correspondent après tout le traitement
    final_batch_size = images_stacked.size(0)
    if final_batch_size != labels_padded_out.size(0) or final_batch_size != label_lengths_out.size(0):
        print(f"Critical Error in collate_fn: Batch size mismatch after processing!")
        print(
            f"Images shape: {images_stacked.shape}, Labels shape: {labels_padded_out.shape}, Lengths shape: {label_lengths_out.shape}")
        # Tenter d'ajuster ou lever une erreur ? Lever une erreur est plus sûr.
        raise RuntimeError(
            f"Batch size mismatch in collate_fn outputs ({final_batch_size}, {labels_padded_out.size(0)}, {label_lengths_out.size(0)})")

    return images_stacked, labels_padded_out, label_lengths_out


# Pour sauvegarder/charger l'état
def save_checkpoint(model, optimizer, epoch, filepath):
    """Sauvegarde l'état du modèle et de l'optimiseur."""
    print(f"Saving checkpoint for epoch {epoch} to {filepath}...")
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Ajouter d'autres infos si besoin (ex: vocabulaire, config, historique loss)
        # 'config': config.__dict__, # Sauvegarder la config peut être utile
        # 'char_to_int': label_converter.char_to_int # Sauvegarder le mapping
    }
    # Créer le répertoire parent s'il n'existe pas
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        torch.save(state, filepath)
        print("Checkpoint saved successfully.")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def load_checkpoint(filepath, model, optimizer=None):
    """Charge l'état du modèle et optionnellement de l'optimiseur."""
    if not os.path.exists(filepath):
        print(f"Error: Checkpoint file not found at {filepath}")
        return 0  # Retourner époque 0 si le fichier n'existe pas

    # Charger sur CPU si CUDA n'est pas dispo, sinon sur le device actuel
    if not torch.cuda.is_available():
        map_location = torch.device('cpu')
        print("Loading checkpoint onto CPU.")
    else:
        # Charger sur le device actuel (normalement le GPU configuré)
        map_location = None  # None utilise le device par défaut (config.DEVICE)
        print(f"Loading checkpoint onto device: {config.DEVICE}")

    try:
        checkpoint = torch.load(filepath, map_location=map_location)

        # Charger les poids du modèle
        model.load_state_dict(checkpoint['model_state_dict'])

        # Charger l'état de l'optimiseur si fourni et présent
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}. Optimizer state will be reset.")

        # Récupérer l'époque (ajouter 1 pour démarrer l'époque *suivante*)
        # L'époque sauvegardée est celle qui vient de se terminer.
        epoch = checkpoint.get('epoch', 0)
        print(f"Checkpoint loaded successfully from epoch {epoch} ({filepath})")

        # Retourner l'époque à laquelle commencer le nouvel entraînement
        return epoch  # L'entraînement reprendra à partir de cette époque (ex: si epoch=10 est chargé, on commence l'époque 10)

    except Exception as e:
        print(f"Error loading checkpoint from {filepath}: {e}")
        # Retourner 0 pour indiquer qu'il faut recommencer depuis le début
        return 0