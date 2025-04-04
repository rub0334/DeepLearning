import json
import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset  # DataLoader n'est pas nécessaire ici, importé dans train/eval
from torchvision import transforms
from tqdm import tqdm  # Importé pour la barre de progression
import numpy as np
import random

# Importer la configuration et les utilitaires nécessaires
import config
from utils import CTCLabelConverter


# --- CLASSE POUR LA TRANSFORMATION ResizePad ---
class ResizePadTransform:
    """
    Classe callable pour redimensionner une image à une hauteur fixe
    en conservant le ratio, utilisable avec torchvision.transforms.Compose
    et compatible avec le pickling pour multiprocessing.
    """

    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        """Applique la transformation."""
        # S'assurer que l'input est une image PIL
        if not isinstance(img, Image.Image):
            print(
                f"Warning: Input to ResizePadTransform is not a PIL Image (type: {type(img)}). Attempting to convert.")
            # Essayer de convertir si c'est un format commun, sinon erreur
            try:
                # Exemple simple: si c'est un tenseur, le convertir
                if isinstance(img, torch.Tensor):
                    img = transforms.ToPILImage()(img)
                else:
                    # Si on ne sait pas convertir, on retourne une image vide
                    raise TypeError("Cannot convert input to PIL Image")
            except Exception as e:
                print(f"Error converting input to PIL Image in ResizePadTransform: {e}. Returning dummy image.")
                return Image.new('L', (10, self.height))  # Retourne image grayscale

        img_h = img.height
        img_w = img.width

        # Gérer le cas où l'image source a une hauteur nulle ou négative
        if img_h <= 0:
            print(
                f"Warning: Input image for ResizePadTransform has non-positive height ({img_h}). Creating dummy image.")
            # Utiliser le mode de l'image d'entrée si possible, sinon 'L' (grayscale)
            mode = img.mode if hasattr(img, 'mode') else 'L'
            return Image.new(mode, (max(1, img_w), self.height))  # Garde la largeur si possible

        # Calculer la nouvelle largeur en conservant le ratio
        new_w = int(img_w * (self.height / img_h))

        # S'assurer que la nouvelle largeur est au moins 1 pixel
        new_w = max(1, new_w)

        # Redimensionner l'image
        try:
            # Utiliser les filtres de rééchantillonnage modernes si disponibles
            if hasattr(Image.Resampling, 'LANCZOS'):
                resample_filter = Image.Resampling.LANCZOS
            elif hasattr(Image, 'LANCZOS'):  # Pour certaines versions de Pillow
                resample_filter = Image.LANCZOS
            elif hasattr(Image, 'ANTIALIAS'):  # Fallback pour versions plus anciennes
                resample_filter = Image.ANTIALIAS
            else:  # Si aucune n'est trouvée (très vieux Pillow ?), utiliser BILINEAR ou NEAREST
                resample_filter = Image.BILINEAR

            img_resized = img.resize((new_w, self.height), resample_filter)

        except Exception as e:
            print(f"Error during image resize (to {new_w}x{self.height}): {e}. Returning dummy image.")
            mode = img.mode if hasattr(img, 'mode') else 'L'
            # Retourner une image dummy de la taille cible est plus sûr pour le batching
            return Image.new(mode, (new_w, self.height))

        # --- Optionnel: Padding pour largeur multiple ---
        # Décommenter et adapter si nécessaire pour votre architecture CNN
        # target_multiple = 8 # Ex: multiple de 8
        # if new_w % target_multiple != 0:
        #      pad_w = target_multiple - (new_w % target_multiple)
        #      # padding = (left, top, right, bottom) - Ajouter à droite
        #      padding = (0, 0, pad_w, 0)
        #      # Déterminer la couleur de fond (gris moyen pour normalisation [-1, 1])
        #      # Pour mode 'L' (grayscale), 128 est ~0 après ToTensor et Normalize(0.5, 0.5)
        #      # Pour mode 'RGB', (128, 128, 128)
        #      fill_color = 128 if img_resized.mode == 'L' else (128, 128, 128)
        #      try:
        #          img_resized = ImageOps.expand(img_resized, border=padding, fill=fill_color)
        #      except Exception as e:
        #          print(f"Error during optional image padding: {e}. Using unpadded image.")

        return img_resized

    # Méthodes pour aider au pickling
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


# --- FIN DE LA CLASSE ResizePadTransform ---


# --- CLASSE DU DATASET TextOCR ---
class TextOCRDataset(Dataset):
    """
    Dataset PyTorch pour TextOCR. Chaque échantillon est une instance de texte recadrée.
    Permet de limiter le nombre d'échantillons pour train/validation via config.py.
    """

    def __init__(self, annotations_path, image_dir, label_converter, transform=None, is_train=True,
                 validation_split=config.VALIDATION_SPLIT, seed=config.SEED):
        super().__init__()
        self.image_dir = image_dir
        self.label_converter = label_converter
        self.is_train = is_train
        # Utiliser _get_default_transform si aucun transform n'est passé
        # Passer self.is_train pour potentiellement appliquer des augmentations différentes
        self.transform = transform if transform is not None else self._get_default_transform(self.is_train)

        print(f"Loading annotations from {annotations_path}...")
        try:
            with open(annotations_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Annotation file not found at {annotations_path}")
            raise  # Renvoyer l'erreur pour arrêter le programme proprement
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {annotations_path}")
            raise

        # Vérifier que les clés nécessaires sont présentes
        if 'anns' not in data or 'imgs' not in data or 'imgToAnns' not in data:
            raise ValueError("Annotation JSON is missing required keys ('anns', 'imgs', 'imgToAnns')")

        self.annotations = data['anns']
        self.img_metadata = data['imgs']
        self.img_to_anns = data['imgToAnns']

        # --- Construction de la liste des échantillons (instances de texte) ---
        self.samples = []
        valid_count = 0
        invalid_count = 0
        missing_files = 0

        print("Processing annotations to build samples...")
        # Utiliser img_metadata comme base pour itérer sur les images du set 'train'
        for img_id, img_info in tqdm(self.img_metadata.items(), desc="Checking images"):
            # Filtrer pour garder seulement les images du set 'train'
            if img_info.get('set') != 'train':
                continue

            img_path_suffix = img_info.get('file_name')
            if not img_path_suffix: continue

            # Construire le chemin complet de l'image
            base_img_name = os.path.basename(img_path_suffix)  # Prend juste le nom de fichier
            full_img_path = os.path.join(self.image_dir, base_img_name)

            if not os.path.exists(full_img_path):
                missing_files += 1
                continue  # Ignorer si le fichier image n'est pas trouvé

            # Récupérer les annotations pour cette image
            ann_ids = self.img_to_anns.get(img_id, [])
            for ann_id in ann_ids:
                ann = self.annotations.get(ann_id)
                if not ann: continue

                text = ann.get('utf8_string', '').strip()
                bbox = ann.get('bbox')  # Format: [x_min, y_min, width, height]

                # Ajouter des filtres de validité plus stricts si nécessaire
                # Ex: ignorer si le texte contient des caractères non gérés, etc.
                min_box_width = 5  # Largeur minimale en pixels
                min_box_height = 5  # Hauteur minimale en pixels

                if text and bbox and len(bbox) == 4 and bbox[2] >= min_box_width and bbox[3] >= min_box_height:
                    # Convertir bbox en coordonnées de crop (left, top, right, bottom)
                    x_min, y_min, width, height = bbox
                    crop_coords = (x_min, y_min, x_min + width, y_min + height)

                    self.samples.append({
                        'image_path': full_img_path,
                        'crop_coords': crop_coords,
                        'text': text,
                        'img_id': img_id,  # Garder pour débogage
                        'ann_id': ann_id  # Garder pour débogage
                    })
                    valid_count += 1
                else:
                    invalid_count += 1

        print(f"Found {valid_count} valid text instances from 'train' set images.")
        if invalid_count > 0: print(f"Ignored {invalid_count} invalid, empty, or small annotations.")
        if missing_files > 0: print(
            f"Warning: {missing_files} image files referenced in JSON ('train' set) were not found.")

        if not self.samples:
            raise RuntimeError("No valid samples could be loaded from the dataset! Check paths and annotations.")

        # --- Séparation Train/Validation ---
        random.seed(seed)
        random.shuffle(self.samples)

        n_total_samples = len(self.samples)
        split_idx = int(n_total_samples * (1 - validation_split))

        if self.is_train:
            self.samples = self.samples[:split_idx]
            dataset_type = "training"
            # --- AJOUT: Limiter le nombre d'échantillons d'entraînement ---
            limit = getattr(config, 'MAX_TRAIN_SAMPLES', -1)  # Récupère la limite depuis config, défaut -1
            if limit > 0 and len(self.samples) > limit:
                print(f"Limiting training samples from {len(self.samples)} to {limit}.")
                self.samples = self.samples[:limit]
            # ----------------------------------------------------------
        else:
            self.samples = self.samples[split_idx:]
            dataset_type = "validation"
            # --- AJOUT: Limiter le nombre d'échantillons de validation ---
            limit = getattr(config, 'MAX_VAL_SAMPLES', -1)  # Récupère la limite depuis config, défaut -1
            if limit > 0 and len(self.samples) > limit:
                print(f"Limiting validation samples from {len(self.samples)} to {limit}.")
                self.samples = self.samples[:limit]
            # ----------------------------------------------------------

        print(f"Using {len(self.samples)} samples for {dataset_type}.")
        if len(self.samples) == 0:
            print(f"Warning: {dataset_type.capitalize()} dataset is empty after split and/or limiting!")

    def _get_default_transform(self, is_train):
        """Définit les transformations par défaut, potentiellement différentes pour train/val."""

        transforms_list = []

        # 1. Redimensionnement et Padding (commun à train et val)
        transforms_list.append(ResizePadTransform(height=config.IMG_HEIGHT))

        # 2. Conversion en niveaux de gris (commun)
        transforms_list.append(transforms.Grayscale(num_output_channels=1))

        # 3. Augmentations de données (appliquées seulement pour l'entraînement)
        if is_train and getattr(config, 'APPLY_AUGMENTATION',
                                False):  # Vérifier si l'augmentation est activée dans config
            print("Applying data augmentation for training...")
            transforms_list.extend([
                # Exemples d'augmentations (ajuster les paramètres et probabilités)
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1)
                    # Saturation/Hue ont peu d'effet sur grayscale mais peuvent être là
                ], p=0.5),
                transforms.RandomAffine(degrees=4, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=3, fill=128),
                # fill=gris moyen
                # Ajouter d'autres augmentations ici: ex: RandomPerspective, GaussianBlur, etc.
                # transforms.RandomPerspective(distortion_scale=0.1, p=0.2, fill=128),
                # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))
            ])

        # 4. Conversion en Tenseur (commun)
        transforms_list.append(transforms.ToTensor())  # Convertit en [0, 1]

        # 5. Normalisation (commun)
        # Normalise les pixels pour être autour de 0 (ici entre -1 et 1)
        transforms_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

        return transforms.Compose(transforms_list)

    def __len__(self):
        """Retourne le nombre total d'échantillons dans ce dataset (train ou val)."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Charge, recadre, transforme et retourne une instance de texte et son label."""
        if idx >= len(self.samples):
            raise IndexError("Index out of bounds")

        sample = self.samples[idx]
        image_path = sample['image_path']
        crop_coords = sample['crop_coords']
        text = sample['text']

        try:
            # --- Chargement et Recadrage ---
            # Charger l'image entière (ouvrir une seule fois par échantillon)
            # Convertir en RGB ici peut aider à gérer certains formats d'image (ex: P mode)
            img = Image.open(image_path).convert('RGB')

            # Obtenir dimensions pour le recadrage
            img_w, img_h = img.size
            x1, y1, x2, y2 = map(int, crop_coords)

            # Valider et clipper les coordonnées de recadrage aux limites de l'image
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            # Assurer que x2 > x1 et y2 > y1, et restent dans les limites
            x2 = max(x1 + 1, min(x2, img_w))
            y2 = max(y1 + 1, min(y2, img_h))

            # Vérifier si le crop résultant a une dimension valide (W>0, H>0)
            if x1 >= x2 or y1 >= y2:
                print(
                    f"Warning: Invalid calculated crop dimensions (W={x2 - x1}, H={y2 - y1}) for AnnID {sample['ann_id']}. Returning dummy data.")
                # Retourner des données "dummy" cohérentes en type et forme attendue (si possible)
                dummy_img = torch.zeros((1, config.IMG_HEIGHT, 10),
                                        dtype=torch.float)  # C=1, H=config, W=arbitraire petit
                dummy_label = torch.IntTensor([self.label_converter.blank_token_idx])  # Label blank
                dummy_len = torch.IntTensor([1])  # Longueur 1 (le blank)
                return dummy_img, dummy_label, dummy_len

            # Effectuer le recadrage
            text_crop = img.crop((x1, y1, x2, y2))

            # --- Transformations ---
            # Appliquer la séquence de transformations (inclut ResizePad, Grayscale, ToTensor, Normalize, et Augmentations si train)
            processed_img = self.transform(text_crop)

            # --- Encodage du Label ---
            label = self.label_converter.encode(text)
            label_length = torch.IntTensor([len(label)])  # Longueur réelle avant padding éventuel par collate_fn

            # Gérer le cas où l'encodage donne un label vide (tous caractères inconnus)
            if len(label) == 0:
                # print(f"Warning: Encoded label is empty for text '{text}' (AnnID: {sample['ann_id']}). Assigning blank label.")
                label = torch.IntTensor([self.label_converter.blank_token_idx])  # Assigner un label 'blank'
                label_length = torch.IntTensor([1])  # Longueur 1

            # Retourner l'image traitée, le label encodé, et sa longueur
            return processed_img, label, label_length

        except FileNotFoundError:
            print(f"Error: Image file not found during __getitem__: {image_path}. Returning dummy data.")
            dummy_img = torch.zeros((1, config.IMG_HEIGHT, 10), dtype=torch.float)
            dummy_label = torch.IntTensor([self.label_converter.blank_token_idx])
            dummy_len = torch.IntTensor([1])
            return dummy_img, dummy_label, dummy_len
        except Exception as e:
            # Capturer d'autres erreurs potentielles (PIL, transformations, encodage...)
            print(
                f"Error processing sample in __getitem__ (idx {idx}, AnnID: {sample.get('ann_id', 'N/A')}): {e}. Path: {image_path}. Returning dummy data.")
            # Logguer l'erreur complète si nécessaire pour le débogage
            # import traceback
            # traceback.print_exc()
            dummy_img = torch.zeros((1, config.IMG_HEIGHT, 10), dtype=torch.float)
            dummy_label = torch.IntTensor([self.label_converter.blank_token_idx])
            dummy_len = torch.IntTensor([1])
            return dummy_img, dummy_label, dummy_len