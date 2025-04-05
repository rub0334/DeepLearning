# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
# PAS besoin de torchvision.transforms ici
from PIL import Image, ImageDraw
import json
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 # Importer OpenCV explicitement

import config
import utils

class TextOCRDetectionDataset(Dataset):
    """
    Dataset PyTorch pour la TÂCHE DE DÉTECTION sur TextOCR.
    Charge les images et les annotations de boîtes englobantes.
    Applique les transformations via Albumentations.
    """
    def __init__(self, annotation_file, image_dir, albumentations_transform=None):
        print(f"Chargement des annotations depuis: {annotation_file}")
        print(f"Chargement des images depuis: {image_dir}")

        self.image_dir = image_dir
        self.transforms = albumentations_transform

        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.annotations = data['anns']
        self.img_to_anns = data['imgToAnns']
        self.images_info = data['imgs']
        self.image_ids = [img_id for img_id in data['imgs'].keys() if img_id in self.img_to_anns and len(self.img_to_anns[img_id]) > 0]

        print(f"Nombre total d'images dans le JSON: {len(data['imgs'])}")
        print(f"Nombre d'images avec annotations utilisables: {len(self.image_ids)}")

        # Créer un pipeline de fallback minimal (juste ToTensorV2) une seule fois
        self.minimal_transform = A.Compose([ToTensorV2()],
                                           bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images_info[image_id]
        json_file_path_part = img_info['file_name']
        image_filename = os.path.basename(json_file_path_part)
        image_path = os.path.join(self.image_dir, image_filename)

        try:
            image = np.array(Image.open(image_path).convert("RGB"))
            img_height, img_width, _ = image.shape
        except FileNotFoundError:
            print(f"ERREUR (Dataset): Image non trouvée à {image_path}.")
            raise FileNotFoundError(f"Image not found during dataset access: {image_path}")
        except Exception as e:
            print(f"ERREUR (Dataset): Impossible de charger l'image {image_path}: {e}")
            raise e

        annotation_ids = self.img_to_anns.get(image_id, [])
        boxes = []
        labels = []

        for ann_id in annotation_ids:
            ann = self.annotations[ann_id]
            points = ann.get('points')
            bbox = utils.get_bounding_box_from_points(points)

            if bbox is None: continue
            xmin, ymin, xmax, ymax = map(float, bbox)
            xmin = max(0.0, xmin)
            ymin = max(0.0, ymin)
            xmax = min(float(img_width), xmax)
            ymax = min(float(img_height), ymax)
            if xmax <= xmin or ymax <= ymin: continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)

        # Variables pour stocker le résultat final
        final_image_tensor = None
        final_boxes = []
        final_labels = []

        # Appliquer les transformations principales
        if self.transforms:
            try:
                transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)
                # ToTensorV2 devrait être appliqué ici si c'est la dernière étape
                final_image_tensor = transformed['image']
                final_boxes = transformed['bboxes']
                final_labels = transformed['class_labels']

            except ValueError as e_alb: # Erreur de coordonnées/bbox
                print(f"AVERTISSEMENT: Erreur ValueError Albumentations sur image {image_id}: {e_alb}. Tentative avec ToTensorV2 seul.")
                # Appliquer le fallback minimal
                try:
                   transformed = self.minimal_transform(image=image, bboxes=boxes, class_labels=labels)
                   final_image_tensor = transformed['image']
                   final_boxes = transformed['bboxes']
                   final_labels = transformed['class_labels']
                except Exception as e_fallback:
                   print(f"ERREUR critique même avec ToTensorV2 seul sur image {image_id}: {e_fallback}")
                   # Retourner des tenseurs vides pour éviter de planter le batch? Ou lever une erreur?
                   # Lever une erreur est plus sûr pour l'instant.
                   raise RuntimeError(f"Impossible de traiter l'image {image_id}.") from e_fallback

            except Exception as e: # Autres erreurs Albumentations
                print(f"ERREUR Inattendue Albumentations sur image {image_id}: {e}. Tentative avec ToTensorV2 seul.")
                # Appliquer le fallback minimal
                try:
                   transformed = self.minimal_transform(image=image, bboxes=boxes, class_labels=labels)
                   final_image_tensor = transformed['image']
                   final_boxes = transformed['bboxes']
                   final_labels = transformed['class_labels']
                except Exception as e_fallback:
                   print(f"ERREUR critique même avec ToTensorV2 seul sur image {image_id}: {e_fallback}")
                   raise RuntimeError(f"Impossible de traiter l'image {image_id}.") from e_fallback

        # Si aucune transformation n'a été définie (ex: validation avant Albumentations)
        # Ou si final_image_tensor n'a pas été assigné (ne devrait pas arriver avec fallback)
        if final_image_tensor is None:
             print(f"INFO: Aucune transformation principale, application de ToTensorV2 pour image {image_id}")
             try:
                transformed = self.minimal_transform(image=image, bboxes=boxes, class_labels=labels)
                final_image_tensor = transformed['image']
                final_boxes = transformed['bboxes']
                final_labels = transformed['class_labels']
             except Exception as e_fallback:
                 print(f"ERREUR critique avec ToTensorV2 sur image {image_id} sans transfo préalable: {e_fallback}")
                 raise RuntimeError(f"Impossible de traiter l'image {image_id}.") from e_fallback


        # --- Vérification Finale et Formatage de la Target ---
        # Assurer que l'image est FloatTensor [0, 1]
        if final_image_tensor.dtype != torch.float32:
            # print(f"DEBUG: Correction dtype pour image {image_id}. Type initial: {final_image_tensor.dtype}") # Décommenter pour debug
            final_image_tensor = final_image_tensor.to(torch.float32).div(255.0)
        elif final_image_tensor.max() > 1.0: # Vérifier si ToTensorV2 a bien divisé par 255
             # print(f"DEBUG: Correction range pour image {image_id}. Max initial: {final_image_tensor.max()}") # Décommenter pour debug
             final_image_tensor = final_image_tensor.div(255.0)


        # Traitement des boîtes et labels finaux
        if not final_boxes or len(final_boxes) == 0:
            target_boxes = torch.empty((0, 4), dtype=torch.float32)
            target_labels = torch.empty((0,), dtype=torch.int64)
            target_area = torch.empty((0,), dtype=torch.float32)
            target_iscrowd = torch.zeros((0,), dtype=torch.uint8)
        else:
            target_boxes_tensor = torch.tensor(final_boxes, dtype=torch.float32)
            target_labels_tensor = torch.tensor(final_labels, dtype=torch.int64)

            valid_boxes_mask = (target_boxes_tensor[:, 2] > target_boxes_tensor[:, 0]) & (target_boxes_tensor[:, 3] > target_boxes_tensor[:, 1])
            target_boxes = target_boxes_tensor[valid_boxes_mask]
            target_labels = target_labels_tensor[valid_boxes_mask]

            if target_boxes.shape[0] == 0:
                target_area = torch.empty((0,), dtype=torch.float32)
                target_iscrowd = torch.zeros((0,), dtype=torch.uint8)
            else:
                target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
                target_iscrowd = torch.zeros((target_boxes.shape[0],), dtype=torch.uint8)

        target = {}
        target["boxes"] = target_boxes
        target["labels"] = target_labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = target_area
        target["iscrowd"] = target_iscrowd

        return final_image_tensor, target # Retourner le tenseur image final

# --- Fonction Collate (inchangée) ---
def collate_fn(batch):
    # Filtrer les éléments None si une erreur dans __getitem__ retournait None
    # batch = [b for b in batch if b is not None]
    # if not batch: return None, None # Gérer le cas où tout un batch échoue
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets




# --- Définition des Pipelines Albumentations (Avec Corrections) ---
def get_train_transform():
    """Transformations pour l'entraînement avec Albumentations."""
    return A.Compose([
        # --- Augmentations Géométriques ---
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.0625, 0.0625),
            rotate=(-15, 15),
            # --- CORRECTION ICI ---
            # Spécifier 0 ou une plage nulle pour désactiver le cisaillement
            shear=0,
            # shear=(-5, 5), # Alternative: petit cisaillement aléatoire
            # --- FIN CORRECTION ---
            cval=0,
            p=0.7
        ),
        A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3),

        # --- Augmentations Photométriques ---
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.4),

        # --- Normalisation et Conversion Tensor ---
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc',
                                label_fields=['class_labels'],
                                min_visibility=0.1,
                                min_area=10)
      )


def get_val_transform():
    """Transformations pour la validation/test."""
    return A.Compose([
        ToTensorV2(), # Juste convertir en FloatTensor [0-1]
    ])

# --- Fonction create_dataloaders (vérifier import cv2) ---
def create_dataloaders(batch_size):
    print("Création des DataLoaders avec Albumentations...")
    global cv2
    try:
        import cv2
    except ImportError:
        # Affine peut fonctionner sans cv2 pour certains modes, mais mieux vaut l'avoir
        print("AVERTISSEMENT: cv2 (OpenCV) non trouvé. Certaines augmentations pourraient échouer ou être limitées.")
        print("Veuillez installer via 'pip install opencv-python-headless'")
        # On continue, mais Affine pourrait planter
        # raise ImportError("OpenCV (cv2) est requis...") # Optionnel: arrêter ici

    train_dataset = TextOCRDetectionDataset(
        annotation_file=config.TRAIN_ANNOTATION_FILE,
        image_dir=config.TRAIN_IMAGE_DIR,
        albumentations_transform=get_train_transform()
    )
    val_dataset = TextOCRDetectionDataset(
        annotation_file=config.VAL_ANNOTATION_FILE,
        image_dir=config.VAL_IMAGE_DIR,
        albumentations_transform=get_val_transform()
    )

    print(f"Taille du Dataset d'entraînement: {len(train_dataset)}")
    print(f"Taille du Dataset de validation: {len(val_dataset)}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Un des datasets est vide...")

    num_workers = 4
    if os.name == 'nt' and num_workers > 0:
         # num_workers = 0 # Mettre à 0 si erreurs multiprocessing persistent
         print(f"Info: num_workers={num_workers} sous Windows. Mettre à 0 si des erreurs surviennent.")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    return train_loader, val_loader

# --- Bloc if __name__ == '__main__' (inchangé) ---
if __name__ == '__main__':
    print("Test du DataLoader avec Albumentations...")
    try:
        train_loader, val_loader = create_dataloaders(batch_size=2)

        print("\nTest récupération d'un batch d'entraînement:")
        images, targets = next(iter(train_loader))
        print(f"Nombre d'images dans le batch: {len(images)}")
        print(f"Type image[0]: {type(images[0])}, Shape: {images[0].shape}, Dtype: {images[0].dtype}, Min: {images[0].min():.2f}, Max: {images[0].max():.2f}")
        print(f"Nombre de cibles dans le batch: {len(targets)}")
        print(f"Type target[0]: {type(targets[0])}")
        print(f"Clés target[0]: {targets[0].keys()}")
        print(f"Exemple boxes target[0] (après transfo): {targets[0]['boxes'].shape}")
        print(f"Exemple labels target[0] (après transfo): {targets[0]['labels'].shape}")

        print("\nTest récupération d'un batch de validation:")
        images_val, targets_val = next(iter(val_loader))
        print(f"Nombre d'images dans le batch val: {len(images_val)}")
        print(f"Shape image_val[0]: {images_val[0].shape}, Dtype: {images_val[0].dtype}")
        print(f"Nombre de cibles dans le batch val: {len(targets_val)}")

    except Exception as e:
        print(f"\nERREUR lors du test du DataLoader : {e}")
        import traceback
        traceback.print_exc()
        print("\nVérifiez les installations (albumentations, opencv-python-headless), les chemins dans config.py et l'intégrité des fichiers JSON/images.")

    print("\nTest de data_loader.py terminé.")