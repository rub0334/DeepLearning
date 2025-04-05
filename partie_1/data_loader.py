# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw # Pillow pour charger les images
import json
import os
import numpy as np

import config # Importer notre configuration
import utils  # Importer nos utilitaires (pour get_bounding_box_from_points)

class TextOCRDetectionDataset(Dataset):
    """
    Dataset PyTorch pour la TÂCHE DE DÉTECTION sur TextOCR.
    Charge les images et les annotations de boîtes englobantes.
    """
    def __init__(self, annotation_file, image_dir, transforms=None):
        """
        Args:
            annotation_file (str): Chemin vers le fichier JSON d'annotations (train ou val).
            image_dir (str): Chemin vers le dossier contenant les images.
            transforms (callable, optional): Transformations à appliquer sur les images.
        """
        print(f"Chargement des annotations depuis: {annotation_file}")
        print(f"Chargement des images depuis: {image_dir}")

        self.image_dir = image_dir
        self.transforms = transforms

        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.annotations = data['anns']
        self.img_to_anns = data['imgToAnns']
        self.images_info = data['imgs']

        # Filtrer les images pour ne garder que celles présentes dans imgToAnns
        # (Certaines images peuvent ne pas avoir d'annotations)
        self.image_ids = [img_id for img_id in data['imgs'].keys() if img_id in self.img_to_anns and len(self.img_to_anns[img_id]) > 0]

        print(f"Nombre total d'images dans le JSON: {len(data['imgs'])}")
        print(f"Nombre d'images avec annotations utilisables: {len(self.image_ids)}")


    def __len__(self):
        """Retourne le nombre d'images dans le dataset."""
        return len(self.image_ids)



    def __getitem__(self, idx):
        """
        Retourne une image et ses annotations associées.

        Args:
            idx (int): Index de l'image à récupérer.

        Returns:
            tuple: (image, target) où
                - image (torch.Tensor): Image transformée.
                - target (dict): Dictionnaire contenant les clés:
                    - 'boxes' (torch.Tensor): Boîtes englobantes [N, 4] au format [xmin, ymin, xmax, ymax].
                    - 'labels' (torch.Tensor): Étiquettes de classe [N] (toujours 1 pour 'text').
                    - 'image_id' (torch.Tensor): ID de l'image (index du dataset ici).
                    - 'area' (torch.Tensor): Aire des boîtes englobantes [N].
                    - 'iscrowd' (torch.Tensor): Indicateur de foule [N] (toujours 0 ici).
        """
        image_id = self.image_ids[idx]
        img_info = self.images_info[image_id]
        json_file_path_part = img_info['file_name']  # Ex: "train/abc.jpg" ou "val/xyz.jpg"

        # --- CORRECTION/SIMPLIFICATION PATH CONSTRUCTION ---
        # Extraire seulement le nom du fichier depuis le chemin dans le JSON
        image_filename = os.path.basename(json_file_path_part)  # Obtient "abc.jpg" ou "xyz.jpg"

        # Utiliser directement self.image_dir qui a été fourni lors de la création du Dataset
        # Pour train_dataset, self.image_dir sera config.TRAIN_IMAGE_DIR
        # Pour val_dataset, self.image_dir sera config.VAL_IMAGE_DIR
        image_path = os.path.join(self.image_dir, image_filename)
        # --- FIN CORRECTION/SIMPLIFICATION ---

        try:
            # Charger l'image en RGB
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size
        except FileNotFoundError:
            # L'erreur sera maintenant plus directe si le self.image_dir était incorrectement défini
            # ou si le fichier manque réellement dans le bon dossier.
            print(
                f"ERREUR (Dataset): Image non trouvée à {image_path}. Vérifiez que le fichier existe dans {self.image_dir} et que les configs sont correctes.")
            raise FileNotFoundError(f"Image not found during dataset access: {image_path}")
        except Exception as e:
            print(f"ERREUR (Dataset): Impossible de charger l'image {image_path}: {e}")
            raise e  # Re-lève l'exception

        # --- Traitement des annotations ---
        annotation_ids = self.img_to_anns.get(image_id, [])
        boxes = []
        labels = []
        areas = []

        for ann_id in annotation_ids:
            ann = self.annotations[ann_id]

            # Utiliser les 'points' pour dériver la bbox horizontale (source de vérité)
            points = ann.get('points')
            bbox = utils.get_bounding_box_from_points(points)  # Format [xmin, ymin, xmax, ymax]

            if bbox is None:  # Ignorer si la boîte est invalide/dégénérée
                continue

            # S'assurer que les coordonnées sont dans les limites de l'image chargée
            xmin, ymin, xmax, ymax = bbox
            # Convertir en float avant max/min pour éviter les erreurs de type si les points sont des entiers
            # et s'assurer que les dimensions de l'image sont aussi des floats pour la comparaison
            img_width_f = float(img_width)
            img_height_f = float(img_height)
            xmin = max(0.0, float(xmin))
            ymin = max(0.0, float(ymin))
            xmax = min(img_width_f, float(xmax))
            ymax = min(img_height_f, float(ymax))

            # Vérifier à nouveau après le clipping si la boîte est toujours valide (aire > 0)
            if xmax <= xmin or ymax <= ymin:
                # Optionnel : décommenter pour voir les boîtes dégénérées après clipping
                # print(f"Warning: Box {ann_id} for image {image_id} became degenerate after clipping. Original: {bbox}, Clipped: {[xmin, ymin, xmax, ymax]}")
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Classe 1 pour 'text' (classe 0 réservée au fond)
            # L'aire est calculée sur la bbox clippée
            areas.append((xmax - xmin) * (ymax - ymin))

        # Conversion en Tensors PyTorch
        if not boxes:  # S'il n'y a AUCUNE annotation valide pour cette image
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            areas = torch.empty((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.uint8)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            # Pour TextOCR, nous n'avons pas d'info 'iscrowd', on met tout à 0
            iscrowd = torch.zeros((len(boxes),), dtype=torch.uint8)

        # Création du dictionnaire 'target' attendu par les modèles torchvision
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # Utiliser l'index du dataset comme ID simple ici, bien que l'image_id original soit aussi disponible
        target["image_id"] = torch.tensor([idx])
        target["area"] = areas
        target["iscrowd"] = iscrowd

        # Appliquer les transformations à l'image
        if self.transforms:
            # Note: Si les transformations incluent des opérations géométriques (resize, crop),
            # il faudra aussi transformer les 'boxes' dans la target.
            # Les transformations standard comme ToTensor, Normalize n'affectent pas les boxes.
            image = self.transforms(image)

        return image, target
# Fonction Collate pour gérer les lots (batches) d'images et de cibles de tailles variables
def collate_fn(batch):
    """
    Combine une liste de tuples (image, target) en un batch.
    Nécessaire car les 'targets' (dictionnaires avec des tenseurs) ne peuvent pas
    être empilés automatiquement par le DataLoader par défaut si le nombre
    d'annotations varie entre les images.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    # Les images sont déjà des tenseurs (ou seront empilées par le DataLoader si ToTensor est appliqué)
    # On ne les empile pas ici, mais on les retourne comme une liste de tenseurs.
    # Le moteur d'entraînement s'attend à une liste d'images et une liste de cibles.
    # Note: Si les images n'ont pas toutes la même taille APRES transformation, il faudra les empiler
    # ou utiliser une gestion de batch spécifique (padding). Pour Faster R-CNN avec FPN,
    # les tailles variables sont généralement gérées en interne.
    return images, targets


# Transformations standard pour les modèles torchvision
# (On n'inclut pas de redimensionnement ici pour l'instant, Faster R-CNN peut gérer différentes tailles,
# mais un redimensionnement à une taille fixe ou dans une plage peut améliorer les performances/stabilité)
def get_transform(train):
    """Applique les transformations de base."""
    transforms_list = []
    # Convertit l'image PIL (H, W, C) en tensor FloatTensor (C, H, W) dans [0.0, 1.0]
    transforms_list.append(transforms.ToTensor())
    # On pourrait ajouter la normalisation si on utilisait un backbone pré-entraîné,
    # mais pour un entraînement from scratch, ce n'est pas strictement nécessaire au début.
    # transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # Augmentation de données simple pour l'entraînement (Optionnel)
    # if train:
    #     transforms_list.append(transforms.RandomHorizontalFlip(0.5))

    return transforms.Compose(transforms_list)


# Fonction pour créer les DataLoaders
def create_dataloaders(batch_size):
    """Crée les DataLoaders pour l'entraînement et la validation."""

    train_dataset = TextOCRDetectionDataset(
        annotation_file=config.TRAIN_ANNOTATION_FILE,
        image_dir=config.TRAIN_IMAGE_DIR,
        transforms=get_transform(train=True)
    )

    val_dataset = TextOCRDetectionDataset(
        annotation_file=config.VAL_ANNOTATION_FILE,
        image_dir=config.VAL_IMAGE_DIR, # Utiliser le bon dossier pour la validation
        transforms=get_transform(train=False)
    )

    print(f"Taille du Dataset d'entraînement: {len(train_dataset)}")
    print(f"Taille du Dataset de validation: {len(val_dataset)}")

    # Si le dataset est vide, c'est probablement un problème de chemin ou de filtre
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Un des datasets est vide. Vérifiez les chemins d'accès aux données et aux annotations dans config.py, ainsi que la structure des fichiers JSON.")


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4, # Ajuster selon votre CPU/système (0 sous Windows peut être plus stable)
        collate_fn=collate_fn,
        pin_memory=True # Accélère le transfert CPU -> GPU si la mémoire le permet
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, # Souvent on peut utiliser un batch_size plus grand en validation
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader

if __name__ == '__main__':
    # Petit test pour vérifier que le chargement fonctionne
    print("Test du DataLoader...")
    train_loader, val_loader = create_dataloaders(batch_size=2)

    print("\nTest récupération d'un batch d'entraînement:")
    try:
        images, targets = next(iter(train_loader))
        print(f"Nombre d'images dans le batch: {len(images)}")
        print(f"Type image[0]: {type(images[0])}, Shape: {images[0].shape}")
        print(f"Nombre de cibles dans le batch: {len(targets)}")
        print(f"Type target[0]: {type(targets[0])}")
        print(f"Clés target[0]: {targets[0].keys()}")
        print(f"Exemple boxes target[0]: {targets[0]['boxes'].shape}")
        print(f"Exemple labels target[0]: {targets[0]['labels'].shape}")
    except Exception as e:
        print(f"Erreur lors de la récupération d'un batch : {e}")
        print("Vérifiez les chemins dans config.py et l'intégrité des fichiers JSON/images.")


    print("\nTest récupération d'un batch de validation:")
    try:
        images_val, targets_val = next(iter(val_loader))
        print(f"Nombre d'images dans le batch val: {len(images_val)}")
        print(f"Shape image_val[0]: {images_val[0].shape}")
        print(f"Nombre de cibles dans le batch val: {len(targets_val)}")
    except Exception as e:
        print(f"Erreur lors de la récupération d'un batch de validation : {e}")

    print("\nTest de data_loader.py terminé.")