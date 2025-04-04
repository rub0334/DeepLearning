import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T # Utilisation des nouvelles transforms v2
from PIL import Image
import os
import json
import numpy as np
from utils import set_seed, save_checkpoint, load_checkpoint, visualize_detection
# Dans data_loader.py, au début OU dans if __name__ == '__main__':
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from utils import polygon_to_bbox
import config

class TextOCRDatasetDetection(Dataset):
    def __init__(self, json_path, img_dir, transforms=None, target_img_size=None):
        """
        Dataset pour la détection de texte sur TextOCR.

        Args:
            json_path (str): Chemin vers le fichier JSON d'annotations (format COCO-Text).
            img_dir (str): Chemin vers le répertoire contenant les images.
            transforms (callable, optional): Transformations à appliquer à l'image et aux cibles.
            target_img_size (tuple, optional): Taille (H, W) vers laquelle redimensionner les images et les boîtes.
        """
        print(f"Loading annotations from: {json_path}")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotation file not found: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        print("Annotations loaded.")

        self.img_dir = img_dir
        if not os.path.exists(self.img_dir):
             raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self.transforms = transforms
        self.target_img_size = target_img_size # (height, width)

        self.imgs = data['imgs']
        self.anns = data['anns']
        self.img_to_anns = data.get('imgToAnns', None) # Utiliser get pour la compatibilité

        if self.img_to_anns is None:
             # Créer img_to_anns si manquant (peut arriver avec des formats dérivés)
             print("imgToAnns not found in JSON, creating it...")
             self.img_to_anns = {}
             for ann_id, ann in self.anns.items():
                 img_id = str(ann['image_id']) # Assurer que l'img_id est une chaîne si les clés sont des chaînes
                 if img_id not in self.img_to_anns:
                     self.img_to_anns[img_id] = []
                 self.img_to_anns[img_id].append(ann_id)
             print("imgToAnns created.")

        self.img_ids = list(self.imgs.keys())

        print(f"Dataset initialized with {len(self.img_ids)} images.")
        # Pré-vérification rapide de quelques images/annotations
        if len(self.img_ids) > 0:
             self._check_sample(0)


    def _check_sample(self, idx):
        """Vérifie si un échantillon peut être chargé."""
        try:
            img_id = self.img_ids[idx]
            img_info = self.imgs[img_id]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            if not os.path.exists(img_path):
                 print(f"Warning: Image file missing for img_id {img_id}: {img_path}")
                 # Optionnellement, on pourrait supprimer cet img_id de la liste
            # Tenter de charger les annotations
            ann_ids = self.img_to_anns.get(img_id, [])
            if not ann_ids:
                # Images sans texte sont valides mais peuvent nécessiter une gestion spéciale
                pass #print(f"Info: Image {img_id} has no text annotations.")
        except Exception as e:
             print(f"Error checking sample index {idx}, img_id {self.img_ids[idx]}: {e}")


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        try:
            # Utiliser 'L' pour charger en niveaux de gris puis convertir en RGB pour gérer les images N&B
            image = Image.open(img_path).convert("RGB")
            original_w, original_h = image.size
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}. Returning None.")
            # Retourner des placeholders ou lever une exception gérable dans collate_fn
            # Pour l'instant, on retourne None, ce qui sera filtré dans collate_fn
            return None
        except Exception as e:
            print(f"Error opening image {img_path}: {e}. Returning None.")
            return None


        ann_ids = self.img_to_anns.get(img_id, [])
        annotations = [self.anns[ann_id] for ann_id in ann_ids]

        boxes = []
        labels = [] # Pour la détection, toutes les boîtes sont de classe 'texte' (label 1)
        areas = []
        iscrowd = [] # Peut être utile pour certains modèles/métriques, ici on met 0 par défaut

        for ann in annotations:
            # Utiliser 'points' comme source de vérité géométrique
            points = ann.get('points')
            if not points: continue # Ignorer les annotations sans points

            bbox = polygon_to_bbox(points) # Convertir polygone en bbox [xmin, ymin, xmax, ymax]
            if bbox is None: continue # Ignorer si la conversion échoue (ex: points colinéaires)

            # Vérifier que la bbox est dans les limites de l'image (peut arriver avec des annotations bruitées)
            xmin, ymin, xmax, ymax = bbox
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(original_w, xmax)
            ymax = min(original_h, ymax)

            # Ignorer les boîtes dégénérées après clipping
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1) # Classe 1 pour 'texte'
            area = (xmax - xmin) * (ymax - ymin)
            areas.append(area)
            iscrowd.append(ann.get('iscrowd', 0)) # Utiliser iscrowd si disponible, sinon 0

        # Créer le dictionnaire de cibles (target)
        target = {}
        # Convertir en Tensors Float pour les boîtes, Long pour les labels
        # Utiliser torch.float32 pour les boîtes comme attendu par de nombreux modèles/fonctions de perte
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)

        # Utiliser torch.int64 pour les labels comme souvent attendu
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        # Utiliser l'index du dataset comme ID numérique pour ce tenseur
        target["image_id"] = torch.tensor([idx], dtype=torch.int64)
        target["area"] = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.empty((0,), dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.uint8) if iscrowd else torch.empty((0,), dtype=torch.uint8)

        # Appliquer les transformations (qui doivent gérer image ET cibles si nécessaire)
        # Les transforms v2 de torchvision gèrent cela
        if self.transforms:
            # Les transforms v2 attendent image et target (ou une liste/tuple)
            # Assurez-vous que vos transforms sont compatibles (ex: T.Compose([T.ToImageTensor(), T.ConvertImageDtype()]))
            # Si redimensionnement, les boîtes doivent être ajustées. T.Resize le fait automatiquement pour les clés connues comme "boxes".
            image, target = self.transforms(image, target)


        # Assurer que les boîtes sont toujours valides après transformations (certaines peuvent devenir minuscules/inversées)
        if 'boxes' in target and target['boxes'].shape[0] > 0:
             boxes = target['boxes']
             # Filtrer les boîtes potentiellement invalides après transformation
             valid_boxes_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
             target['boxes'] = boxes[valid_boxes_mask]
             # Filtrer les autres champs correspondants s'ils existent et ont la même taille initiale
             for key in ['labels', 'area', 'iscrowd']:
                  if key in target and target[key].shape[0] == valid_boxes_mask.shape[0]:
                      target[key] = target[key][valid_boxes_mask]

        return image, target


def get_detection_transforms(is_train=True, target_img_size=(640, 640)):
    """Crée les transformations pour la détection."""
    transforms = []
    # Convertit PIL Image en Tensor PyTorch et normalise les pixels en [0, 1]
    transforms.append(T.ToImage()) # Correction ici !
    transforms.append(T.ConvertImageDtype(torch.float32)) # Convertit en float32 [0, 1]

    # Redimensionnement - Important: T.Resize de v2 ajuste les boîtes automatiquement!
    if target_img_size:
        transforms.append(T.Resize(target_img_size, antialias=True)) # Garde ratio aspect initialement? Non, fixe la taille.

    if is_train:
        # Augmentation de données (optionnel mais recommandé)
        # Exemple: retournement horizontal aléatoire (ajuste les boîtes)
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        # Autres augmentations possibles : T.ColorJitter, T.RandomAffine, T.GaussianBlur etc.
        # Attention: certaines transformations (comme RandomCrop) nécessitent une gestion plus complexe des boîtes.
        pass

    # Normalisation (valeurs typiques pour modèles entraînés sur ImageNet, même si on entraîne from scratch, c'est un bon point de départ)
    # Ou utiliser la moyenne/std du dataset TextOCR si calculée. Pour "from scratch" strict, on pourrait ne pas normaliser ou utiliser 0.5/0.5
    # transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])) # Optionnel

    return T.Compose(transforms)


def collate_fn_detection(batch):
    """Fonction de collation pour DataLoader de détection.
       Gère les images de tailles potentiellement différentes (avant transfo)
       et les nombres variables de boîtes par image. Filtre les None."""
    # Filtrer les échantillons None (qui résultent d'erreurs de chargement)
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: # Si le batch est vide après filtrage
        return None, None # Ou lever une exception/retourner des tenseurs vides

    images, targets = zip(*batch)
    # Les images devraient avoir la même taille après les transformations (si Resize est utilisé)
    # Empiler les images dans un batch
    images = torch.stack(images, 0)
    # Les cibles sont des listes de dictionnaires, elles restent comme ça pour la plupart des modèles de détection
    return images, list(targets)


def create_detection_dataloaders(train_json, val_json, train_img_dir, val_img_dir, batch_size, num_workers, img_size):
    """Crée les DataLoaders pour l'entraînement et la validation de la détection."""

    train_transforms = get_detection_transforms(is_train=True, target_img_size=img_size)
    val_transforms = get_detection_transforms(is_train=False, target_img_size=img_size)

    train_dataset = TextOCRDatasetDetection(
        json_path=train_json,
        img_dir=train_img_dir,
        transforms=train_transforms,
        target_img_size=img_size # Redondant si déjà dans transforms, mais OK
    )
    val_dataset = TextOCRDatasetDetection(
        json_path=val_json,
        img_dir=val_img_dir,
        transforms=val_transforms,
        target_img_size=img_size # Redondant
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_detection,
        pin_memory=True # Améliore potentiellement le transfert CPU -> GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, # Souvent plus petit ou égal au batch size d'entraînement
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_detection,
        pin_memory=True
    )

    print(f"Train DataLoader: {len(train_loader)} batches, {len(train_dataset)} samples.")
    print(f"Validation DataLoader: {len(val_loader)} batches, {len(val_dataset)} samples.")

    return train_loader, val_loader

# Exemple d'utilisation (pour tester)
if __name__ == '__main__':
    print("Testing TextOCR Dataset and DataLoader creation...")
    set_seed(config.SEED)

    # Utiliser les chemins de config.py
    train_json_path = config.TRAIN_JSON
    train_image_directory = config.TRAIN_IMG_DIR
    val_json_path = config.VAL_JSON
    val_image_directory = config.VAL_IMG_DIR
    batch_sz = config.BATCH_SIZE
    num_w = config.NUM_WORKERS
    img_sz = config.IMG_SIZE

    try:
        train_loader, val_loader = create_detection_dataloaders(
            train_json=train_json_path,
            val_json=val_json_path,
            train_img_dir=train_image_directory,
            val_img_dir=val_image_directory,
            batch_size=batch_sz,
            num_workers=num_w,
            img_size=img_sz
        )

        print("\nDataLoaders created successfully.")

        # Afficher un batch d'entraînement pour vérification
        print("\nFetching one batch from train_loader...")
        images, targets = next(iter(train_loader))

        if images is not None and targets is not None:
            print("Batch fetched successfully:")
            print(f"Images shape: {images.shape}") # Devrait être [batch_size, C, H, W]
            print(f"Targets length: {len(targets)}") # Devrait être batch_size
            print("Sample target (first item in batch):")
            # Afficher les clés et formes des tenseurs dans la première cible
            if targets:
                first_target = targets[0]
                for key, value in first_target.items():
                     if isinstance(value, torch.Tensor):
                         print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                     else:
                         print(f"  - {key}: {value}")

                # Visualiser la première image du batch avec ses boîtes ground truth
                print("\nVisualizing first image of the batch...")
                from utils import visualize_detection # Importer ici pour éviter dépendance circulaire si appelé directement
                # Dénormaliser l'image si la normalisation a été appliquée dans les transforms
                # (Ici, on n'a pas normalisé dans get_detection_transforms pour l'instant)
                # Convertir l'image tensor (C, H, W) en format (H, W, C) pour matplotlib
                img_to_show = images[0].permute(1, 2, 0).cpu().numpy()
                # Les pixels sont en [0, 1], matplotlib les gère. Si >1 ou négatif, clipper/ajuster.
                img_to_show = np.clip(img_to_show, 0, 1)

                # Créer un chemin temporaire pour la visualisation
                vis_dir = os.path.join(config.BASE_PROJECT_DIR, "visualizations")
                vis_path = os.path.join(vis_dir, "train_batch_sample_detection.png")

                 # Besoin de convertir les coordonnées des boîtes si elles ont été normalisées ou si l'image a été redimensionnée
                 # Ici, les transforms T.Resize ajustent les boîtes automatiquement aux coordonnées de l'image redimensionnée.
                visualize_detection(
                    image_path=None, # On fournit directement l'image traitée
                    targets=first_target,
                    predictions=None,
                    output_path=vis_path
                )
                # Pour afficher l'image directement à partir du tensor:
                fig, ax = plt.subplots(1)
                ax.imshow(img_to_show)
                for box in first_target['boxes']:
                    xmin, ymin, xmax, ymax = box.cpu().numpy()
                    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
                plt.axis('off')
                plt.savefig(vis_path.replace(".png", "_direct.png"))
                plt.close(fig)
                print(f"Sample visualization saved to {vis_path} and {vis_path.replace('.png', '_direct.png')}")


            else:
                 print("No targets found in the first batch item.")

        else:
            print("Failed to fetch a valid batch. Check dataset/collate function.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the paths in config.py are correct and the data exists.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()