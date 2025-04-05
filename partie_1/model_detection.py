# model_detection.py
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.nn as nn

import config # Importer notre configuration

def get_detection_model(num_classes=config.NUM_CLASSES):
    """
    Construit le modèle Faster R-CNN pour la détection de texte.

    Args:
        num_classes (int): Nombre de classes (incluant la classe 'background').

    Returns:
        torch.nn.Module: Le modèle Faster R-CNN.
    """
    print("Construction du modèle de détection...")

    # --- Option 1: Utiliser un backbone ResNet standard SANS pré-entraînement ---
    # Charger un backbone ResNet (ex: ResNet-50) SANS les poids pré-entraînés ImageNet
    # ATTENTION: Entraîner un ResNet-50 from scratch est TRES gourmand et long.
    # Pour un test plus rapide, on pourrait utiliser ResNet-18 ou ResNet-34.
    backbone = torchvision.models.resnet50(pretrained=False, progress=True) # pretrained=False est CRUCIAL

    # Retirer la couche de classification finale (avgpool et fc) du ResNet
    # Faster R-CNN utilise les feature maps avant la classification.
    # Pour ResNet, les couches utiles se terminent généralement à 'layer4'.
    # Nous avons besoin de connaître le nombre de canaux de sortie de la dernière couche utilisée.
    # Pour ResNet-50, la sortie de layer4 a 2048 canaux.
    modules = list(backbone.children())[:-2] # Enlève avgpool et fc
    backbone = nn.Sequential(*modules)
    backbone.out_channels = 2048 # Spécifier manuellement pour ResNet-50

    print(f"  Backbone: ResNet-50 (non pré-entraîné), Output channels: {backbone.out_channels}")

    # --- Option 2: Utiliser un backbone plus léger (ex: MobileNet) ---
    # backbone = torchvision.models.mobilenet_v2(pretrained=False).features
    # backbone.out_channels = 1280 # Pour MobileNetV2

    # Générateur d'ancres pour le RPN (Region Proposal Network)
    # Ces tailles et ratios peuvent nécessiter un ajustement en fonction de la taille
    # typique des objets texte dans TextOCR et de la taille des images d'entrée.
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), # Tailles des ancres
                                       aspect_ratios=((0.5, 1.0, 2.0),)) # Ratios L/H

    print("  Anchor Generator configuré.")

    # Feature Pyramid Network (FPN) est souvent utilisé avec Faster R-CNN pour
    # améliorer la détection d'objets à différentes échelles.
    # Il prend les sorties de différentes couches du backbone.
    # Pour l'utiliser, il faut un backbone compatible (comme ceux de torchvision.models.detection.backbone_utils)
    # Simplifions pour l'instant et utilisons seulement la sortie de la dernière couche du backbone.
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    # Pour utiliser uniquement la sortie de la dernière couche du backbone séquentiel:
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], # '0' car backbone est un nn.Sequential simple
                                                     output_size=7,      # Taille de la feature map après RoI pooling
                                                     sampling_ratio=2)

    print("  RoI Pooler configuré (MultiScaleRoIAlign).")


    # Construire le modèle Faster R-CNN
    model = FasterRCNN(backbone,
                       num_classes=num_classes, # Background + Text
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    print(f"  Modèle Faster R-CNN construit avec {num_classes} classes.")

    # --- Remplacer la tête de classification pré-entraînée (si on avait utilisé pretrained=True) ---
    # Bien que nous n'utilisions pas de poids pré-entraînés pour le backbone,
    # la structure FasterRCNN pourrait initialiser sa tête avec des poids par défaut.
    # Il est bon de s'assurer qu'elle correspond à notre nombre de classes.
    # (Techniquement non nécessaire avec pretrained=False, mais bonne pratique)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print(f"  Tête de classification remplacée pour {num_classes} classes (Input Features: {in_features}).")


    return model

if __name__ == '__main__':
    # Test pour vérifier la construction du modèle
    print("Test de la construction du modèle...")
    model = get_detection_model()
    print("\nArchitecture du Modèle (partiel):")
    # print(model) # Affiche toute l'architecture, peut être très long

    # Test avec une image factice
    print("\nTest forward pass avec image factice:")
    try:
        model.eval() # Mettre en mode évaluation pour le test
        # Créer une image factice (Batch de 1, 3 canaux, Hauteur, Largeur)
        dummy_image = torch.randn(1, 3, 640, 480)
        # Créer une cible factice (pour info, non utilisée en mode eval pur)
        dummy_target = [{
            'boxes': torch.tensor([[10, 10, 100, 100]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        }]

        with torch.no_grad():
            output = model([dummy_image[0]]) # Le modèle attend une liste d'images

        print("Forward pass réussi.")
        print(f"Output type: {type(output)}")
        print(f"Output length (batch size): {len(output)}")
        print(f"Output[0] keys: {output[0].keys()}") # Doit contenir 'boxes', 'labels', 'scores'
        print(f"Output[0]['boxes'].shape: {output[0]['boxes'].shape}")
    except Exception as e:
        print(f"Erreur durant le forward pass factice: {e}")

    print("\nTest de model_detection.py terminé.")