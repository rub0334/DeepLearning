import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# --- Blocs de Construction (BasicBlock reste identique) ---
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# --- Backbone Modifié (Retourne plusieurs niveaux) ---
class ResNetBackboneScratch(nn.Module):
    """Backbone simplifié inspiré de ResNet, retourne C3, C4, C5."""
    def __init__(self, block, layers, num_input_channels=3):
        super().__init__()
        self.in_channels = 64
        self._out_channels = {} # Pour stocker les canaux de sortie

        self.conv1 = nn.Conv2d(num_input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # /4

        self.layer1 = self._make_layer(block, 64, layers[0])         # /4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # C3, /8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # C4, /16
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # C5, /32

        # Stocker les canaux de sortie pour FPN
        # Note: Ces noms (C3, C4, C5) sont conceptuels. Les indices correspondent aux layers.
        self._out_channels = {
            "C3": 128 * block.expansion,
            "C4": 256 * block.expansion,
            "C5": 512 * block.expansion,
        }

        self._initialize_weights()

    def get_out_channels(self) -> Dict[str, int]:
        return self._out_channels

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # /4
        l1 = self.layer1(x) # /4
        l2 = self.layer2(l1) # C3 /8
        l3 = self.layer3(l2) # C4 /16
        l4 = self.layer4(l3) # C5 /32

        # Retourner les features nécessaires pour FPN
        return {"C3": l2, "C4": l3, "C5": l4}

# --- Module FPN ---
class FPN(nn.Module):
    """Feature Pyramid Network (FPN) simple."""
    def __init__(self, in_channels_dict: Dict[str, int], out_channels: int = 256):
        """
        Args:
            in_channels_dict (Dict[str, int]): Canaux d'entrée pour C3, C4, C5.
                                                Ex: {'C3': 128, 'C4': 256, 'C5': 512}
            out_channels (int): Nombre de canaux de sortie pour chaque niveau P.
        """
        super().__init__()
        self.out_channels = out_channels

        # Connexions latérales (1x1 conv pour harmoniser les canaux)
        self.lat_conv_c3 = nn.Conv2d(in_channels_dict["C3"], out_channels, kernel_size=1)
        self.lat_conv_c4 = nn.Conv2d(in_channels_dict["C4"], out_channels, kernel_size=1)
        self.lat_conv_c5 = nn.Conv2d(in_channels_dict["C5"], out_channels, kernel_size=1)

        # Couches 3x3 pour affiner les features fusionnées (produisent P3, P4, P5)
        self.output_conv_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_conv_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_conv_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Optionnel: P6 (souvent utilisé dans RetinaNet/EfficientDet) - MaxPool sur P5 ou C5
        # self.p6 = nn.MaxPool2d(kernel_size=1, stride=2) # Appliqué sur P5

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1) # Kaiming uniform pour FPN (vu dans detectron2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features (Dict[str, Tensor]): Dictionnaire contenant C3, C4, C5.

        Returns:
            Dict[str, Tensor]: Dictionnaire contenant P3, P4, P5 (et P6 si activé).
        """
        c3, c4, c5 = features["C3"], features["C4"], features["C5"]

        # Chemin Top-Down
        p5_lat = self.lat_conv_c5(c5)
        p4_lat = self.lat_conv_c4(c4)
        p3_lat = self.lat_conv_c3(c3)

        # Upsample et addition (interpolation nearest pour simplicité, ou bilinear)
        p4_topdown = F.interpolate(p5_lat, scale_factor=2, mode="nearest")
        p4_merged = p4_lat + p4_topdown # Fusion P4

        p3_topdown = F.interpolate(p4_merged, scale_factor=2, mode="nearest")
        p3_merged = p3_lat + p3_topdown # Fusion P3

        # Couches finales 3x3 pour générer les P levels
        p5 = self.output_conv_p5(p5_lat)
        p4 = self.output_conv_p4(p4_merged)
        p3 = self.output_conv_p3(p3_merged)

        pyramid_features = {"P3": p3, "P4": p4, "P5": p5}

        # Ajouter P6 si désiré
        # p6 = self.p6(p5)
        # pyramid_features["P6"] = p6

        return pyramid_features


# --- Tête de Détection (peut rester simple pour l'instant) ---
class SimpleDetectionHead(nn.Module):
    """Tête de détection simple (classification + régression boîte)."""
    def __init__(self, in_channels, num_classes, num_convs=4, prior_prob=0.01):
        super().__init__()
        self.num_classes = num_classes

        # Couches Conv partagées (optionnel mais courant)
        shared_convs = []
        for _ in range(num_convs):
            shared_convs.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            shared_convs.append(nn.ReLU()) # Ajouter ReLU ici
        self.shared_convs = nn.Sequential(*shared_convs)

        # Tête de classification
        # Sortie: N * num_classes * H * W (logits)
        self.cls_conv = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)

        # Tête de régression
        # Sortie: N * 4 * H * W (dx, dy, dw, dh) - 4 valeurs par localisation
        self.bbox_conv = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)

        self._initialize_weights(prior_prob)

    def _initialize_weights(self, prior_prob):
         # Initialisation Kaiming pour les couches partagées
        for m in self.shared_convs.modules():
             if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                  if m.bias is not None: # Généralement pas de biais si BN suit, mais ici pas de BN dans la tête
                       nn.init.constant_(m.bias, 0)

        # Initialisation spécifique pour les têtes (style RetinaNet/FCOS)
        # Classification : Normal std=0.01, bias spécial pour Focal Loss
        nn.init.normal_(self.cls_conv.weight, std=0.01)
        if self.cls_conv.bias is not None:
             bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
             nn.init.constant_(self.cls_conv.bias, bias_value)

        # Régression : Normal std=0.01, bias=0
        nn.init.normal_(self.bbox_conv.weight, std=0.01)
        if self.bbox_conv.bias is not None:
             nn.init.constant_(self.bbox_conv.bias, 0)

    def forward(self, features):
        # features: Sortie d'un niveau FPN (ex: P3) [N, C, H, W]
        shared_out = self.shared_convs(features)
        cls_logits = self.cls_conv(shared_out) # [N, num_classes, H, W]
        bbox_pred = self.bbox_conv(shared_out) # [N, 4, H, W] -> (dx, dy, dw, dh)

        return cls_logits, bbox_pred


# --- Modèle Complet avec FPN ---
class DetectorWithFPN(nn.Module):
    def __init__(self, num_classes=2, backbone_layers=[2, 2, 2, 2], fpn_out_channels=256, head_num_convs=4):
        super().__init__()
        self.backbone = ResNetBackboneScratch(BasicBlock, backbone_layers)
        backbone_out_channels = self.backbone.get_out_channels() # {'C3': 128, 'C4': 256, 'C5': 512}

        self.fpn = FPN(backbone_out_channels, fpn_out_channels)
        fpn_feature_channels = self.fpn.out_channels # = fpn_out_channels

        # Num classes pour la tête: 1 (texte) + 1 (background implicite via BCE/Focal) = 1 si on utilise Focal/BCE
        # Si on utilise CrossEntropy, il faut N_CLASSES (texte=1, BG=0) => 2
        # Restons sur la logique BCE/Focal Loss => num_classes pour la tête = 1 (texte)
        head_num_classes = 1 # On prédit juste la probabilité d'être 'texte'

        self.detection_head = SimpleDetectionHead(fpn_feature_channels, head_num_classes, head_num_convs)

        # Strides correspondant aux niveaux FPN (P3, P4, P5)
        self.strides = {"P3": 8, "P4": 16, "P5": 32}

    def forward(self, images, targets=None):
        """
        Args:
            images (Tensor): Batch d'images [N, C, H, W].
            targets (list[dict], optional): Cibles pour l'entraînement.

        Returns:
            En mode entraînement (targets is not None):
                Dict[str, Tensor]: Dictionnaire de pertes (calculé DANS engine.py).
                                    Pour l'instant, retourne les preds pour calcul externe.
            En mode évaluation (targets is None):
                List[Dict[str, Tensor]]: Liste (taille N) de dicts contenant 'boxes', 'scores', 'labels'
                                          après post-processing (NMS, etc.) -> Logique à ajouter.
                                          Pour l'instant, retourne les preds brutes.

        Retourne (pour calcul de perte externe):
            Dict[str, List[Tensor]]:
                {'cls_logits': [P3_logits, P4_logits, P5_logits],
                 'bbox_pred': [P3_bbox, P4_bbox, P5_bbox]}
        """
        backbone_features = self.backbone(images) # {'C3': ..., 'C4': ..., 'C5': ...}
        fpn_features = self.fpn(backbone_features) # {'P3': ..., 'P4': ..., 'P5': ...}

        all_cls_logits = []
        all_bbox_preds = []

        # Appliquer la tête à chaque niveau FPN
        for level_name in ["P3", "P4", "P5"]: # Ordre important pour correspondre aux strides
            level_features = fpn_features[level_name]
            cls_logits, bbox_pred = self.detection_head(level_features)

            # Les sorties doivent être mises en forme: [N, C, H, W] -> [N, H*W, C] ou [N, H, W, C]
            # Permuter et aplatir la dimension spatiale est courant
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous().view(N, H * W, -1) # [N, H*W, NumClasses]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(N, H * W, 4)   # [N, H*W, 4]

            all_cls_logits.append(cls_logits)
            all_bbox_preds.append(bbox_pred)

        # --- Logique de Perte/Post-traitement ---
        # La perte sera calculée dans engine.py en utilisant ces sorties multi-niveaux.
        # Le post-traitement (NMS) pour l'évaluation devra aussi combiner les résultats
        # de tous les niveaux.

        output = {
            "cls_logits": all_cls_logits, # Liste de tensors [N, HxW, NumCls]
            "bbox_pred": all_bbox_preds   # Liste de tensors [N, HxW, 4]
        }

        return output

# --- Fonction pour créer le modèle ---
def build_detection_model(num_classes=2, backbone_layers=[2, 2, 2, 2], fpn_out_channels=256, head_num_convs=4):
    """Construit le modèle de détection FPN from scratch."""
    # Note: num_classes n'est pas directement utilisé si la tête prédit 1 classe (texte)
    # et que la perte gère le background implicitement.
    model = DetectorWithFPN(
        num_classes=1, # Tête prédit 1 classe (texte)
        backbone_layers=backbone_layers,
        fpn_out_channels=fpn_out_channels,
        head_num_convs=head_num_convs
    )
    print(f"Built detection model: {model.__class__.__name__} with FPN")
    # Ajouter un print pour le nombre de paramètres si utile
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Model parameters: {num_params / 1e6:.2f} M")
    return model

# Test rapide de construction
if __name__ == '__main__':
     print("Testing FPN model construction...")
     model_fpn = build_detection_model()
     print("DetectorWithFPN built.")

     # Test rapide du forward pass
     try:
         bs = 2
         # Utiliser une taille d'image divisible par la stride max (32)
         img_h, img_w = 640, 640 # config.IMG_SIZE devrait être divisible par 32
         dummy_images = torch.randn(bs, 3, img_h, img_w)
         model_fpn.eval()
         with torch.no_grad():
             outputs = model_fpn(dummy_images)

         print("\nForward pass test (eval mode):")
         print(f"Input shape: {dummy_images.shape}")
         print("Output keys:", outputs.keys())
         print(f"Num levels: {len(outputs['cls_logits'])}")

         strides = [8, 16, 32] # P3, P4, P5
         for i, (logits, boxes) in enumerate(zip(outputs['cls_logits'], outputs['bbox_pred'])):
              level_h, level_w = img_h // strides[i], img_w // strides[i]
              print(f"  Level P{i+3} (stride {strides[i]}):")
              print(f"    cls_logits shape: {logits.shape}") # Ex: [bs, level_h*level_w, 1]
              print(f"    bbox_pred shape: {boxes.shape}")   # Ex: [bs, level_h*level_w, 4]
              assert logits.shape[1] == level_h * level_w
              assert boxes.shape[1] == level_h * level_w

     except Exception as e:
         print(f"\nError during model forward pass test: {e}")
         import traceback
         traceback.print_exc()