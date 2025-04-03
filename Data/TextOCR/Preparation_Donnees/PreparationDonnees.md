# Préparation des Données pour TextOCR (Étape 1)

Ce document explique les étapes de préparation des données pour la préparation des données pour l'étape 1, incluant la correction des rotations d'images et la séparation des ensembles d'entraînement et de validation.

## Structure du Dossier TextOCR

Le dossier est organisé comme suit:

- **TextOCR/**
    - **Preparation_Donnees/** - Scripts de préparation des données
        - `FiltrerCSV_Rotation.py` - Filtre les CSV pour ne garder que les images avec rotation
        - `Rotation_Images.py` - Corrige la rotation des images
        - `CopierImagepourValidation.py` - Copie les images de validation
        - `SupprimerValidationDansTrain.py` - Supprime les images de validation du dossier d'entraînement
    - **Test/**
        - **Test_Images/** - Images de test originales
        - **Test_Images_Correct/** - Images de test avec rotation corrigée
        - `test-images-with-rotation.csv` - Informations sur les rotations
        - `test-images-with-rotation-filtered.csv` - Informations filtrées sur les rotations
        - `TextOCR_AnnotationsTest.json` - Annotations des images de test
    - **Train/**
        - **Train_Images/** - Images d'entraînement
        - **Train_Images_Correct/** - Images d'entraînement avec rotation corrigée
        - `train-images-boxable-with-rotation.csv` - Informations sur les rotations
        - `train-images-boxable-with-rotation-filtered.csv` - Informations filtrées sur les rotations
        - `TextOCR_AnnotationsTraining.json` - Annotations des images d'entraînement
    - **Validation/**
        - **Validation_Images/** - Images de validation
        - **Validation_Images_Correct/** - Images de validation avec rotation corrigée
        - `TextOCR_AnnotationsValidations.json` - Annotations des images de validation


## Scripts de Préparation des Données

### 1. FiltrerCSV_Rotation.py

Ce script filtre les fichiers CSV pour ne conserver que les lignes où la rotation n'est pas égale à 0.0 (images nécessitant une rotation).


### 2. Rotation_Images.py

Ce script corrige la rotation des images selon les valeurs indiquées dans les CSV filtrés.


### 3. CopierImagepourValidation.py

Ce script extrait les images de validation du dossier d'entraînement et les copie dans le dossier de validation.

### 4. SupprimerValidationDansTrain.py

Ce script supprime les images de validation du dossier d'entraînement après les avoir copiées dans le dossier de validation.


## Nettoyage Final

Une fois toutes les étapes de préparation terminées, vous pouvez supprimer les dossiers suivants car ils sont vides ou ne sont plus nécessaires:

- Test_Images_Correct
- Train_Images_Correct
- Validation_Images_Correct

Ces dossiers ont servi à stocker temporairement les images corrigées, mais après avoir remplacé les images originales par les versions corrigées, ils ne sont plus utiles.
