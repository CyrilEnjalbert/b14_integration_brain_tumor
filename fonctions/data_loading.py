import os
import shutil
import numpy as np
import cv2



def format_filename(suffix_number, padding=5, prefix="img_", extension="jpeg"):
    formatted_number = str(suffix_number).zfill(padding)
    filename = f"{prefix}{formatted_number}.{extension}"
    return filename


def create_dir(directory_path, remove_if_exists=True):
    # Supprimer le répertoire existant s'il existe
    if os.path.exists(directory_path) and remove_if_exists:
        shutil.rmtree(directory_path)

    # Créer le nouveau répertoire
    os.makedirs(directory_path, exist_ok=(not remove_if_exists))


def load_images(path_to_folder):
    # Liste des classes (dossiers "yes" et "no")
    classes = ["yes", "no"]
    classes_enc = {'yes': 1, 'no': 0}

    # Compter le nombre total d'images pour allouer les tableaux numpy
    total_images = sum(len(os.listdir(os.path.join(path_to_folder, class_name))) for class_name in classes)

    X = [None] * total_images  # Préallouer pour les images
    y = [None] * total_images  # Préallouer pour les labels


    last_index = 0
    
    # Parcourir chaque classe
    for class_name in classes:
        class_path = os.path.join(path_to_folder, class_name)

        # Parcourir chaque image dans la classe
        for idx, image_name in enumerate(os.listdir(class_path)):
            image_path = os.path.join(class_path, image_name)

            # Lire l'image avec OpenCV
            image = cv2.imread(image_path)

            # Ajouter l'image et le label aux tableaux X et y
            X[last_index + idx] = image
            y[last_index + idx] = classes_enc[class_name]

        last_index = last_index + idx + 1

    return np.array(X, dtype='object'), np.array(y)