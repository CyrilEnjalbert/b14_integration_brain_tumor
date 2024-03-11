import os
import cv2
import random
import shutil
import numpy as np
import math
import mlflow
import mlflow.keras
import pandas as pd
import tensorflow as tf

from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from fonctions.data_processing import *
from fonctions.data_loading import *



# Data loading ------

random.seed(422)
tf.random.set_seed(422)


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



# Chemin vers le répertoire racine
root_path = "./data/"
raw_path = os.path.join(root_path, 'raw')
proc_path = os.path.join(root_path, 'proc')

# Créer les répertoires train et test
train_path = os.path.join(proc_path, "train")
val_path = os.path.join(proc_path, "val")
test_path = os.path.join(proc_path, "test")

create_dir(train_path)
create_dir(val_path)
create_dir(test_path)

# Liste des classes (yes, no)
classes = ["yes", "no"]

file_mapping = []

counter = 0

# Pour chaque classe
for class_name in classes:
    class_path = os.path.join(raw_path, class_name)
    images = os.listdir(class_path)
    
    # Mélanger aléatoirement les images
    random.shuffle(images)
    
    # Calculer la séparation des données (60/20/20)
    val_split_index = int(0.6 * len(images))
    test_split_index = int(0.8 * len(images))
    
    # Diviser les données en ensembles d'entraînement et de test
    train_images = images[:val_split_index]
    val_images = images[val_split_index: test_split_index]
    test_images = images[test_split_index:]
    
    # Créer les répertoires de classe dans les ensembles d'entraînement et de test
    train_class_path = os.path.join(train_path, class_name)
    val_class_path = os.path.join(val_path, class_name)
    test_class_path = os.path.join(test_path, class_name)
    
    create_dir(train_class_path)
    create_dir(val_class_path)
    create_dir(test_class_path)

    for dataset_name, dataset_images, dataset_class_path in [('train', train_images, train_class_path), 
                                                             ('val', val_images, val_class_path),
                                                             ('test', test_images, test_class_path)]:
        for image in dataset_images:
            src = os.path.join(class_path, image)
            dst = os.path.join(dataset_class_path, format_filename(counter))
            shutil.copy(src, dst)
            file_mapping += [{'raw_img_path': src, 'proc_img_path': dst, 'class_name': class_name, 'dataset_name': dataset_name}]
            counter += 1


df = pd.DataFrame.from_records(file_mapping)
df.to_csv(os.path.join(root_path, 'file_mapping.csv'), index=False, header=True)

X_train, y_train = load_images(train_path)
X_val, y_val = load_images(val_path)
X_test, y_test = load_images(test_path)



# Data processing ------ 

# Utilisation de la fonction avec X (images non normalisées) et la taille cible
target_size = (224, 224)
X_train_norm = normalize_images(X_train, target_size)
X_val_norm = normalize_images(X_val, target_size)
X_test_norm = normalize_images(X_test, target_size)



# Training and testing model ------

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Créer un modèle VGG-16 pré-entraîné (ne pas inclure la couche dense finale)
base_model = VGG16(include_top=False, input_shape=(224, 224, 3))

NUM_CLASSES = 1

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='sigmoid'))

# Figer les poids du VGG
model.layers[0].trainable = False

# Compiler le modèle
model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=1e-4),
    metrics=['accuracy']
)

print("Model successfully loaded and compiled.")

# Afficher la structure du modèle
model.summary()

# Créer un générateur d'images pour la data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.01,
    height_shift_range=0.01,
    zoom_range=0.05,
    shear_range=0.01,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    vertical_flip=True
)

# Ajuster le générateur aux données d'entraînement
datagen.fit(X_train_norm)

# Créer un callback d'arrêt anticipé
early_stopping = EarlyStopping(
    monitor='val_loss',  # Surveiller la perte sur l'ensemble de validation
    patience=10,  # Arrêter l'entraînement si la perte ne diminue pas pendant 3 époques consécutives
    restore_best_weights=True,  # Restaurer les poids du modèle aux meilleurs atteints pendant l'entraînement
    verbose=1  # Afficher des messages lors de l'arrêt anticipé
)

BATCH_SIZE = 16

print("Model successfully loaded and created.")



# Saving model, params and metrics ------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Entraîner le modèle avec l'augmentation de données
with mlflow.start_run(run_name="b14_tumor_detection_model") as run:
    # Log the size of X_train_norm
    mlflow.log_metric("X_train_norm_size", len(X_train_norm))
    
    history = model.fit(datagen.flow(X_train_norm, y_train, batch_size=BATCH_SIZE),
                        epochs=10,
                        steps_per_epoch=len(X_train_norm) // BATCH_SIZE,
                        validation_data=(X_val_norm, y_val),
                        callbacks=[early_stopping])
    # Evaluate the model
    loss, acc = model.evaluate(X_test_norm, y_test)
    
    # Log the model
    mlflow.log_metric("test_accuracy",acc )
    mlflow.log_metric("test_loss", loss )
    mlflow.keras.log_model(model, "models", registered_model_name="b14_tumor_detection_model")


