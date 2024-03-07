import os
import random
import mlflow
import mlflow.keras
import tensorflow as tf

from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from mlflow.tracking import MlflowClient

from fonctions.data_loading import *
from fonctions.data_processing import normalize_images



# Data loading ------

random.seed(422)
tf.random.set_seed(422)


# Chemin vers le répertoire racine
root_path = "./data/"
raw_path = os.path.join(root_path, 'raw')
proc_path = os.path.join(root_path, 'proc')

# Créer les répertoires train et test
train_path = os.path.join(proc_path, "train")
val_path = os.path.join(proc_path, "val")
test_path = os.path.join(proc_path, "test")

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

# Evaluate the model
model.evaluate(X_test_norm, y_test) 

try: 

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_name = "b14_tumor_detection_model"
    
    # Get the latest version of the model
    client = MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version

    # Fine-tune the model with updated data
    model.fit(X_train_norm, y_train, epochs=10) 
    
    # Log the size of X_train_norm    
    mlflow.log_metric("X_train_norm_size", len(X_train_norm))
    mlflow.log_metric("feedbacks_img_added", (len(X_train_norm) - 151))
    
    # Log the updated model with a new version
    mlflow.keras.log_model(model, "models", registered_model_name=model_name)

    # Update the model version in the Model Registry
    client.update_model_version(
        name=model_name,
        version=latest_version + 1,  # Increment version number
        description="C'est un modèle destiné à détecter des tumeurs cérébrales.",
    )

    print("Model successfully uploaded.")

except Exception as e:
    # You can handle the exception here as needed
    print(f"An error occurred: {str(e)}")