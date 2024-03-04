import base64
from fastapi import FastAPI
from pymongo import MongoClient
import os
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel
from bson import ObjectId
from typing import Optional
import mlflow
import tensorflow as tf
from io import BytesIO

app = FastAPI()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

mlflow.set_tracking_uri("http://127.0.0.1:5000")

logged_model = 'runs:/2643540303a44773955ae0cd8d441403/models'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Connexion à la base de données MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["braintumor"]

# Pydantic Model for Patient Data
class PatientModel(BaseModel):
    name: str
    age: int
    gender: str
    image: bytes
    prediction: Optional[float]

# Fonction pour normaliser les images
def normalize_images(X, target_size):
    normalized_images = [None] * len(X)

    for i, img in enumerate(X):
        if len(img.shape) == 3:
            # Convertir en niveaux de gris si ce n'est pas déjà le cas
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img

        # Appliquer un filtre pour supprimer le bruit (par exemple, un filtre gaussien)
        denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # Détecter les contours pour trouver le crop optimal
        _, thresh = cv2.threshold(denoised_img, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Trouver le contour avec la plus grande aire
            max_contour = max(contours, key=cv2.contourArea)

            # Obtenir les coordonnées du rectangle englobant
            x, y, w, h = cv2.boundingRect(max_contour)

            # Cropper l'image pour obtenir la région d'intérêt
            cropped_img = img[y:y+h, x:x+w]

            # Redimensionner à target_size (pour s'assurer que toutes les images ont la même taille)
            normalized_images[i] = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)
        else:
            # Redimensionner à target_size si aucun contour n'est détecté
            normalized_images[i] = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return np.array(normalized_images)

# Fonction pour prétraiter l'image
def preprocess_image(image):
    target_size = (224, 224)

    # Decode base64 string to bytes
    #image_bytes = base64.b64decode(image)
    
    # Open bytes as image using PIL
    img = Image.open(BytesIO(image))
    
    # Convertir l'image en un tableau numpy
    img_array = np.array(img)
    
    # Normaliser et redimensionner l'image
    img_array = normalize_images([img_array], target_size)
    
    return img_array

# Function to fetch data from MongoDB
def fetch_patients():
    patients = []
    for document in db.patients.find({}):
        document["image"] = base64.b64decode(document["image"])
        patient_data = PatientModel(**document)
        patients.append(patient_data)
    return patients

# Function to execute prediction model
def predict(patient):
    # Preprocess the uploaded image
    processed_image = preprocess_image(patient.image)
    
    # Make predictions using the loaded model
    predictions = loaded_model.predict(processed_image)

    # Assuming predictions is a single float value
    patient.prediction = float(predictions[0])
    
    return patient

# Function to update MongoDB collection with predictions
def update_collection(patients):
    for patient in patients:
        db.patients.update_one({"name": patient.name}, {"$set": {"prediction": patient.prediction}}, upsert=True)

# Define FastAPI endpoint
@app.post("/predict")
async def run_prediction():
    # Fetch patients from MongoDB
    patients = fetch_patients()

    if not patients:
        print("No patients found in the database.")
        return {"message": "No patients found in the database."}

    # Execute prediction model for each patient
    for patient in patients:
        print(f"Processing prediction for patient: {patient.name}")
        patient = predict(patient)
        print(f"Prediction for patient {patient.name}: {patient.prediction}")

    # Update MongoDB collection with predictions
    update_collection(patients)

    print("Predictions updated successfully")
    return {"message": "Predictions updated successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
