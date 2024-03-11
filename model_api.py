import os
import base64
import mlflow
import mlflow.pyfunc
import numpy as np

from fastapi import FastAPI
from pymongo import MongoClient
from PIL import Image
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from mlflow import MlflowClient

from config.paths import mongo_path
from fonctions.data_loading import *
from fonctions.data_processing import *

app = FastAPI()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'



# MLFlow ------

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_name = "b14_tumor_detection_model"

# Get the latest version of the model
client = MlflowClient()
latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version

loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{latest_version}")

client = MlflowClient()



# Mongo DB ------

# Connexion à la base de données MongoDB
client = MongoClient(mongo_path)
db = client["braintumor"]

class PatientModel(BaseModel):
    name: str
    age: int
    gender: str
    image: bytes
    prediction: Optional[float] = None
    validation: Optional[str] = None



# Fonction pour prétraiter l'image
def preprocess_image(image):
    target_size = (224, 224)

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



# Fonctions ------

local_directory = "./data/proc/train"
train_dir = "./data/proc/train"

# Function to check MongoDB for non-valid documents and save images locally
def feedback_non_valid_patients(train_dir):
    # Create the local directory if it doesn't exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # Query MongoDB for documents with "validation" field containing "non-valide"
    non_valid_patients = db.patients.find({"validation": "non-valide"})
    print(f"Patients non-validés trouvés : {non_valid_patients}")

    # Iterate over non-valid patients
    for patient in non_valid_patients:
        # Check if the label is valid
        if patient["prediction"] > 0.5:
            label = "no"
        else:
            label = "yes"

        # Create class directory if it doesn't exist
        class_dir = os.path.join(train_dir, label)

        db.patients.update_one({"_id": patient["_id"]}, {"$set": {"validation": "non-valide(feedback)"}})


        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Download the image
        try:
            image_data = base64.b64decode(patient["image"])

            # Get the current count of images in the directory
            img_count = len(os.listdir(class_dir))

            # Increment the count and format it with leading zeros
            img_count += 1
            img_count_str = str(img_count).zfill(5)  # Format count with leading zeros

            # Save the image to the train directory with the appropriate label
            image_filename = f"img_{img_count_str}.jpeg"

            # Save the image to the train directory with the appropriate label
            image_path = os.path.join(class_dir, image_filename)
            with open(image_path, "wb") as f:
                f.write(image_data)
                print(f"Image saved: {image_filename} for patient {patient['name']} in {image_path}")
                print(f"Feedback for patient {patient.name}: {patient.prediction}, validation : {patient.validation}")
        except Exception as e:
            print(f"Error downloading image for patient {patient['name']}: {e}")



# Inputs ------

@app.post("/feedback")
async def run_feedback():
    # Fetch patients from MongoDB
    patients = fetch_patients()

    if not patients:
        print("No patients found in the database.")
        return {"message": "No patients found in the database."}
    else:
        # Execute prediction model for each patient
        for patient in patients:
            print(f"Processing feedback for patient: {patient.name}")
            feedback_non_valid_patients(train_dir)


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
        print(f"Prediction for patient {patient.name}: {patient.prediction} with {model_name} version {latest_version}")

    # Feedback for Upgrade the model
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