import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, Response
from pymongo import MongoClient
from bson import ObjectId
from pydantic import BaseModel
from typing import Optional
import requests
from weasyprint import HTML


app = FastAPI()

# Connexion à la base de données MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["braintumor"]  # Remplacez "your_database_name" par le nom de votre base de données MongoDB


# Modèle Pydantic pour les données du patient
class PatientModel(BaseModel):
    name: str
    age: int
    gender: str
    image: bytes
    prediction: Optional[float]
    
# Modèles Pydantic pour la modification du patient
class PatientUpdateModel(BaseModel):
    name: str
    age: int
    gender: str
    image: bytes
    prediction: Optional[str]

# Modèles Pydantic pour la visualisation des patients
class PatientViewModel(BaseModel):
    name: str
    age: int
    gender: str
    id: str
    prediction: Optional[float]

    
# Modèle Pydantic pour les prédictions (à adapter selon vos besoins)
class PredictionModel(BaseModel):
    # Ajoutez les champs nécessaires pour les prédictions
    pass


# Montez le répertoire 'static' pour servir les fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")


# Instance du moteur de modèles Jinja2 pour la gestion des templates HTML
templates = Jinja2Templates(directory="templates")


@app.get("/search_patients", response_class=HTMLResponse)
async def search_patients(request: Request, search: Optional[str] = None):
    if search:
        patients_from_db = db.patients.find({"name": {"$regex": search, "$options": "i"}})
    else:
        return RedirectResponse(url="/view_patients")   
    patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in patients_from_db]
    return templates.TemplateResponse("view_patients.html", {"request": request, "patients": patients})

@app.get("/view_image/{patient_id}", response_class=HTMLResponse)
async def view_image(request: Request, patient_id: str):
    # Récupérer les informations du patient pour affichage dans la page view_image.html
    patient = PatientModel(**db.patients.find_one({"_id": ObjectId(patient_id)}))
    return templates.TemplateResponse("view_image.html", {"request": request, "patient": patient})

# Route pour ajouter un patient
@app.get("/add_patient", response_class=HTMLResponse)
def add_patient(request: Request):
    return templates.TemplateResponse("add_patient.html", {"request": request})


@app.post("/add_patient")
async def add_patient_post(patient: PatientModel):
    # Insérer le patient dans la base de données
    patient_data = patient.dict()

    print(patient)

    db.patients.insert_one(patient_data)
    # URL of the FastAPI endpoint

    # Send a POST request to the endpoint
    requests.post("http://localhost:8000/predict")
    return JSONResponse(content={"redirect_url": "/view_patients"})


# Route pour visualiser tous les patients
@app.get("/view_patients", response_class=HTMLResponse)
async def view_patients(request: Request):
    # Récupérer tous les patients depuis la base de données
    patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find()]
    return templates.TemplateResponse("view_patients.html", {"request": request, "patients": patients})


# Route pour éditer un patient
@app.get("/edit_patient/{patient_id}", response_class=HTMLResponse)
async def edit_patient(request: Request, patient_id: str):
    # Récupérer les informations du patient pour affichage dans le formulaire
    patient = PatientModel(**db.patients.find_one({"_id": ObjectId(patient_id)}))
    return templates.TemplateResponse("edit_patient.html", {"request": request, "patient": patient,
                                                            "patient_id": patient_id})


@app.post("/edit_patient/{patient_id}")
async def edit_patient_post(patient_id: str, patient: PatientUpdateModel):
    # Mettre à jour le patient dans la base de données
    db.patients.update_one({"_id": ObjectId(patient_id)}, {"$set": patient.model_dump()})
    return RedirectResponse(url="/view_patients")

# --------------------------------------------


@app.get("/download_pdf_predict")
async def download_pdf_predict():
    # Créer le HTML avec les données des patients
    html_content = """
    <html>
    <head><title>Predictions PDF</title></head>
    <body>
        <h1>Predictions PDF</h1>
        <table border="1">
            <thead>
                <tr>
                    <th></th>
                    <th>Age</th>
                    <th>Gender</th>
                    <th>Prediction</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>John Doe</td>
                    <td>30</td>
                    <td>Male</td>
                    <td>0.75</td>
                </tr>
                <tr>
                    <td>Jane Doe</td>
                    <td>25</td>
                    <td>Female</td>
                    <td>0.80</td>
                </tr>
            </tbody>
        </table>
    </body>
    </html>
    """

    # Générer le PDF à partir du HTML
    pdf_bytes = HTML(string=html_content).write_pdf()

    # Renvoyer le PDF comme une réponse HTTP
    response = Response(content=pdf_bytes, media_type='application/pdf')
    response.headers['Content-Disposition'] = 'attachment; filename="predictions_tumor.pdf"'
    return response

if __name__ == '__main__':
    import uvicorn    
    uvicorn.run(app, host='0.0.0.0', port=8010)