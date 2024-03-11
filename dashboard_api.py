import uvicorn
import requests
import base64

from pymongo import MongoClient
from bson import ObjectId
from pydantic import BaseModel
from typing import Optional
from weasyprint import HTML
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, Response

from config.paths import mongo_path


app = FastAPI()

# Connexion à la base de données MongoDB
client = MongoClient(mongo_path)
db = client["braintumor"]  # Remplacez "your_database_name" par le nom de votre base de données MongoDB
col_add = db["patients"]
col_role = db["logins_users"]

class PatientLoginModel(BaseModel):
    name: str
    password: str

# Modèle Pydantic pour les données du patient
class PatientModel(BaseModel):
    name: str
    age: int
    gender: str
    image: str
    # prediction: float
    validation: str = 'En attente de validation'
    
# Modèles Pydantic pour la modification du patient
class PatientUpdateModel(BaseModel):
    name: str
    age: int
    gender: str
    image: str
    # prediction: float
    # validation: str

# Modèles Pydantic pour la visualisation des patients
class PatientViewModel(BaseModel):
    name: str
    age: int
    gender: str
    id: str
    prediction: float
    validation: str

# Modèles Pydantic pour la details view des patients
class PatientDetailsModel(BaseModel):
    name: str
    age: int
    gender: str
    id: str
    prediction: float
    encoded_image: str
    
    
# Modèle Pydantic pour les prédictions (à adapter selon vos besoins)
class PredictionModel(BaseModel):
    # Ajoutez les champs nécessaires pour les prédictions
    pass


# Montez le répertoire 'static' pour servir les fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")


# Instance du moteur de modèles Jinja2 pour la gestion des templates HTML
templates = Jinja2Templates(directory="templates")

# Route pour afficher la page de connexion
@app.get("/login", response_class=HTMLResponse)
def get_login(request: Request):
    # Renvoyer le contenu HTML de la page de connexion
    return templates.TemplateResponse("login.html", {"request": request})

# Route pour gérer la soumission du formulaire de connexion
@app.post("/login")
async def post_login(name: str = Form(...), password: str = Form(...)):
    # Vérifier si l'utilisateur existe dans la base de données
    user = col_role.find_one({"name": name, "password": password})
    if user:
        role = user.get("role", "")
        if role == "Docteur":
            # Rediriger le docteur vers la page view_patients.html
            return JSONResponse(content={"redirect_url": "/view_patients"})
        elif role == "Assistante":
            # Rediriger l'assistante vers la page view_assistant_patients.html
            
            return JSONResponse(content={"redirect_url": "/view_assistant_patients"})
        else:
            raise HTTPException(status_code=403, detail="Invalid role")
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")




# Route pour ajouter un patient
@app.get("/add_patient", response_class=HTMLResponse)
def add_patient(request: Request):
    return templates.TemplateResponse("add_patient.html", {"request": request})

@app.post("/add_patient")
async def add_patient_post(patient: PatientModel):
    # Insérer le patient dans la base de données
    patient_data = patient.model_dump()

    db.patients.insert_one(patient_data)
    # URL of the FastAPI endpoint

    # Send a POST request to the endpoint
    requests.post( f"http://localhost:8000/predict")
    return JSONResponse(content={"redirect_url": "/view_assistant_patients"})

@app.post("/view_patients")
async def view_patients(request: Request):
    to_validate_patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find({"validation": "En attente de validation"})]
    corrected_patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find({"validation": "Corrected"})]
    validated_patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find({"validation": "Validated"})]

    for patient in to_validate_patients:
        prediction_percentage = "{:.2f}%".format(patient.prediction * 100)  # Supposant que 'prediction' est déjà un pourcentage
        patient.prediction = prediction_percentage

    for patient in corrected_patients:
        prediction_percentage = "{:.2f}%".format(patient.prediction * 100)  # Supposant que 'prediction' est déjà un pourcentage
        patient.prediction = prediction_percentage

    for patient in validated_patients:
        prediction_percentage = "{:.2f}%".format(patient.prediction * 100)  # Supposant que 'prediction' est déjà un pourcentage
        patient.prediction = prediction_percentage

    return templates.TemplateResponse("view_patients.html", {"request": request,
                                                             "to_validate_patients": to_validate_patients,
                                                             "corrected_patients": corrected_patients,
                                                             "validated_patients": validated_patients})

@app.get("/view_patients", response_class=HTMLResponse)
async def view_patients(request: Request):
    to_validate_patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find({"validation": "En attente de validation"})]
    corrected_patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find({"validation": "Corrected"})]
    validated_patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find({"validation": "Validated"})]

    for patient in to_validate_patients:
        prediction_percentage = "{:.2f}%".format(patient.prediction * 100)  # Supposant que 'prediction' est déjà un pourcentage
        patient.prediction = prediction_percentage

    for patient in corrected_patients:
        prediction_percentage = "{:.2f}%".format(patient.prediction * 100)  # Supposant que 'prediction' est déjà un pourcentage
        patient.prediction = prediction_percentage

    for patient in validated_patients:
        prediction_percentage = "{:.2f}%".format(patient.prediction * 100)  # Supposant que 'prediction' est déjà un pourcentage
        patient.prediction = prediction_percentage

    return templates.TemplateResponse("view_patients.html", {"request": request,
                                                             "to_validate_patients": to_validate_patients,
                                                             "corrected_patients": corrected_patients,
                                                             "validated_patients": validated_patients})

@app.get("/view_patients2", response_class=HTMLResponse)
async def view_patients(request: Request):
    # Récupérer tous les patients depuis la base de données
    patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find()]

    # Itérer sur chaque patient et formater la prédiction en pourcentage
    for patient in patients:
        prediction_percentage = "{:.2f}%".format(patient.prediction * 100)  # Supposant que 'prediction' est déjà un pourcentage
        patient.prediction = prediction_percentage

    return templates.TemplateResponse("view_patients.html", {"request": request, "patients": patients})

@app.get("/view_assistant_patients", response_class=HTMLResponse)
async def view_patients(request: Request):
    to_validate_patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find({"validation": "En attente de validation"})]
    corrected_patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find({"validation": "Corrected"})]
    validated_patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find({"validation": "Validated"})]

    for patient in to_validate_patients:
        prediction_percentage = "{:.2f}%".format(patient.prediction * 100)  # Supposant que 'prediction' est déjà un pourcentage
        patient.prediction = prediction_percentage

    for patient in corrected_patients:
        prediction_percentage = "{:.2f}%".format(patient.prediction * 100)  # Supposant que 'prediction' est déjà un pourcentage
        patient.prediction = prediction_percentage

    for patient in validated_patients:
        prediction_percentage = "{:.2f}%".format(patient.prediction * 100)  # Supposant que 'prediction' est déjà un pourcentage
        patient.prediction = prediction_percentage

    return templates.TemplateResponse("view_assistant_patients.html", {"request": request,
                                                             "to_validate_patients": to_validate_patients,
                                                             "corrected_patients": corrected_patients,
                                                             "validated_patients": validated_patients})


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

from typing import Optional

@app.get("/search_patients", response_class=HTMLResponse)
async def search_patients(request: Request, search: Optional[str] = None):
    if search:
        patients_from_db = db.patients.find({"name": {"$regex": search, "$options": "i"}})
    else:
        return RedirectResponse(url="/view_patients")   

    patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in patients_from_db]

    # Itérer sur chaque patient pour calculer prediction_percentage
    for patient in patients:
        prediction_percentage = "{:.2f}%".format(patient.prediction * 100)
        patient.prediction = prediction_percentage

    return templates.TemplateResponse("view_patients.html", {"request": request, "patients": patients})




@app.get("/view_image/{patient_id}", response_class=HTMLResponse)
async def view_image(request: Request, patient_id: str):
    # Récupérer les informations du patient pour affichage dans la page view_image.html
    patient = PatientModel(**db.patients.find_one({"_id": ObjectId(patient_id)}))
    return templates.TemplateResponse("view_image.html", {"request": request, "patient": patient})


from fastapi import HTTPException
from starlette.responses import FileResponse
import tempfile
import os

# Utilisez un répertoire temporaire pour stocker temporairement les PDF
temp_dir = tempfile.TemporaryDirectory()

@app.get("/preview_pdf_predict/{patient_id}")
async def preview_pdf_predict(patient_id: str):
    try:
        # Fetch patient data
        patient = db.patients.find_one({"_id": ObjectId(patient_id)})
        if patient:
            # Génération du contenu HTML pour le PDF
            html_content = generate_pdf_content(patient)

            # Génération du PDF à partir du contenu HTML
            pdf_bytes = HTML(string=html_content).write_pdf()

            # Stocker le PDF temporairement
            pdf_path = f"{temp_dir.name}/{patient_id}.pdf"
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(pdf_bytes)

            # Retourner une réponse de redirection vers le PDF temporaire
            return RedirectResponse(url=f"/pdf_preview/{patient_id}")
        else:
            raise HTTPException(status_code=404, detail="Patient not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while generating the PDF.")

@app.get("/pdf_preview/{patient_id}")
async def pdf_preview(patient_id: str):
    # Récupérer le chemin du PDF temporaire
    pdf_path = f"{temp_dir.name}/{patient_id}.pdf"
    if os.path.exists(pdf_path):
        # Retourner le PDF pour prévisualisation
        return FileResponse(pdf_path, media_type='application/pdf')
    else:
        raise HTTPException(status_code=404, detail="PDF not found.")

@app.get("/download_pdf_predict/{patient_id}")
async def download_pdf_predict(patient_id: str):
    try:
        # Fetch patient data
        patient = db.patients.find_one({"_id": ObjectId(patient_id)})
        if patient:
            # Génération du contenu HTML pour le PDF
            html_content = generate_pdf_content(patient)

            # Génération du PDF à partir du contenu HTML
            pdf_bytes = HTML(string=html_content).write_pdf()

            # Retourner le PDF en tant que réponse HTTP pour téléchargement
            response = Response(content=pdf_bytes, media_type='application/pdf')
            response.headers['Content-Disposition'] = f'attachment; filename="{patient["name"]}_predictions_tumor.pdf"'
            return response
        else:
            return JSONResponse(content={"message": "Patient not found."}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"message": "An error occurred while generating the PDF."}, status_code=500)

def generate_pdf_content(patient):
    # Récupérer l'image depuis la base de données
    image_bytes = patient['image']

    # Décodez l'image base64
    image_html = f"<img src='data:image/jpeg;base64,{image_bytes}' />"

    # Style CSS pour le PDF
    css_style = """
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #dddddd;
            text-align: left;
            padding: 8px;
        }
        th {
            background-color: #f2f2f2;
        }
        img {
            max-width: 1000px;
            max-height: 1000px;
        }
    </style>
    """

    # Calcul de la prédiction en pourcentage
    prediction_percentage = "{:.2f}%".format(patient['prediction'] * 100)

    html_content = f"""
    <html>
    <head><title>Predictions - {patient['name']}</title>{css_style}</head>
    <body>
        <h1>Predictions - {patient['name']}</h1>
        
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Age</th>
                    <th>Gender</th>
                    <th>Prediction</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>{patient['_id']}</td>
                    <td>{patient['age']}</td>
                    <td>{patient['gender']}</td>
                    <td>{prediction_percentage}</td>                            
                </tr>
            </tbody>
        </table>
        {image_html}
    </body>
    </html>
    """
    return html_content




# Dans votre route FastAPI pour afficher les détails du patient
@app.get("/details_patients/{patient_id}", response_class=HTMLResponse)
async def details_patients(request: Request, patient_id: str):
    patient = db.patients.find_one({"_id": ObjectId(patient_id)})
    if patient:
        # Récupérer l'image depuis la base de données
        image_data = patient['image']
        if isinstance(image_data, str):
            # Si les données sont déjà encodées en base64, pas besoin de les ré-encoder
            encoded_image = image_data
        else:
            # Encodez les données en base64
            encoded_image = base64.b64encode(image_data).decode('utf-8')
        # Convertir la prédiction en pourcentage et formater avec deux chiffres après la virgule
        prediction_percentage = "{:.2f}".format(patient.get('prediction', 0.0) * 100)
        # Créez l'instance de PatientDetailsModel en spécifiant les champs nécessaires
        patient_view_model = PatientDetailsModel(
            id=str(patient['_id']),
            name=patient['name'],
            age=patient['age'],
            gender=patient['gender'],
            prediction=prediction_percentage,  # Convertir la prédiction en pourcentage
            encoded_image=encoded_image
        )
        return templates.TemplateResponse("details_patients.html", {"request": request, "patient": patient_view_model})
    else:
        return Response(content="Patient not found.", status_code=404)


# Modèle Pydantic pour la modification du champ de validation
class ValidationUpdateModel(BaseModel):
    validation: str


@app.post("/validate_patient/{patient_id}", response_class=HTMLResponse)
async def validate_patient_post(patient_id: str, action: str = Form(...)):
    patient = db.patients.find_one({"_id": ObjectId(patient_id)})

    if action == 'validated':
        # Update validation status to "validated"
        db.patients.update_one({"_id": ObjectId(patient_id)}, {"$set": {"validation": "Validated"}})
        print("Patient validated successfully")
    elif action == 'corrected':
        # Update validation status to "corrected"
        db.patients.update_one({"_id": ObjectId(patient_id)}, {"$set": {"validation": "Corrected"}})
        print("Patient corrected successfully")
        # Here you can add additional functionality if needed, like sending a POST request to another endpoint.
    
    # Redirect back to the view_patients page
    return RedirectResponse(url="/view_patients", status_code=303)


if __name__ == '__main__':
    import uvicorn    
    uvicorn.run(app, host='0.0.0.0', port=8010)