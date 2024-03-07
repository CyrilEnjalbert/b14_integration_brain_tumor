# SIMPLON DEV IA | Brief 14
## Intégration d'un Modèle d'IA pour la Détection de Tumeurs dans une Application de Gestion Médicale

Groupe : Anatole, Cyril, Jimmy et Yassine

### Contexte du projet
En tant que développeur en intelligence artificielle, la mission consiste à intégrer le modèle d'intelligence artificielle de Neurogard dans une application existante de gestion médicale (FastApi, MongoDB). Les principales tâches incluent le développement d'une API pour permettre à l'application de faire des prédictions basées sur des radiographies médicales, la mise en œuvre de fonctionnalités avancées telles que le téléchargement d'images, la validation par des experts et la gestion des divergences entre les diagnostics humains et les prédictions de l'IA.


### Structure du projet
```bash
project/
│
├── config/
│   └── paths.py    # Paths placeholder for MongoDB, etc.
│
├── data/
│   ├── raw/    # Place your raw data in this folder
│   │   ├── no/
│   │   └── yes/
│   └── proc/    # Folder where splitted data will be stored
│       ├── test/
│       ├── train/
│       └── val/
│
├── fonctions/
│   ├── data_loading.py    # Functions to load, rename data and create 'proc' folder
│   └── data_processing.py    # Function to normalize images
│
├── static/
│   └── styles.css    # CSS styling for dashboard
│
├── .gitignore
├── dashboard_api.py    # API script for dashboard
├── model_api.py    # Model API to give predict response
├── README.md
├── train_and_log_model.py    # Script to create our first model and log it
└── update_model.py    # Script that updates our model with doctor's feedbacks
```

### Procédure 
-> Rentrer votre "connection string" fournie par Mongo DB dans le champ prévu à cet effet dans config/paths.py

-> Si besoin créer manuellement une Database "braintumor" et une collection "patients" depuis votre interface Mongo DB

-> Lancer dans un Shell dans le dossier b14_integration_brain_tumor :
```bash
mlflow ui
```

-> Pour la création du modèle de prédiction utilisé dans le model_api.py, utiliser la commande :
```bash
python train_and_log_model.py
```

->Pour accéder à l'interface, lancer les deux API :
```bash
python model_api
```
```bash
python dashboard_api
```

-> Pour réaliser une nouvelle version du modèle prenant compte des feedback des prédictions "non-valide",
   lancer update_model.py (model_api.py chargera automatiquement la dernière version du modèle).
```bash
python update_model
```


### Repartition des tâches
Anatole a travaillé sur le front orienté API : co-travaillé avec Jimmy sur l'interface, la séparation des patients selon le statut de validation de leur prédiction, l'ajout en front des fonctionnalités : ajout d'image dans le add_patient, le front du view_patients. <br>


Jimmy a travaillé sur le front orienté API : co-travaillé avec Anatole sur l'API app.py, la création du pdf_tumor pour chaque utilisateur, la création de la barre de recherche utilisateurs, le debug pour l'affichage de l'image base64, la 
création du template HTML details_patients. <br>


Cyril a travaillé sur le back-end orienté modèle de prédiction : la prédiction du modèle, les fichiers de création et actualisation du modèle MLflow et les retours "Feedback" pour améliorer l'entrainement du modèle. <br>