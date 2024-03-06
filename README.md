# Brief 14 Intégration d'un Modèle d'IA pour la Détection de Tumeurs dans une Application de Gestion Médicale :
## Intégration d'un Modèle d'IA pour la Détection de Tumeurs dans une Application de Gestion Médicale

Groupe : Anatole, Cyril, Jimmy et Yassine

## Contexte du projet :
En tant que développeur en intelligence artificielle, la mission consiste à intégrer le modèle d'intelligence artificielle de Neurogard dans une application existante de gestion médicale (FastApi, MongoDB). Les principales tâches incluent le développement d'une API pour permettre à l'application de faire des prédictions basées sur des radiographies médicales, la mise en œuvre de fonctionnalités avancées telles que le téléchargement d'images, la validation par des experts et la gestion des divergences entre les diagnostics humains et les prédictions de l'IA.

## Procédure : 
-> Rentrer votre "connection string" fourni par Mongo DB dans le champ prévu à cet effet dans mongo_string.py

-> Si besoin créer manuellement une Database "braintumor" et une collection "patients" depuis votre interface Mongo DB

-> Lancer dans un Shell dans le dossier b14_integration_brain_tumor : $> mlflow ui

-> Pour la création du modèle de prédiction utilisé dans le main.py pour afficher les prédictions,
   lancer train_and_log_model.py ou train_and_log_model.py

-> Lancer app.py et model_api/main.py

-> Pour réaliser une nouvelle version du modèle prenant compte des feedback des prédictions "non-valide",
   lancer update_model.py (main.py chargera automatiquement la dernière version du modèle)


## Repartition des tâches :
Anatole a travaillé sur le front orienté API : co-travaillé avec Jimmy sur l'API app.py, la séparation des patients selon le statut de validation de leur prédiction, l'ajout en front des fonctionnalités : ajout d'image dans le add_patient, le front du view_patients.


Jimmy a travaillé sur le front orienté API : co-travaillé avec Anatole sur l'API app.py, la création du pdf_tumor pour chaque utilisateur, la création de la barre de recherche utilisateurs, le debug pour l'affichage de l'image base64, la 
création de la page Html details_patients


Cyril a travaillé sur le back-end orienté modèle de prédiction : la prédiction du modèle, les fichiers de création et actualisation du modèle MLflow et les retours "Feedback" pour améliorer l'entrainement du modèle.


