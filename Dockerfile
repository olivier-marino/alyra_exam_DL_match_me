# Image de base
FROM python:3.12-slim

RUN pip install --upgrade pip

# création d'un dossier /app dans le conteneur
RUN mkdir /app
WORKDIR /app

# création de l'environement virtuel python
RUN --mount=source=requirements.txt,destination=requirements.txt pip install -r requirements.txt; exit 0;

# Copy des fichiers du dossier courant de l'hote dans le dossier courant du conteneur
COPY lstm_toxic_classifier_from_scratch.keras app.py .

# Ouvre le port 8501 du docker pour accéder au serveur streamlit
EXPOSE 8000

# Commande a lancer lors du run de l'image avec les options associé
ENTRYPOINT [ "uvicorn", "--host", "0.0.0.0", "app:app" ]
