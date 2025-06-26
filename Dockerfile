# Choix de l'image officielle Python
FROM python:3.12-slim

# Définir le répertoire de travail dans le container
WORKDIR /app

# Copier uniquement le fichier des dépendances d'abord (pour optimiser le cache Docker)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le contenu du projet dans le container
COPY . .

# Commande par défaut pour lancer les tests unitaires avec pytest
CMD ["pytest", "--maxfail=1", "--disable-warnings", "-q"]
