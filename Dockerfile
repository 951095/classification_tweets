FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Télécharger les ressources NLTK nécessaires
RUN python -m nltk.downloader punkt punkt_tab stopwords


COPY . .

CMD ["python", "main.py"]
