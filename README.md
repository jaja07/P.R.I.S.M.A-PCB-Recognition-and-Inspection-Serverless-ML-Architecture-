# P.R.I.S.M.A-PCB-Recognition-and-Inspection-Serverless-ML-Architecture-
Automating electronic quality control with an event-driven serverless machine learning architec

### Explication pas-à-pas du flux de données

1. **Ingestion (Edge to Cloud) :** Les caméras en bout de chaîne de montage déposent les photos des PCB sur un compte de stockage Azure Blob Storage (conteneur `input`). Le nommage suit une convention stricte : `[IDTemplate]_[NomDuDefaut].jpg` (ex : `01_missing_hole_01.jpg`).
2. **Déclenchement événementiel (Dispatcher) :** Une Azure Function de type *Blob Trigger* (`blob_dispatcher`) surveille en continu le conteneur `input`. Dès qu'une image est enregistrée, elle extrait le nom du fichier et pousse un message léger au format JSON dans une file d'attente sécurisée (`process-queue`).
3. **Traitement asynchrone (Worker) :** Une seconde Azure Function de type *Queue Trigger* (`queue_worker`) dépile les messages dès qu'elle dispose de ressources. Elle télécharge l'image physique depuis le Blob Storage, extrait l'identifiant du template de référence (le préfixe avant le caractère `_`), et transmet le tout à l'API de Deep Learning.
4. **Inférence (Inference Engine) :** L'API FastAPI, conteneurisée sous Docker et déployée sur **Azure Container Apps**, reçoit l'image et l'ID du template. Elle charge dynamiquement le modèle optimisé au format **ONNX**, compare la carte inspectée avec l'image "golden" (référence saine), et renvoie un diagnostic structurel enrichi.
5. **Persistance et Archivage :** * Le Worker récupère la réponse et écrit le rapport complet d'inférence (JSON) dans le conteneur `output`.
* Parallèlement, il insère les métadonnées de l'inférence (ID unique, statut du défaut, latence, score de confiance, timestamp) dans une base de données **Azure Cosmos DB** (NoSQL).


6. **Supervision (Dashboard Front-End) :** Le tableau de bord web statique (hébergé sur *Azure Static Web Apps*) effectue des appels réguliers toutes les 5 secondes (technique du *Near Real-Time HTTP Polling*) vers une API dédiée portée par une troisième Azure Function de type *HTTP Trigger* (`get_inferences`). Cette fonction lit les 10 enregistrements les plus récents de Cosmos DB pour rafraîchir l'écran de l'opérateur sans rechargement de page.

---

## 3. Structure du Projet

```text
PRISMA/
├── .venv/                      # Environnement virtuel local Python
├── api/                        # Code source de l'API d'inférence (FastAPI)
│   ├── endpoints.py            # Définition des routes de prédiction
│   └── main.py                 # Point d'entrée de l'API FastAPI
├── docs/                       # Documentations techniques et schémas d'architecture
│   └── architecture_flow.png   # Diagramme d'architecture exporté
├── model/                      # Fichiers de modèles de Deep Learning
│   ├── data/
│   │   └── PCB_DATASET/        # Contient les images de référence (Golden Templates)
│   └── pcb_classifier_v1.0.0.onnx # Fichier de poids du modèle ONNX
├── prisma-functions/           # Projet Azure Functions v2 (Serverless Python)
│   ├── .vscode/                # Configurations de débogage pour VS Code
│   ├── function_app.py         # Code source unifié (Dispatcher, Worker, API Front-End)
│   ├── host.json               # Configuration globale de l'hôte Azure Functions
│   ├── local.settings.json     # Variables d'environnement et secrets locaux (ignoré par Git)
│   └── requirements.txt        # Dépendances Python spécifiques aux fonctions
├── web/                        # Code de l'interface utilisateur
│   └── index.html              # Dashboard d'administration minimaliste (HTML/Vanilla JS)
├── Dockerfile                  # Recette de conteneurisation pour l'API d'inférence
├── pyproject.toml              # Fichier de configuration du gestionnaire UV
├── requirements.txt            # Dépendances globales du projet
└── uv.lock                     # Fichier de verrouillage des versions par UV

```

---

## 4. Prérequis d'Environnement

Avant de procéder à l'installation, assurez-vous de disposer des outils suivants installés sur votre machine :

* **Python 3.13+** (recommandé pour une gestion optimale des dépendances asynchrones)
* **uv** : Le gestionnaire de packages Python ultra-rapide moderne (remplaçant avantageux de `pip` et `poetry`)
* **Docker Desktop** : Requis pour la construction de l'image de l'API et sa validation
* **Azure CLI** : L'outil en ligne de commande pour interagir avec vos ressources Cloud Microsoft Azure
* **Azure Functions Core Tools v4** : Nécessaire pour faire tourner et déboguer les fonctions serverless en local (`func`)

---

## 5. Guide d'Installation et Configuration Locale

### Étape 1 : Clonage et initialisation de l'environnement Python avec `uv`

Placez-vous dans votre dossier de travail et préparez l'environnement virtuel :

```powershell
# Création de l'environnement virtuel avec uv
uv venv --python 3.13

# Activation de l'environnement virtuel (Windows)
.venv\\Scripts\\activate

# Installation des dépendances globales du projet
uv pip install -r requirements.txt

```

### Étape 2 : Configuration des secrets locaux des Azure Functions

Naviguez dans le dossier `prisma-functions` et configurez le fichier de configuration locale pour lier vos fonctions locales à l'infrastructure Azure provisionnée :

Ouvrez ou créez le fichier `prisma-functions/local.settings.json` :

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "DefaultEndpointsProtocol=https;AccountName=prismastorage123;AccountKey=VOTRE_CLE_BLOB_STORAGE;EndpointSuffix=core.windows.net",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "API_PREDICT_URL": "[https://prisma-api-app.icywater-748180f5.germanywestcentral.azurecontainerapps.io/prisma/predict](https://prisma-api-app.icywater-748180f5.germanywestcentral.azurecontainerapps.io/prisma/predict)",
    "CosmosDBConnectionString": "AccountEndpoint=[https://prismacosmos123.documents.azure.com:443/;AccountKey=VOTRE_CLE_COSMOS_DB](https://prismacosmos123.documents.azure.com:443/;AccountKey=VOTRE_CLE_COSMOS_DB);"
  },
  "Host": {
    "CORS": "*"
  }
}

```

*Note : Le paramètre `"Host": {"CORS": "*"}` est impératif en local pour permettre à votre fichier HTML d'interroger la fonction HTTP sans blocage de sécurité navigateur.*

### Étape 3 : Lancement des Azure Functions en local

Installez les dépendances spécifiques du dossier de fonctions et démarrez le moteur d'exécution local d'Azure :

```powershell
cd prisma-functions
uv pip install -r requirements.txt

# Démarrage de l'hôte local Azure Functions
func start

```

Vous devriez voir les fonctions `blob_dispatcher`, `queue_worker`, et `get_inferences` s'initialiser correctement sous l'URL `http://localhost:7071`.

### Étape 4 : Lancement du Dashboard Web

Ouvrez simplement le fichier `web/index.html` dans le navigateur web de votre choix (Chrome, Firefox, Edge). Grâce à la boucle Javascript, l'interface va interroger l'Azure Function toutes les 5 secondes et se mettre à jour dynamiquement.

---

## 6. Guide de Déploiement Cloud (Production)

### 1. Inférence Engine : Docker & Azure Container Apps

Pour mettre à jour ou déployer l'API d'inférence IA dans le cloud :

```powershell
# Connexion à l'Azure CLI et au registre d'images privé (ACR)
az login
az acr login --name prismaregistry123

# Build de l'image Docker locale optimisée pour l'API PRISMA
docker build -t prismaregistry123.azurecr.io/prisma-api:v1 .

# Envoi de l'image sur Azure Container Registry
docker push prismaregistry123.azurecr.io/prisma-api:v1

# Déploiement ou mise à jour sur Azure Container Apps (avec mot de passe administrateur ACR requis)
az containerapp create `
  --name prisma-api-app `
  --resource-group PRISMA-RG `
  --environment prisma-env `
  --image prismaregistry123.azurecr.io/prisma-api:v1 `
  --target-port 8000 `
  --ingress 'external' `
  --min-replicas 0 `
  --max-replicas 3 `
  --registry-server prismaregistry123.azurecr.io `
  --registry-username prismaregistry123 `
  --registry-password ENTRER_VOTRE_MOT_DE_PASSE_ACR

```

### 2. Publication des Azure Functions sur le Cloud

Une fois les tests locaux validés, vous pouvez packager et pousser vos fonctions serverless directement sur votre ressource Azure Function App dédiée :

```powershell
cd prisma-functions
func azure functionapp publish NomDeVotreFunctionAppSurAzure

```

### 3. Déploiement de l'interface graphique

Le répertoire `web/` contenant uniquement des éléments statiques (HTML standard et requêtes fetch asynchrones), il se déploie très facilement via **Azure Static Web Apps** :

```powershell
az staticwebapp create `
  --name prisma-dashboard `
  --resource-group PRISMA-RG `
  --source "[https://github.com/votre-depot/prisma](https://github.com/votre-depot/prisma)" `
  --location germanywestcentral `
  --branch main `
  --app-location "web"

```

---

## 7. Spécifications techniques d'Inférence (Détails Jurys)

Lors de la soutenance technique ou de l'analyse MLOps, les choix d'ingénierie suivants peuvent être mis en avant :

* **Sensibilité à la Casse (Linux Core) :** L'API s'exécute sur une image de base Linux au sein d'un conteneur. Le système de fichier y est strictement sensible à la casse. Le code extrait l'extension `.JPG` en majuscule afin de correspondre parfaitement aux images physiques importées depuis Windows.
* **Gestion de la clé de partition Cosmos DB :** Le conteneur Cosmos DB est optimisé avec la clé de partition `/prediction`. Ce choix garantit un routage physique performant des requêtes (faible coût en RU/s) lors des requêtes de filtrage sur le dashboard (ex: isoler rapidement l'ensemble des PCB rejetés pour cause de "Short Circuit").
* **FinOps & Scalabilité :** Grâce à l'utilisation combinée du plan de Consommation Azure Functions (1M requêtes gratuites/mois) et du niveau gratuit Cosmos DB (1000 RU/s à vie), l'infrastructure complète affiche un coût opérationnel proche de 0€ en dehors des phases d'inférence active sur Container Apps.
"""
```