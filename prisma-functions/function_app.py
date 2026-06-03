import azure.functions as func
import logging
import json
import os
import requests
import uuid
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient

app = func.FunctionApp()

# ==========================================
# FONCTION 1 : LE DISPATCHER (Blob Trigger)
# ==========================================
@app.blob_trigger(arg_name="myblob", path="input/{name}", connection="AzureWebJobsStorage")
@app.queue_output(arg_name="msg", queue_name="process-queue", connection="AzureWebJobsStorage")
def blob_dispatcher(myblob: func.InputStream, msg: func.Out[str]):
    logging.info(f"Nouveau fichier détecté : {myblob.name}")
    
    # On extrait uniquement le nom du fichier (sans le dossier "input/")
    filename = myblob.name.split('/')[-1] # type: ignore
    
    message_payload = {"filename": filename}
    msg.set(json.dumps(message_payload))
    logging.info(f"Message envoyé dans process-queue pour : {filename}")


# ==========================================
# FONCTION 2 : LE WORKER (Queue Trigger)
# ==========================================
@app.queue_trigger(arg_name="msg", queue_name="process-queue", connection="AzureWebJobsStorage")
def queue_worker(msg: func.QueueMessage):
    try:
        # 1. Lecture du message de la file d'attente
        req_body = json.loads(msg.get_body().decode('utf-8'))
        filename = req_body.get('filename')
        
        # Le template_name pour l'API est le nom du fichier sans l'extension (ex: "01.JPG" -> "01")
        template_name = filename.split('_')[0]
        logging.info(f"[WORKER] Début du traitement pour : {filename}")

        # 2. Téléchargement de l'image depuis le Blob Storage
        connection_string = os.environ["AzureWebJobsStorage"]
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        input_blob_client = blob_service_client.get_blob_client(container="input", blob=filename)
        
        image_data = input_blob_client.download_blob().readall()

        # 3. Envoi de l'image à ton API Container Apps
        api_url = os.environ["API_PREDICT_URL"]
        files = {"file": (filename, image_data, "image/jpeg")}
        data = {"template_name": template_name}
        
        response = requests.post(api_url, files=files, data=data)
        response.raise_for_status() # Lève une erreur si l'API ne renvoie pas un code 200 OK
        api_result = response.json()

        # 4. Sauvegarde du résultat JSON dans le conteneur 'output'
        output_blob_client = blob_service_client.get_blob_client(container="output", blob=f"result_{filename}.json")
        output_blob_client.upload_blob(json.dumps(api_result, indent=4), overwrite=True)

        # 5. Enregistrement des métadonnées dans Cosmos DB
        cosmos_connection_string = os.environ["CosmosDBConnectionString"]
        cosmos_client = CosmosClient.from_connection_string(cosmos_connection_string)
        database = cosmos_client.get_database_client("PrismaDB")
        container = database.get_container_client("Inferences")

        document = {
            "id": str(uuid.uuid4()), # Cosmos DB exige un ID unique sous forme de chaîne
            "filename": filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "defect_detected": api_result.get("defect_detected"),
            "prediction": api_result.get("prediction"),
            "confidence": api_result.get("confidence"),
            "latency_ms": api_result.get("latency_ms")
        }
        container.create_item(body=document)

        logging.info(f"[WORKER] Succès ! Inférence terminée et archivée pour {filename}")

    except Exception as e:
        logging.error(f"[WORKER] Erreur critique lors du traitement : {str(e)}")
        raise e
    
# ==========================================
# FONCTION 3 : L'API FRONT-END (HTTP Trigger)
# ==========================================
@app.route(route="inferences", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
def get_inferences(req: func.HttpRequest) -> func.HttpResponse:
    """
    Récupère les 10 dernières inférences depuis Cosmos DB pour le Dashboard.
    """
    try:
        cosmos_connection_string = os.environ["CosmosDBConnectionString"]
        cosmos_client = CosmosClient.from_connection_string(cosmos_connection_string)
        database = cosmos_client.get_database_client("PrismaDB")
        container = database.get_container_client("Inferences")

        # Requête NoSQL pour récupérer les 10 plus récentes
        query = "SELECT * FROM c ORDER BY c.timestamp DESC OFFSET 0 LIMIT 10"
        items = list(container.query_items(
            query=query, 
            enable_cross_partition_query=True
        ))

        return func.HttpResponse(
            json.dumps(items), 
            mimetype="application/json", 
            status_code=200
        )
    except Exception as e:
        logging.error(f"[HTTP] Erreur : {str(e)}")
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=500)