from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import time
import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

try:
    from api.preprocess import run_opencv_detection, preprocess_opencv_crop
except ModuleNotFoundError:
    from preprocess import run_opencv_detection, preprocess_opencv_crop

ROOT_DIR = Path(__file__).resolve().parent.parent

# 2. On construit des chemins absolus blindés (Path gère les \ de Windows tout seul)
MODEL_PATH = str(ROOT_DIR / "model" / "pcb_classifier_v1.0.0.onnx")
GOLDEN_SAMPLES_DIR = str(ROOT_DIR / "model" / "data" / "PCB_DATASET" /"PCB_USED")
CLASSES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper'] 

router = APIRouter()

try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle ONNX : {e}")

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/version")
async def get_version():
    return {"model_version": "1.1.0", "onnx_model": MODEL_PATH}

@router.post("/predict")
async def predict_pcb(
    file: UploadFile = File(...),
    template_name: str = Form(...) # Reçoit l'identifiant du circuit (ex: "circuit_type_A")
):
    start_time = time.time()
    
    # 1. Vérification et chargement du Golden Sample correspondant
    golden_filename = f"{template_name}.JPG" # ou .png selon tes fichiers
    golden_path = os.path.join(GOLDEN_SAMPLES_DIR, golden_filename)
    
    if not os.path.exists(golden_path):
        raise HTTPException(
            status_code=400, 
            detail=f"Fichier introuvable. Le serveur a cherché exactement ici : {golden_path}"
        )
    
    ref_img = cv2.imread(golden_path)
    
    # 2. Lecture de l'image de test envoyée par la requête
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        test_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Fichier image invalide ou corrompu.")

    # 3. Étape de Détection (Soustraction OpenCV)
    defect_crop, bbox = run_opencv_detection(ref_img, test_img) # type: ignore
    
    # Cas où la carte ne présente aucune différence significative
    if defect_crop is None:
        latency = round((time.time() - start_time) * 1000, 2)
        return {
            "defect_detected": False,
            "prediction": "Healthy",
            "confidence": 1.0,
            "bbox": None,
            "latency_ms": latency
        }
        
    # 4. Étape de Classification (Si un défaut a été isolé)
    try:
        input_tensor = preprocess_opencv_crop(defect_crop)
        
        # Inférence ONNX Runtime
        outputs = session.run(None, {input_name: input_tensor}) # type: ignore
        predictions = outputs[0]
        
        # Softmax pour obtenir les probabilités
        exp_preds = np.exp(predictions[0] - np.max(predictions[0])) # type: ignore
        probabilities = exp_preds / exp_preds.sum()
        
        predicted_idx = np.argmax(probabilities)
        confidence = float(probabilities[predicted_idx])
        predicted_class = CLASSES[predicted_idx]
        
        latency = round((time.time() - start_time) * 1000, 2)
        
        # Réponse enrichie et hautement actionnable pour la base de données
        return {
            "defect_detected": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "bbox": bbox, # Renvoie les coordonnées exactes pour l'affichage de l'encadré rouge
            "latency_ms": latency
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {str(e)}")