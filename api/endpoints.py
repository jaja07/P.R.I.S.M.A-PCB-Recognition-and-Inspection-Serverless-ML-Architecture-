from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import time
import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from api.preprocess import run_opencv_detection, preprocess_opencv_crop
# try:
#     from api.preprocess import run_opencv_detection, preprocess_opencv_crop
# except ModuleNotFoundError:
#     from preprocess import run_opencv_detection, preprocess_opencv_crop

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = str(ROOT_DIR / "model" / "pcb_classifier_v1.0.0.onnx")
GOLDEN_SAMPLES_DIR = str(ROOT_DIR / "model" / "data" / "PCB_DATASET" /"PCB_USED")
CLASSES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper'] 

router = APIRouter()

ALL_PREDICTIONS = ["Healthy"] + CLASSES

APP_METRICS = {
    # Métriques Système
    "total_calls": 0,
    "total_latency_ms": 0.0,
    "model_loading_time_ms": 0.0,
    
    # Métriques Machine Learning (Observabilité)
    "ml_observability": {
        "prediction_distribution": {cls: 0 for cls in ALL_PREDICTIONS},
        "cumulative_confidence": {cls: 0.0 for cls in ALL_PREDICTIONS},
        "low_confidence_warnings": 0, # Prédictions avec confiance < 70%
        "confidence_threshold": 0.70
    }
}

start_load_time = time.time()
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    APP_METRICS["model_loading_time_ms"] = round((time.time() - start_load_time) * 1000, 2)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle ONNX : {e}")

# ---  Modèles Pydantic ---

class BBoxModel(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int

class PredictionResponseModel(BaseModel):
    defect_detected: bool
    prediction: str
    confidence: float
    bbox: BBoxModel | None = None
    latency_ms: float

# ---  Endpoints ---
@router.get("/health")
async def health_check():
    return {"status": "OK", "model_loading_time_ms": APP_METRICS["model_loading_time_ms"]}

@router.get("/version")
async def get_version():
    return {"api_version": "1.1.0", "model_version": "1.0.0", "onnx_model": os.path.basename(MODEL_PATH)}

@router.post("/predict", response_model=PredictionResponseModel)
async def predict_pcb(
    file: UploadFile = File(...),
    template_name: str = Form(..., pattern=r"^[a-zA-Z0-9_-]+$")
):
    start_time = time.time()
    APP_METRICS["total_calls"] += 1
    
    golden_filename = f"{template_name}.JPG" 
    golden_path = os.path.join(GOLDEN_SAMPLES_DIR, golden_filename)
    
    if not os.path.exists(golden_path):
        raise HTTPException(status_code=400, detail=f"Fichier introuvable : {golden_path}")
    
    ref_img = cv2.imread(golden_path)
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        test_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Fichier image invalide.")

    defect_crop, bbox = run_opencv_detection(ref_img, test_img) # type: ignore
    
    # --- Cas 1 : Healthy ---
    if defect_crop is None:
        latency = round((time.time() - start_time) * 1000, 2)
        APP_METRICS["total_latency_ms"] += latency
        
        # Mise à jour des métriques ML
        APP_METRICS["ml_observability"]["prediction_distribution"]["Healthy"] += 1
        APP_METRICS["ml_observability"]["cumulative_confidence"]["Healthy"] += 1.0 # Confiance de 100% (OpenCV)
        
        return PredictionResponseModel(
            defect_detected=False, prediction="Healthy",
            confidence=1.0, bbox=None, latency_ms=latency
        )
        
    # --- Cas 2 : Défaut ---
    try:
        input_tensor = preprocess_opencv_crop(defect_crop)
        outputs = session.run(None, {input_name: input_tensor})
        predictions = outputs[0]
        
        exp_preds = np.exp(predictions[0] - np.max(predictions[0])) # type: ignore
        probabilities = exp_preds / exp_preds.sum()
        predicted_idx = np.argmax(probabilities)
        
        confidence = float(probabilities[predicted_idx])
        predicted_class = CLASSES[predicted_idx]
        
        latency = round((time.time() - start_time) * 1000, 2)
        APP_METRICS["total_latency_ms"] += latency
        
        # --- MISE À JOUR DES MÉTRIQUES ML ---
        ml_obs = APP_METRICS["ml_observability"]
        ml_obs["prediction_distribution"][predicted_class] += 1
        ml_obs["cumulative_confidence"][predicted_class] += confidence
        
        if confidence < ml_obs["confidence_threshold"]:
            ml_obs["low_confidence_warnings"] += 1
        
        return PredictionResponseModel(
            defect_detected=True, prediction=predicted_class,
            confidence=confidence, bbox=BBoxModel(**bbox), latency_ms=latency # type: ignore
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {str(e)}")
    
@router.get("/metrics")
async def get_metrics():
    """
    Retourne les métriques système ET l'état de santé du Machine Learning.
    Ce JSON est formaté pour être ingéré par Azure Application Insights ou Cosmos DB.
    """
    calls = APP_METRICS["total_calls"]
    avg_latency = round(APP_METRICS["total_latency_ms"] / calls, 2) if calls > 0 else 0.0
    
    # Calcul dynamique des confiances moyennes par classe
    avg_confidences = {}
    ml_obs = APP_METRICS["ml_observability"]
    
    for cls in ALL_PREDICTIONS:
        count = ml_obs["prediction_distribution"][cls]
        cum_conf = ml_obs["cumulative_confidence"][cls]
        avg_confidences[cls] = round(cum_conf / count, 4) if count > 0 else 0.0

    return {
        "system_metrics": {
            "total_calls": calls,
            "average_latency_ms": avg_latency
        },
        "ml_metrics": {
            "low_confidence_warnings": ml_obs["low_confidence_warnings"],
            "prediction_distribution": ml_obs["prediction_distribution"],
            "average_confidence_per_class": avg_confidences
        }
    }