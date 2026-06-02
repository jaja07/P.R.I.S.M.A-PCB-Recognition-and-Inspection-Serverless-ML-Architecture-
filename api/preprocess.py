import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_opencv_detection(ref_img_np: np.ndarray, test_img_np: np.ndarray):
    """Effectue la soustraction d'image pour isoler le défaut et renvoie le crop et sa bbox."""
    # Alignement de sécurité des dimensions
    if ref_img_np.shape != test_img_np.shape:
        test_img_np = cv2.resize(test_img_np, (ref_img_np.shape[1], ref_img_np.shape[0]))

    # Niveaux de gris et flou gaussien anti-bruit
    ref_gray = cv2.cvtColor(ref_img_np, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img_np, cv2.COLOR_BGR2GRAY)
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    test_blur = cv2.GaussianBlur(test_gray, (5, 5), 0)

    # Soustraction absolue et binarisation
    diff = cv2.absdiff(ref_blur, test_blur)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Extraction des contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Sélection de la plus grande anomalie
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 50: # Seuil pour ignorer le bruit de fond
        return None, None

    # Coordonnées de la Bounding Box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Ajout d'une marge (padding) autour du défaut
    padding = 15
    y1 = max(0, y - padding)
    y2 = min(test_img_np.shape[0], y + h + padding)
    x1 = max(0, x - padding)
    x2 = min(test_img_np.shape[1], x + w + padding)

    # Découpage (Crop) du patch défectueux
    defect_crop = test_img_np[y1:y2, x1:x2]
    
    bbox = {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}
    return defect_crop, bbox


def preprocess_opencv_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Prépare le crop OpenCV (BGR) au format float32 attendu par le modèle ONNX."""
    # Passage de BGR (OpenCV) à RGB (PyTorch/Standard)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (224, 224))
    
    # Normalisation à [0, 1]
    img_array = crop_resized.astype(np.float32) / 255.0
    
    # Normalisation ImageNet (Moyenne et Écart-type forcés en float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Transposition des axes (H, W, C) -> (C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array