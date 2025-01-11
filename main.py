from fastapi import FastAPI, File, UploadFile
import os
import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import torch
from segment_anything import sam_model_registry, SamPredictor
from mediapipe.python.solutions import pose as mp_pose

from fastapi.staticfiles import StaticFiles

app = FastAPI()

UPLOAD_FOLDER = "uploaded_images"
SEGMENTED_DRESS = "segmented_dress"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_DRESS, exist_ok=True)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Adjust if needed

# Load Segment Anything Model (SAM)
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Place the SAM checkpoint in your working directory
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
sam_predictor = SamPredictor(sam)

pose_estimator = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def detect_female(image_path):
    """
    Detect if a female is present in the image using YOLO and DeepFace.
    Returns True if a female is detected, else False.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image.")

    results = model.predict(source=image_path, conf=0.25, verbose=False)
    detections = results[0].boxes

    for det in detections:
        cls = int(det.cls.cpu().numpy()[0])
        if cls == 0:  # 'person' class
            xyxy = det.xyxy.cpu().numpy()[0].astype(int)
            x1, y1, x2, y2 = xyxy
            person_crop = image[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            crop_path = os.path.join(UPLOAD_FOLDER, "crop_temp.jpg")
            cv2.imwrite(crop_path, person_crop)

            # Check gender
            try:
                result = DeepFace.analyze(img_path=crop_path, actions=["gender"], enforce_detection=False)
                dominant_gender = result[0]['dominant_gender'].lower()
                if dominant_gender == "woman":
                    return True  # Female detected
            except Exception as e:
                print("Error in gender detection:", e)
                continue

    return False

def get_torso_bounding_box(image):
    """
    Use MediaPipe Pose to get a bounding box around the torso region.
    """
    results = pose_estimator.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    h, w, _ = image.shape

    # Get torso keypoints: shoulders and hips
    left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
    right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                      int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
    left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
    right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                 int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))

    # Define the bounding box based on keypoints
    x_min = min(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])
    y_min = min(left_shoulder[1], right_shoulder[1])
    x_max = max(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])
    y_max = max(left_hip[1], right_hip[1])

    return [x_min, y_min, x_max, y_max]

def segment_garment_sam(image: np.ndarray) -> np.ndarray:
    """
    Use SAM to segment the garment, focusing on the torso region using pose estimation.
    If pose estimation fails, fall back to a default bounding box.
    """
    h, w, _ = image.shape

    # Get bounding box using pose estimation
    box = get_torso_bounding_box(image)
    if not box:
        print("Pose detection failed. Using fallback bounding box.")
        box = [int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.9)]  # Default box for upper body

    # Slightly expand the box for better segmentation
    x_min, y_min, x_max, y_max = box
    x_min = max(x_min - 20, 0)
    y_min = max(y_min - 20, 0)
    x_max = min(x_max + 20, w)
    y_max = min(y_max + 20, h)

    # Set SAM input and predict
    try:
        sam_predictor.set_image(image, image_format="BGR")
        masks, scores, _ = sam_predictor.predict(
            box=np.array([x_min, y_min, x_max, y_max]),
            multimask_output=True
        )
        if len(masks) == 0:
            print("SAM failed to generate masks.")
            return np.zeros((h, w), dtype=np.uint8)

        # Select the mask with the highest score
        max_idx = np.argmax(scores)
        return (masks[max_idx] * 255).astype(np.uint8)
    except Exception as e:
        print(f"SAM prediction error: {str(e)}")
        return np.zeros((h, w), dtype=np.uint8)


def get_unique_filename(base_path, filename):
    """
    Generate a unique filename by appending a count if the file already exists.
    """
    name, ext = os.path.splitext(filename)
    counter = 1
    unique_path = os.path.join(base_path, filename)
    while os.path.exists(unique_path):
        unique_path = os.path.join(base_path, f"{name}_{counter}{ext}")
        counter += 1
    return unique_path

def recolor_dress(image: np.ndarray, mask: np.ndarray, new_color: tuple) -> np.ndarray:
    """
    Change the color of the dress in the image using the mask.

    Args:
        image (np.ndarray): Original input image.
        mask (np.ndarray): Binary mask of the dress region.
        new_color (tuple): Target color in BGR format, e.g., (0, 255, 0) for green.

    Returns:
        np.ndarray: Image with the recolored dress.
    """
    # Convert the image to HSV color space for better control over color changes
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Ensure the mask is a single-channel binary mask
    mask = (mask > 0).astype(np.uint8)  # Make sure mask is binary (0 or 1)

    # Define target color in HSV (convert BGR -> HSV)
    target_color = np.uint8([[new_color]])  # Wrap color for conversion
    target_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)[0][0]

    # Apply the new color only to the dress region
    hsv_image[:, :, 0] = np.where(mask == 1, target_hsv[0], hsv_image[:, :, 0])  # Replace Hue
    hsv_image[:, :, 1] = np.where(mask == 1, target_hsv[1], hsv_image[:, :, 1])  # Replace Saturation
    hsv_image[:, :, 2] = np.where(mask == 1, cv2.add(hsv_image[:, :, 2], 20), hsv_image[:, :, 2])  # Adjust Brightness

    # Convert back to BGR color space
    recolored_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return recolored_image

@app.post("/recolor-dress/")
async def recolor_dress_endpoint(file: UploadFile = File(...), color: str = "blue"):
    """
    Endpoint to recolor the dress in the uploaded image.

    Args:
        file (UploadFile): Input image file.
        color (str): Desired color for the dress (e.g., 'red', 'blue', 'green').

    Returns:
        JSON response with the path to the recolored image.
    """
    color_map = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "purple": (128, 0, 128),
        "pink": (255, 182, 193)
    }

    if color not in color_map:
        return {"message": "Invalid color. Available colors: red, blue, green, yellow, purple, pink"}

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    print(f"File saved at: {file_path}")

    # Step 1: Detect if a female is present
    if not detect_female(file_path):
        return {"contains_female": False, "message": "No female detected in the image."}

    # Step 2: Segment the dress
    image = cv2.imread(file_path)
    garment_mask = segment_garment_sam(image)
    if cv2.countNonZero(garment_mask) == 0:
        return {"contains_female": True, "dress_detected": False, "message": "No dress detected by segmentation."}

    # Step 3: Apply recoloring
    new_color = color_map[color]
    recolored_image = recolor_dress(image, garment_mask, new_color)

    # Step 4: Save the final image
    recolored_output_path = get_unique_filename(SEGMENTED_DRESS, f"recolored_dress_{color}.png")
    cv2.imwrite(recolored_output_path, recolored_image)

    return {
        "contains_female": True,
        "dress_detected": True,
        "recolored_dress_image": recolored_output_path,
        "message": f"Dress recolored successfully to {color}."
    }


app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")
app.mount("/segmented_dress", StaticFiles(directory="segmented_dress"), name="segmented_dress")