from fastapi import FastAPI, File, UploadFile
import os
from deepface import DeepFace
from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import KMeans
from rembg import remove  # For segmentation
from ultralytics import YOLO
import torch

# Segment Anything imports
from segment_anything import sam_model_registry, SamPredictor


app = FastAPI()

UPLOAD_FOLDER = "uploaded_images"
OUTPUT_FOLDER = "color_variations"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Function: Check if the image contains a female
def is_female(image_path):
    try:
        # Analyze the image using DeepFace
        result = DeepFace.analyze(img_path=image_path, actions=["gender"], enforce_detection=False)
        dominant_gender = result[0]['dominant_gender']
        print(f"Dominant Gender: {dominant_gender}")  # Debugging
        return dominant_gender.lower() == "woman"
    except Exception as e:
        print("Error in gender detection:", e)
        return False


# Function: Extract dominant color
def extract_dominant_color(image_path, k=1):
    try:
        # Load image and prepare it for KMeans clustering
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, (100, 100))  # Downscale for faster processing
        reshaped_image = resized_image.reshape((-1, 3))

        # KMeans clustering for color detection
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(reshaped_image)
        dominant_color = [int(c) for c in kmeans.cluster_centers_[0]]
        print(f"Dominant Color: {dominant_color}")  # Debugging
        return tuple(dominant_color)
    except Exception as e:
        print("Error extracting color:", e)
        return (0, 0, 0)


# Function: Generate color variation images
def generate_color_variation_images(image_path, base_color, variations):
    try:
        # Segment clothing using rembg
        input_image = Image.open(image_path)
        segmented_image = remove(input_image)  # Remove background, keep foreground

        # Convert segmented image to OpenCV format
        segmented_np = np.array(segmented_image)
        mask = cv2.cvtColor(segmented_np, cv2.COLOR_RGBA2GRAY)  # Grayscale mask
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)  # Binary mask

        output_images = []
        for idx, color in enumerate(variations):
            # Create a blank image filled with the target color
            color_image = np.zeros_like(segmented_np[:, :, :3])
            color_image[:] = color

            # Apply the color to the segmented clothing area
            clothing_colored = cv2.bitwise_and(color_image, color_image, mask=mask)
            background = cv2.bitwise_and(segmented_np[:, :, :3], segmented_np[:, :, :3], mask=cv2.bitwise_not(mask))
            final_image = cv2.add(clothing_colored, background)

            # Save the generated image
            output_path = os.path.join(OUTPUT_FOLDER, f"variation_{idx + 1}.png")
            cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
            output_images.append(output_path)

        print("Generated images:", output_images)  # Debugging
        return output_images
    except Exception as e:
        print("Error generating variations:", e)
        return []


# API Endpoint: Generate color variations for clothing if female is detected
@app.post("/generate-color-variations/")
async def generate_color_variations(file: UploadFile = File(...)):
    # Save the uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    print(f"File saved at: {file_path}")  # Debugging

    # Step 1: Check if the image contains a female
    female_detected = is_female(file_path)
    if not female_detected:
        return {"contains_female": False, "message": "No female detected in the image."}

    # Step 2: Extract dominant clothing color
    dominant_color = extract_dominant_color(file_path)

    # Step 3: Define color variations
    color_variations = [
        (min(dominant_color[0] + 30, 255), dominant_color[1], dominant_color[2]),
        (dominant_color[0], min(dominant_color[1] + 30, 255), dominant_color[2]),
        (dominant_color[0], dominant_color[1], min(dominant_color[2] + 30, 255)),
        (max(dominant_color[0] - 30, 0), dominant_color[1], dominant_color[2]),
        (dominant_color[0], max(dominant_color[1] - 30, 0), dominant_color[2]),
        (dominant_color[0], dominant_color[1], max(dominant_color[2] - 30, 0))
    ]

    # Step 4: Generate images with color variations
    generated_images = generate_color_variation_images(file_path, dominant_color, color_variations)

    # Return the results
    return {
        "contains_female": True,
        "dominant_color": dominant_color,
        "generated_images": generated_images
    }


app = FastAPI()

UPLOAD_FOLDER = "uploaded_images"
OUTPUT_FOLDER = "color_variations"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Function: Check if the image contains a female
def is_female(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=["gender"], enforce_detection=False)
        dominant_gender = result[0]['dominant_gender']
        print(f"Dominant Gender: {dominant_gender}")
        return dominant_gender.lower() == "woman"
    except Exception as e:
        print("Error in gender detection:", e)
        return False



# Function: Detect dress region using segmentation (simple mask refinement)
def extract_clothing_mask(image_path):
    try:
        input_image = cv2.imread(image_path)
        segmented_image = remove(Image.open(image_path))  # Remove background
        segmented_np = np.array(segmented_image)

        # Convert to grayscale and refine the mask
        gray = cv2.cvtColor(segmented_np, cv2.COLOR_RGBA2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find largest connected component (assumes clothing is the largest region)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create refined mask
        refined_mask = np.zeros_like(mask)
        cv2.drawContours(refined_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        return refined_mask, input_image
    except Exception as e:
        print("Error extracting clothing mask:", e)
        return None, None


# Function: Replace color in the clothing region
def generate_color_variation_images(image_path, variations):
    try:
        # Extract refined mask and original image
        mask, original_image = extract_clothing_mask(image_path)
        if mask is None:
            return []

        output_images = []
        for idx, color in enumerate(variations):
            # Create a blank canvas with the target color
            color_image = np.zeros_like(original_image)
            color_image[:, :] = color

            # Apply mask to replace the color only in the clothing region
            clothing_colored = cv2.bitwise_and(color_image, color_image, mask=mask)
            background = cv2.bitwise_and(original_image, original_image, mask=cv2.bitwise_not(mask))
            final_image = cv2.add(clothing_colored, background)

            # Save the result
            output_path = os.path.join(OUTPUT_FOLDER, f"variation_{idx + 1}.png")
            cv2.imwrite(output_path, final_image)
            output_images.append(output_path)

        return output_images
    except Exception as e:
        print("Error generating variations:", e)
        return []



# API Endpoint: Generate color variations for clothing if female is detected
@app.post("/generate-color-variations/")
async def generate_color_variations(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    print(f"File saved at: {file_path}")

    # Step 1: Check if the image contains a female
    female_detected = is_female(file_path)
    if not female_detected:
        return {"contains_female": False, "message": "No female detected in the image."}

    # Step 2: Define color variations
    color_variations = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 165, 0)   # Orange
    ]

    # Step 3: Generate images with color variations
    generated_images = generate_color_variation_images(file_path, color_variations)

    return {
        "contains_female": True,
        "generated_images": generated_images
    }



app = FastAPI()

UPLOAD_FOLDER = "uploaded_images"
OUTPUT_FOLDER = "color_variations"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Function: Check if the image contains a female
def is_female(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=["gender"], enforce_detection=False)
        dominant_gender = result[0]['dominant_gender']
        print(f"Dominant Gender: {dominant_gender}")
        return dominant_gender.lower() == "woman"
    except Exception as e:
        print("Error in gender detection:", e)
        return False


# Function: Detect T-shirt using color thresholding and replace its color
def change_tshirt_color(image_path, variations):
    try:
        # Load the image
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for detecting the T-shirt color (orange in this case)
        lower_range = np.array([5, 100, 100])   # Lower HSV for orange
        upper_range = np.array([15, 255, 255])  # Upper HSV for orange

        # Create a mask for the T-shirt
        mask = cv2.inRange(hsv, lower_range, upper_range)
        mask = cv2.medianBlur(mask, 7)  # Smooth the mask for better edges

        output_images = []

        for idx, color in enumerate(variations):
            # Create a blank canvas with the new color in BGR format
            new_color_image = np.zeros_like(image)
            new_color_image[:] = color  # Set the new color (BGR format)

            # Replace the T-shirt region with the new color
            tshirt_colored = cv2.bitwise_and(new_color_image, new_color_image, mask=mask)
            background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
            final_image = cv2.add(tshirt_colored, background)

            # Save the generated image
            output_path = os.path.join(OUTPUT_FOLDER, f"variation_{idx + 1}.png")
            cv2.imwrite(output_path, final_image)
            output_images.append(output_path)

        return output_images
    except Exception as e:
        print("Error changing T-shirt color:", e)
        return []


# API Endpoint: Generate T-shirt color variations if a female is detected
@app.post("/generate-color-variations/")
async def generate_color_variations(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    print(f"File saved at: {file_path}")

    # Step 1: Check if the image contains a female
    female_detected = is_female(file_path)
    if not female_detected:
        return {"contains_female": False, "message": "No female detected in the image."}

    # Step 2: Define new colors for variations (BGR format)
    color_variations = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (128, 0, 128)   # Purple
    ]

    # Step 3: Change T-shirt color
    generated_images = change_tshirt_color(file_path, color_variations)

    return {
        "contains_female": True,
        "generated_images": generated_images
    }

app = FastAPI()

UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Adjust if needed

# Load Segment Anything Model (SAM)
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Place the SAM checkpoint in your working directory
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
sam_predictor = SamPredictor(sam)


def detect_female_and_crop(image_path):
    """
    Detect persons using YOLO, check each one for female gender, and return a crop of the first female found.
    If no female is found, return None.
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
                    return person_crop  # Return the crop of the female
            except Exception as e:
                print("Error in gender detection:", e)
                continue
    return None


def segment_garment_sam(image: np.ndarray) -> np.ndarray:
    """
    Use SAM to segment a region of the image that presumably contains the garment.
    For demonstration, we assume the upper half of the person_crop is the garment area.
    We'll provide a box prompt over the upper half of the crop to SAM.

    This is a very naive approach:
    - We take the top half of the person image as a bounding box prompt.
    - SAM returns a mask. Ideally, you'd refine the prompt or use interactive tools to get a precise garment mask.
    """
    h, w, _ = image.shape
    # Define a box around the upper body (this is a guess/placeholder)
    # For a better approach, you could use pose estimation to find torso keypoints.
    box = np.array([w * 0.1, h * 0.05, w * 0.9, h * 0.6])  # [x_min, y_min, x_max, y_max]

    sam_predictor.set_image(image, image_format="BGR")
    masks, scores, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=True
    )

    # Pick the mask with the highest score
    if len(masks) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    max_idx = np.argmax(scores)
    selected_mask = masks[max_idx]

    # Convert boolean mask to uint8 mask
    mask = (selected_mask * 255).astype(np.uint8)
    return mask


def extract_dominant_color(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute the mean color in the garment region and convert it to a named color.
    """
    binary_mask = (mask > 128)
    garment_pixels = image[binary_mask]

    if len(garment_pixels) == 0:
        return None

    mean_bgr = np.mean(garment_pixels, axis=0)
    mean_bgr_reshaped = np.uint8([[mean_bgr]])
    mean_hsv = cv2.cvtColor(mean_bgr_reshaped, cv2.COLOR_BGR2HSV)[0][0]

    hsv_tuple = (int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2]))
    color_name = hsv_to_color_name(hsv_tuple)

    return {
        "bgr": (float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])),
        "hsv": hsv_tuple,
        "color_name": color_name
    }


def hsv_to_color_name(hsv):
    """
    Convert HSV to a nearest color name. This is a simplistic approach:
    We define a few major colors by HSV ranges and return the closest match.
    hsv: (H, S, V)
    """
    H, S, V = hsv

    # If saturation is low, it might be gray/white/black
    if S < 30:
        if V < 50:
            return "Black"
        elif V < 200:
            return "Gray"
        else:
            return "White"

    # Otherwise, use hue ranges to determine the color
    # Common hue ranges (0-179 in OpenCV):
    # Red: 0-10 or 170-179, Yellow: 20-30, Green: 35-85, Cyan: 85-100, Blue: 100-130, Magenta: 140-170
    if (H <= 10 or H >= 170):
        return "Red"
    elif 11 <= H < 30:
        return "Orange/Yellow"
    elif 30 <= H < 35:
        return "Yellowish"
    elif 35 <= H < 85:
        return "Green"
    elif 85 <= H < 100:
        return "Cyan"
    elif 100 <= H < 130:
        return "Blue"
    elif 130 <= H < 170:
        return "Magenta/Purple"
    else:
        return "Unknown"


@app.post("/identify-dress-color/")
async def identify_dress_color(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    print(f"File saved at: {file_path}")

    # Detect a female and get her cropped image
    person_crop = detect_female_and_crop(file_path)
    if person_crop is None:
        return {"contains_female": False, "message": "No female detected in the image."}

    # Segment the garment using SAM
    garment_mask = segment_garment_sam(person_crop)
    if cv2.countNonZero(garment_mask) == 0:
        return {"contains_female": True, "dress_detected": False, "message": "No garment detected by segmentation."}

    # Identify the color of the garment
    dominant_color = extract_dominant_color(person_crop, garment_mask)
    if dominant_color is None:
        return {"contains_female": True, "dress_detected": True, "message": "Could not determine dominant color."}

    return {
        "contains_female": True,
        "dress_detected": True,
        "dominant_color": dominant_color,
        "message": "Dominant color identified successfully."
    }



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

def detect_female_and_crop(image_path):
    """
    Detect persons using YOLO, check each one for female gender, and return a crop of the first female found.
    If no female is found, return None.
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
                    return person_crop  # Return the crop of the female
            except Exception as e:
                print("Error in gender detection:", e)
                continue
    return None

def segment_garment_sam(image: np.ndarray) -> np.ndarray:
    """
    Use SAM to segment the garment by focusing on the upper body and excluding non-garment regions.
    """
    h, w, _ = image.shape
    # Box around the upper body - assumes garment is in this region
    box = np.array([w * 0.15, h * 0.3, w * 0.85, h * 0.9])  # Adjusted box for upper body

    sam_predictor.set_image(image, image_format="BGR")
    masks, scores, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=True
    )

    if len(masks) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Select the mask with the highest score and refine
    max_idx = np.argmax(scores)
    selected_mask = masks[max_idx]
    refined_mask = (selected_mask * 255).astype(np.uint8)

    # Step to clean up mask: Remove small areas (e.g., face/hands)
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10000:  # Threshold to remove small regions
            cv2.drawContours(refined_mask, [cnt], -1, 0, -1)

    return refined_mask

@app.post("/select-dress/")
async def select_dress(file: UploadFile = File(...)):
    """
    Select and segment the dress from an image containing a female, highlight the dress in light blue.
    """
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    print(f"File saved at: {file_path}")

    # Step 1: Detect the female and crop the image
    person_crop = detect_female_and_crop(file_path)
    if person_crop is None:
        return {"contains_female": False, "message": "No female detected in the image."}

    # Step 2: Segment the dress using SAM with refined mask
    garment_mask = segment_garment_sam(person_crop)
    if cv2.countNonZero(garment_mask) == 0:
        return {"contains_female": True, "dress_detected": False, "message": "No dress detected by segmentation."}

    # Step 3: Resize mask to match original image
    original_image = cv2.imread(file_path)
    garment_mask_resized = cv2.resize(garment_mask, (original_image.shape[1], original_image.shape[0]))

    # Step 4: Create a transparent blue overlay
    blue_overlay = np.zeros_like(original_image, dtype=np.uint8)
    blue_overlay[:, :] = (255, 0, 0)  # Light blue (BGR)
    alpha = 0.5  # Transparency factor

    # Step 5: Apply the mask to the original image
    highlighted_image = original_image.copy()
    mask_indices = garment_mask_resized > 0
    highlighted_image[mask_indices] = cv2.addWeighted(
        original_image[mask_indices], 1 - alpha, blue_overlay[mask_indices], alpha, 0
    )

    # Save the highlighted dress result
    dress_output_path = os.path.join(SEGMENTED_DRESS, "highlighted_dress.png")
    cv2.imwrite(dress_output_path, highlighted_image)

    return {
        "contains_female": True,
        "dress_detected": True,
        "highlighted_dress_image": dress_output_path,
        "message": "Dress highlighted successfully with light blue color."
    }



