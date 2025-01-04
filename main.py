from fastapi import FastAPI, File, UploadFile
import os
from deepface import DeepFace
from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import KMeans
from rembg import remove  # For segmentation

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


from fastapi import FastAPI, File, UploadFile
import os
from deepface import DeepFace
from PIL import Image
import cv2
import numpy as np
from rembg import remove  # For basic segmentation

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
