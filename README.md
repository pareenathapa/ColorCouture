WEBSITE NAME-ColorCouture

ColorCouture is a deep learning-based fashion application that allows users to upload images and recolor dresses using advanced AI models, including YOLO (You Only Look Once), DeepFace, and Segment Anything Model (SAM). The app detects and segments the dress worn by the person in the image, then applies a chosen color to it. This process can be used for various fashion-related tasks and helps users visualize clothing in different colors.

Features
- Gender Detection: Detects if a female is present in the image using YOLO and DeepFace.
- Garment Segmentation: Uses the Segment Anything Model (SAM) to segment the dress.
- Dress Recoloring: Allows users to recolor the dress in the uploaded image to a selected color (e.g., red, blue, green, yellow, purple, pink).
- Pose Estimation: Uses MediaPipe Pose to estimate the torso region for more accurate segmentation.

 Requirements
To run the ColorCouture app, the following Python libraries needs to be installed:

- fastapi: Web framework for building the API.
- opencv-python: For image processing tasks such as reading and writing images.
- deepface: For gender recognition using deep learning models.
- ultralytics: For loading and using the YOLO model for object detection.
- torch: PyTorch for running models, including the Segment Anything Model (SAM).
- segment-anything: For the SAM-based segmentation of garments.
- mediapipe: For pose estimation to accurately detect body parts.

Installation Instructions

1. Clone this repository:
    ```bash
    git clone https://github.com/pareenathapa/ColorCouture.git
    ```

2. Navigate into the project directory:
    ```bash
    cd ColorCouture
    ```

3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

5. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

Usage

1. Run the FastAPI server:
    Once the required packages are installed, run the FastAPI server to start the application:
    ```bash
    uvicorn main:app --reload
    ```

2. Upload an image:
    Upload an image with a dress and select a color to recolor the dress.

    POST endpoint:
    - URL: `http://127.0.0.1:8000/recolor-dress/`
    - Method: `POST`
    - Parameters:
        - `file`: The image file to upload (multipart form-data).
        - `color`: The desired color for the dress (optional, default: blue). Available colors: `red`, `blue`, `green`, `yellow`, `purple`, `pink`.

    Example: 
    ```bash
    curl -X 'POST' \
    'http://127.0.0.1:8000/recolor-dress/?color=red' \
    -F 'file=@your_image.jpg'
    ```

3. View the result:
    After successfully processing the image, the server will return the recolored dress image path in the response and access the image at the URL:
    - `http://127.0.0.1:8000/segmented_dress/recolored_dress_red.png`

Example Output

After processing an image, you will get a response similar to this:

```json
{
  "contains_female": true,
  "dress_detected": true,
  "recolored_dress_image": "segmented_dress/recolored_dress_red.png",
  "message": "Dress recolored successfully to red."
}

File Structure

ColorCouture/
│
├── uploaded_images/              # Temporary directory to store uploaded images
├── segmented_dress/              # Directory to store the recolored dress images
├── main.py                       # FastAPI application file
├── requirements.txt              # Python dependencies file
├── sam_vit_h_4b8939.pth          # Segment Anything Model checkpoint
├── yolov8n.pt                    # YOLOv8 model checkpoint
├── frontend/                     # Frontend files (optional)
└── README.md                     # Project documentation






