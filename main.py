from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Load the trained YOLOv8 model (ensure you're using the smallest version like yolov8n.pt)
model = YOLO("runs/detect/train6/weights/best.pt")  # Update if your path is different

# Move the model to GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

app = Flask(__name__)
CORS(app)

@app.route('/detect-wall', methods=['POST'])
def detect_wall():
    print("📸 Request received for wall detection")
    
    try:
        if 'image' not in request.files:
            print("❌ No image in request")
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        
        # Read the image using OpenCV
        img_array = np.frombuffer(file.read(), np.uint8)  # Convert to numpy array from the file buffer
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode image as color (RGB)
        
        # Resize the image to reduce memory footprint
        image_resized = cv2.resize(image, (640, 480))  # Resize the image (you can adjust this size)
        
        # Convert image to RGB for the YOLO model (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Convert image to a tensor and move to the same device as the model
        img_tensor = torch.from_numpy(img_rgb).float().to(device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert to (C, H, W) and add batch dimension

        # Run the model to detect walls
        results = model(img_tensor)

        # Process the results
        detected_wall = False
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if "wall" in class_name.lower():
                    print("✅ Wall detected")
                    detected_wall = True
                    break

        return jsonify({"wallDetected": detected_wall})

    except Exception as e:
        print("⚠️ Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)