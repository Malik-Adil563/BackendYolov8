from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch

# Load the trained YOLOv8 model (ensure you're using the smallest version like yolov8n.pt)
model = YOLO("runs/detect/train3/weights/best.pt")  # Update if your path is different

# Move the model to GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return {"status": "Python backend running"}

@app.route('/detect-wall', methods=['POST'])
def detect_wall():
    print("📸 Request received for wall detection")
    
    try:
        if 'image' not in request.files:
            print("❌ No image in request")
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        
        # Open the image, convert to RGB, and resize it to reduce memory footprint
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((640, 480))
        img = np.array(image)

        # Convert image to tensor
        img_tensor = torch.from_numpy(img).float().to(device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        # Run detection with confidence threshold set to 0.017 (1.7%)
        results = model(img_tensor, conf=0.017)

        # Check for wall detection
        detected_wall = False
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if "wall" in class_name.lower():
                    print("✅ Wall detected with confidence")
                    detected_wall = True
                    break

        return jsonify({"wallDetected": detected_wall})

    except Exception as e:
        print("⚠️ Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)