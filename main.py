from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import os
from datetime import datetime

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train3/weights/best.pt")  # Update path if needed

# Move the model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Flask app setup
app = Flask(__name__)
CORS(app)

# Directory to save uploaded images
SAVE_DIR = "saved_uploads"
os.makedirs(SAVE_DIR, exist_ok=True)

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

        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scene_{timestamp}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        file.save(filepath)
        print(f"📥 Image saved to: {filepath}")

        # Open the image and convert to RGB
        image = Image.open(filepath).convert('RGB')
        img = np.array(image)

        # Convert image to tensor and normalize to [0,1]
        img_tensor = torch.from_numpy(img).float().to(device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor / 255.0  # Normalize

        # Run YOLO detection with confidence threshold
        results = model(img_tensor, conf=0.95)

        # Check if "wall" was detected
        detected_wall = False
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if "wall" in class_name.lower():
                    print("✅ Wall detected with confidence ≥ 0.95")
                    detected_wall = True
                    break
            if detected_wall:
                break

        return jsonify({
            "wallDetected": detected_wall,
            "imageDownloadUrl": f"/downloads/{filename}"
        })

    except Exception as e:
        print("⚠️ Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/downloads/<path:filename>', methods=['GET'])
def download_image(filename):
    return send_from_directory(SAVE_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
