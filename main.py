from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import torch
import os
from datetime import datetime

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train3/weights/best.pt")

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Flask app
app = Flask(__name__)
CORS(app)

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

        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scene_{timestamp}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        file.save(filepath)
        print(f"📥 Image saved to: {filepath}")

        # Run YOLOv8 detection (let it handle preprocessing)
        results = model(filepath, conf=0.95)

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
