from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")  # adjust if needed

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
        image = Image.open(file.stream).convert('RGB')
        img = np.array(image)

        # Run detection
        results = model(img)
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
