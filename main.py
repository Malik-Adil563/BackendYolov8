from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import os

app = Flask(__name__)
CORS(app)

# Load YOLOv8 model
yolo_model = YOLO("runs/detect/train6/weights/best.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model.to(device)

# Load ResNet50 (feature extractor)
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()

# Load saved wall features from .pt
feature_file_path = os.path.join("dataset", "resnet50_features.pt")
saved_features = torch.load(feature_file_path, map_location=device)["features"]  # shape: [N, 2048]

# Image preprocessing for ResNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route("/detect-wall", methods=["POST"])
def detect_wall():
    print("📸 Request received for wall detection")

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        img_array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image_resized = cv2.resize(image, (640, 480))
        img_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Prepare image for YOLO
        img_tensor = torch.from_numpy(img_rgb).float().to(device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        # Run YOLO detection
        results = yolo_model(img_tensor)
        yolo_detected_wall = False

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = yolo_model.names[class_id]
                if "wall" in class_name.lower():
                    print("🟡 YOLO says: wall detected")
                    yolo_detected_wall = True
                    break

        if not yolo_detected_wall:
            print("❌ YOLO didn't detect wall — rejecting")
            return jsonify({"wallDetected": False})

        # If YOLO detected wall, validate with ResNet features
        with torch.no_grad():
            input_tensor = transform(img_rgb).unsqueeze(0).to(device)
            current_feat = resnet(input_tensor)  # shape: [1, 2048]

            # Compare with saved wall features
            sims = F.cosine_similarity(current_feat, saved_features)
            max_sim = torch.max(sims).item()
            print(f"🧠 ResNet cosine similarity: {max_sim:.3f}")

            # Final decision based on threshold
            if max_sim >= 0.8:
                print("✅ Confirmed: Wall detected by both YOLO and ResNet")
                return jsonify({"wallDetected": True})
            else:
                print("❌ ResNet disagrees — rejecting")
                return jsonify({"wallDetected": False})

    except Exception as e:
        print("⚠️ Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)