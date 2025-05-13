import os
import torch
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from PIL import Image
from tqdm import tqdm

# === Set up feature extractor ===
class Resnet50WithFPN(torch.nn.Module):
    def __init__(self):
        super(Resnet50WithFPN, self).__init__()
        m = resnet50(pretrained=True)
        self.body = create_feature_extractor(
            m, return_nodes={
                'layer1': 'feat1',
                'layer2': 'feat2',
                'layer3': 'feat3',
                'layer4': 'feat4',
            })
        # Infer channel sizes
        dummy = torch.randn(1, 3, 224, 224)
        out = self.body(dummy)
        in_channels_list = [o.shape[1] for o in out.values()]
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()
        )

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Initialize model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Resnet50WithFPN().to(device).eval()

# === Dataset path ===
dataset_root = "dataset"  # Change if needed

features = {}

for split in ['train', 'val']:
    split_path = os.path.join(dataset_root, split)
    features[split] = {}

    for class_folder in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        features[split][class_folder] = []
        print(f"Processing: {split}/{class_folder}")

        for img_name in tqdm(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Failed to open {img_path}: {e}")
                continue

            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model(input_tensor)
                feat_tensor = torch.cat([v.mean([2, 3]) for v in feat.values()], dim=1)
                features[split][class_folder].append(feat_tensor.cpu())

# Save all features to file
torch.save(features, "resnet50_features.pt")
print("✅ Feature extraction complete. Saved to resnet50_features.pt")