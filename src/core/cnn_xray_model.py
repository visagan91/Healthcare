import io
import json
import os
from typing import Any, Dict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class XrayMultiLabelClassifier:
    def __init__(self, artifact_dir: str, model_filename: str = "cnn_xray_resnet18_multilabel.pt"):
        self.artifact_dir = artifact_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        config_path = os.path.join(artifact_dir, "image_config.json")
        label_map_path = os.path.join(artifact_dir, "finding_label_map.json")
        model_path = os.path.join(artifact_dir, model_filename)

        with open(config_path, "r") as f:
            self.config = json.load(f)

        with open(label_map_path, "r") as f:
            raw_label_map = json.load(f)

        self.label_map = {int(k): v for k, v in raw_label_map.items()}
        self.labels = [self.label_map[i] for i in range(len(self.label_map))]

        self.image_size = int(self.config["image_size"])
        self.input_mode = str(self.config["input_mode"])
        self.num_classes = int(self.config["num_classes"])
        self.threshold = float(self.config.get("threshold", 0.5))
        self.multilabel = bool(self.config.get("multilabel", True))
        self.normalize = bool(self.config.get("normalize", False))
        self.arch = str(self.config["arch"])

        self.transform = self._build_transform()
        self.model = self._build_model()

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self) -> nn.Module:
        if self.arch != "resnet18":
            raise ValueError(f"Unsupported architecture: {self.arch}")

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def _build_transform(self):
        tfms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]

        if self.normalize:
            tfms.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        return transforms.Compose(tfms)

    def _read_image(self, image_bytes: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(image_bytes))
        if self.input_mode.upper() == "RGB":
            image = image.convert("RGB")
        elif self.input_mode.upper() == "L":
            image = image.convert("L")
        else:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")
        return image

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        image = self._read_image(image_bytes)
        x = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(x)

        if self.multilabel:
            probs = torch.sigmoid(logits)[0].cpu().tolist()
            predicted_findings = [
                self.labels[i] for i, p in enumerate(probs) if p >= self.threshold
            ]
        else:
            probs_tensor = torch.softmax(logits, dim=1)[0]
            probs = probs_tensor.cpu().tolist()
            pred_idx = int(torch.argmax(probs_tensor).item())
            predicted_findings = [self.labels[pred_idx]]

        probabilities = {
            self.labels[i]: round(float(probs[i]), 4)
            for i in range(len(self.labels))
        }

        top_scores = sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "model_id": "xray_multilabel_resnet18_v1",
            "task_type": "image_multilabel_classification",
            "threshold": self.threshold,
            "predicted_findings": predicted_findings,
            "probabilities": probabilities,
            "top_scores": top_scores[:5],
        }