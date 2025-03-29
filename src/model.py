import torch
import torch.nn as nn
import timm


class DentalClassificationModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DentalClassificationModel, self).__init__()

        # Load pre-trained EfficientNet-B0
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained)

        # Get the number of features from the last layer
        num_features = self.backbone.classifier.in_features

        # Remove the final classification layer
        self.backbone.classifier = nn.Identity()

        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classification head for multi-label classification
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)

        # Refine features
        refined = self.feature_refine(features)

        # Apply classifier
        logits = self.classifier(refined)

        # Apply sigmoid for multi-label classification
        output = torch.sigmoid(logits)

        return {
            'logits': logits,
            'output': output
        }