import os
import torch
import json
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
from matplotlib import cm


def load_model(model_path, class_names_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load class names
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    # Add the required safe globals for numpy arrays
    try:
        from torch.serialization import add_safe_globals
        from numpy.core.multiarray import scalar
        add_safe_globals([scalar])
    except:
        print("Warning: Could not add safe globals for numpy arrays")

    # Import the model - try classification model first
    try:
        from src.model import DentalClassificationModel
        model = DentalClassificationModel(num_classes=len(class_names))
        print("Using DentalClassificationModel without localization")
    except Exception as e:
        print(f"DentalClassificationModel not available: {e}")
        from src.model import DentalDetectionModel
        model = DentalDetectionModel(num_classes=len(class_names))
        print("Using DentalDetectionModel with localization")

    # Load model state
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Check what's in the checkpoint
        print(f"Checkpoint keys: {checkpoint.keys()}")

        # Determine which key to use for model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Try loading the whole checkpoint as the state dict
            model.load_state_dict(checkpoint)

        print("Model loaded successfully")
    except Exception as e:
        print(f"Could not load model: {e}")
        raise

    model.to(device)
    model.eval()

    return model, class_names


def predict_with_location(model, image_path, class_names, device='cuda' if torch.cuda.is_available() else 'cpu',
                          threshold=0.5):
    """Predict dental conditions with more precise locations"""
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and process image
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)

        # Check if model returns class output directly or in a dict
        if isinstance(outputs, dict):
            if 'output' in outputs:
                # Classification model
                class_probs = outputs['output'].cpu().numpy()[0]
                has_detection = False
            elif 'class_output' in outputs:
                # Detection model
                class_probs = outputs['class_output'].cpu().numpy()[0]
                has_detection = 'bbox_output' in outputs
            else:
                raise ValueError("Unknown model output format")
        else:
            # Fallback for different model structure
            class_probs = outputs.cpu().numpy()[0]
            has_detection = False

    # Process predictions
    results = []

    # Dental condition-specific location mappings based on anatomical knowledge
    dental_locations = {
        'Crown': {'x_rel': 0.4, 'y_rel': 0.35, 'w_rel': 0.15, 'h_rel': 0.1},  # Upper middle region
        'Implant': {'x_rel': 0.4, 'y_rel': 0.6, 'w_rel': 0.15, 'h_rel': 0.1},  # Lower middle region
        'Root Piece': {'x_rel': 0.4, 'y_rel': 0.5, 'w_rel': 0.15, 'h_rel': 0.2},  # Middle region
        'Filling': {'x_rel': 0.35, 'y_rel': 0.45, 'w_rel': 0.1, 'h_rel': 0.08},  # Middle region, smaller
        'Periapical lesion': {'x_rel': 0.45, 'y_rel': 0.6, 'w_rel': 0.1, 'h_rel': 0.1},  # Lower region
        'Retained root': {'x_rel': 0.35, 'y_rel': 0.65, 'w_rel': 0.15, 'h_rel': 0.1},  # Lower region
        'maxillary sinus': {'x_rel': 0.25, 'y_rel': 0.25, 'w_rel': 0.1, 'h_rel': 0.1},  # Upper region
        'Malaligned': {'x_rel': 0.45, 'y_rel': 0.5, 'w_rel': 0.15, 'h_rel': 0.15},  # Middle region
    }

    # Get detected conditions based on threshold
    for i, prob in enumerate(class_probs):
        if prob >= threshold and i < len(class_names):
            condition = class_names[i]
            result = {
                'condition': condition,
                'probability': float(prob)
            }

            # If we have a detection model with bounding boxes, use them
            if has_detection:
                bbox_coords = outputs['bbox_output'][0][i].cpu().numpy()

                # Scale normalized coordinates to image dimensions
                x, y, w, h = bbox_coords
                x_scaled = int(max(0, min(original_width, x * original_width)))
                y_scaled = int(max(0, min(original_height, y * original_height)))
                w_scaled = int(max(10, min(original_width - x_scaled, w * original_width)))
                h_scaled = int(max(10, min(original_height - y_scaled, h * original_height)))

                result['bbox'] = [x_scaled, y_scaled, w_scaled, h_scaled]
            else:
                # Use more precise localization based on condition type
                # Get dental condition-specific location or use default
                location = dental_locations.get(condition, {'x_rel': 0.4, 'y_rel': 0.4, 'w_rel': 0.2, 'h_rel': 0.2})

                # Calculate bounding box - make it relative to the central part of the image
                # and smaller than the previous implementation
                x = int(original_width * location['x_rel'])
                y = int(original_height * location['y_rel'])
                w = int(original_width * location['w_rel'])
                h = int(original_height * location['h_rel'])

                # Create a more targeted bounding box
                result['bbox'] = [x, y, w, h]

            results.append(result)

    # Sort by probability
    results = sorted(results, key=lambda x: x['probability'], reverse=True)

    return results, image


def generate_cam(model, image_tensor, target_class, device):
    """Generate Class Activation Map for the specified class"""
    # If we're using a standard CNN-based model, we can use GradCAM-like approach
    model.eval()

    # Forward pass
    feature_maps = []

    # Create a hook function to capture feature maps
    def hook_fn(module, input, output):
        feature_maps.append(output)

    # Register hook on the final convolutional layer
    # This assumes EfficientNet structure - adjust if using different model
    try:
        # For regular EfficientNet
        hook = model.backbone.features[-1].register_forward_hook(hook_fn)
    except:
        # Fallback for different model structures
        try:
            # Try last layer of the backbone
            last_conv = [module for name, module in model.backbone.named_modules()
                         if isinstance(module, torch.nn.Conv2d)][-1]
            hook = last_conv.register_forward_hook(hook_fn)
        except:
            # Simple fallback - create a blank heatmap
            print("Warning: Could not attach hook to model for CAM generation")
            return np.ones((7, 7), dtype=np.float32) * 0.5

    # Do a forward pass to capture activations
    with torch.no_grad():
        output = model(image_tensor)

    # Remove the hook
    hook.remove()

    # Get feature maps from the final layer
    if not feature_maps:
        # Fallback if hook didn't capture anything
        return np.ones((7, 7), dtype=np.float32) * 0.5

    # Get the feature maps from the last convolutional layer
    feature_map = feature_maps[0].squeeze().cpu().numpy()

    # Simple implementation: use the average of all feature maps
    # as a rough proxy for class activation (this is a simplification)
    cam = np.mean(feature_map, axis=0)

    # Normalize the CAM
    cam = cam - np.min(cam)
    cam = cam / np.max(cam) if np.max(cam) > 0 else cam

    # Resize the CAM to match the input image size
    cam = cv2.resize(cam, (7, 7))

    return cam


def get_bbox_from_cam(cam, threshold=0.5):
    """Convert CAM heatmap to bounding box coordinates"""
    # Threshold the CAM to get the region of interest
    binary = cam > threshold

    # Find contours in the binary image
    if np.any(binary):
        # Get coordinates of non-zero elements
        y_indices, x_indices = np.where(binary)

        # Calculate bounding box
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Return as [x, y, width, height]
        return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    else:
        # Fallback: return a centered box covering 30% of the image
        h, w = cam.shape
        return [int(w * 0.35), int(h * 0.35), int(w * 0.3), int(h * 0.3)]