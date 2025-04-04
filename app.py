import os
import uuid
from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from dotenv import load_dotenv
from src.inference import load_model, predict_with_location
from src.dental_ontology import DentalOntology
from src.claude_integration import ClaudeAssistant
import os
import requests


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


# Example usage
def download_model(model_url, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Extract file ID from Google Drive share link
    file_id = model_url.split('/d/')[1].split('/view')[0]

    print(f"Downloading model from Google Drive with ID: {file_id}")
    try:
        download_file_from_google_drive(file_id, model_path)
        print("Model downloaded successfully")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


# In your main script
MODEL_URL = "https://drive.google.com/file/d/1pjDGKb2xleb0wKjVVFJiiNU3cOAIVp-O/view?usp=sharing"
MODEL_PATH = 'models/best_model.pth'

if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model on startup
MODEL_PATH = 'models/best_model.pth'
CLASS_NAMES_PATH = 'models/class_names.json'

print("Loading model...")
model, class_names = load_model(MODEL_PATH, CLASS_NAMES_PATH)
print(f"Model loaded successfully with {len(class_names)} classes")

# Initialize dental ontology
dental_ontology = DentalOntology()

# Initialize Claude assistant
api_key = os.environ.get("ANTHROPIC_API_KEY")
if api_key:
    print(f"API key loaded: {api_key[:5]}...")
else:
    print("No API key found. Claude integration will not be available.")
claude_assistant = ClaudeAssistant(api_key=api_key)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'xray' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['xray']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save file with unique name
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Make prediction with locations
    try:
        predictions, image = predict_with_location(model, file_path, class_names)

        # Save annotated image for display
        annotated_filename = f"annotated_{filename}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)

        # Draw bounding boxes on a copy of the image
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        # Define colors for different conditions
        condition_colors = {
            'Crown': (255, 0, 0),  # Red
            'Implant': (0, 255, 0),  # Green
            'Root Piece': (0, 0, 255),  # Blue
            'Filling': (255, 255, 0),  # Yellow
            'Periapical lesion': (255, 0, 255),  # Magenta
            'Retained root': (0, 255, 255),  # Cyan
            'maxillary sinus': (255, 165, 0),  # Orange
            'Malaligned': (128, 0, 128),  # Purple
        }

        # Draw precise bounding boxes
        for i, pred in enumerate(predictions):
            if 'bbox' in pred:
                condition = pred['condition']
                bbox = pred['bbox']
                color = condition_colors.get(condition, (255, 255, 255))

                # Draw rectangle
                draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                               outline=color, width=2)

                # Add number and label
                text = f"{i+1}. {condition}: {pred['probability']:.2f}"
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except IOError:
                    font = ImageFont.load_default()

                text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (
                len(text) * 7, 12)
                draw.rectangle([bbox[0], bbox[1] - text_height - 2, bbox[0] + text_width, bbox[1]],
                               fill=color)
                draw.text((bbox[0], bbox[1] - text_height - 2), text, fill="white", font=font)

        # Save the annotated image
        draw_image.save(annotated_path)

        # Get detailed information from the ontology for each prediction
        detailed_predictions = []
        for pred in predictions:
            condition = pred['condition']
            detailed_pred = {
                **pred,
                "details": dental_ontology.get_condition_info(condition)
            }
            detailed_predictions.append(detailed_pred)

        # Generate AI report
        ai_report = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            report_result = claude_assistant.generate_dental_report(predictions, dental_ontology)
            ai_report = report_result.get("report")

        return jsonify({
            'predictions': detailed_predictions,
            'original_image': f"/uploads/{filename}",
            'annotated_image': f"/uploads/{annotated_filename}",
            'ai_report': ai_report
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up original file if not needed
        if os.path.exists(file_path) and "filename" in locals():
            # Keep for debugging
            pass


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)