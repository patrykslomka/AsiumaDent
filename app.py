import os
import uuid
import json
from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from dotenv import load_dotenv
from src.inference import load_model, predict_with_location
from src.dental_ontology import DentalOntology
from src.claude_integration import ClaudeAssistant
from src.feedback_db import FeedbackDatabase
from src.teeth_numbering import TeethNumbering  # New import

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the project root
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pth')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'models', 'class_names.json')

# Debug: Check if the file exists
print(f"Model path: {MODEL_PATH}")
print(f"Does model exist? {os.path.exists(MODEL_PATH)}")

# Load the model
model, class_names = load_model(MODEL_PATH, CLASS_NAMES_PATH)

# Initialize dental ontology
dental_ontology = DentalOntology()

# Initialize teeth numbering system
teeth_numbering = TeethNumbering()

# Initialize Claude assistant
api_key = os.environ.get("ANTHROPIC_API_KEY")
if api_key:
    print(f"API key loaded: {api_key[:5]}...")
else:
    print("No API key found. Claude integration will not be available.")
claude_assistant = ClaudeAssistant(api_key=api_key)

# Initialize feedback database
feedback_db = FeedbackDatabase()


@app.route('/')
def index():
    return render_template('index.html', class_names=class_names)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Helper function for drawing annotations
def draw_annotation_with_teeth_numbers(draw, pred, index, color, font=None):
    """
    Draw bounding box, condition label, and teeth numbers
    """
    # Extract information
    bbox = pred['bbox']
    condition = pred['condition']
    probability = pred.get('probability', 0)
    teeth_numbers = pred.get('teeth_numbers', [])

    # Draw rectangle
    draw.rectangle([
        bbox[0], bbox[1],
        bbox[0] + bbox[2], bbox[1] + bbox[3]
    ], outline=color, width=2)

    # Add number and label
    text = f"{index + 1}. {condition}: {probability:.2f}"

    if not font:
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

    text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (
        len(text) * 7, 12)

    # Draw label background
    draw.rectangle([
        bbox[0], bbox[1] - text_height - 2,
                 bbox[0] + text_width, bbox[1]
    ], fill=color)

    # Draw label text
    draw.text((bbox[0], bbox[1] - text_height - 2), text, fill="white", font=font)

    # Draw teeth numbers if available
    if teeth_numbers:
        # Choose up to 3 most likely teeth numbers to display
        display_numbers = teeth_numbers[:3]
        teeth_text = f"#{', #'.join(str(n) for n in display_numbers)}"

        teeth_text_width, teeth_text_height = draw.textsize(teeth_text, font=font) if hasattr(draw, 'textsize') else (
            len(teeth_text) * 7, 12)

        # Draw teeth numbers below the bounding box
        y_pos = bbox[1] + bbox[3] + 2  # Just below the bounding box

        # Draw background for teeth numbers
        draw.rectangle([
            bbox[0], y_pos,
            bbox[0] + teeth_text_width, y_pos + teeth_text_height
        ], fill=(0, 0, 0, 128))  # Semi-transparent black background

        # Draw teeth numbers text
        draw.text((bbox[0], y_pos), teeth_text, fill=(255, 255, 255), font=font)


# Helper function to add teeth numbers to predictions
def add_teeth_numbers_to_predictions(predictions, image_size):
    """Add teeth numbers to predictions based on their position in the image"""
    for pred in predictions:
        if 'bbox' in pred:
            bbox = pred['bbox']
            # Estimate teeth numbers based on position
            teeth_nums = teeth_numbering.get_tooth_number(
                position=None,  # We don't have this info yet
                bbox=bbox,
                image_size=image_size
            )
            # Add to prediction
            pred['teeth_numbers'] = teeth_nums
    return predictions


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

        # Sort predictions by probability to ensure consistent numbering
        predictions = sorted(predictions, key=lambda x: x.get('probability', 0), reverse=True)

        # Get image size for teeth numbering
        image_size = (image.width, image.height)

        # Add teeth numbers to predictions
        predictions = add_teeth_numbers_to_predictions(predictions, image_size)

        # Apply feedback learning to adjust confidences based on past corrections
        predictions = feedback_db.apply_learning_to_predictions(predictions, filename)

        # Cache predictions for later filtering
        predictions_cache_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}.json")
        with open(predictions_cache_path, 'w') as f:
            json.dump(predictions, f)

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

        # Draw precise bounding boxes with teeth numbers
        for i, pred in enumerate(predictions):
            if 'bbox' in pred:
                condition = pred['condition']
                color = condition_colors.get(condition, (255, 255, 255))
                draw_annotation_with_teeth_numbers(draw, pred, i, color)

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
            'ai_report': ai_report,
            'image_id': filename
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


@app.route('/filter-image', methods=['POST'])
def filter_image():
    """
    Create a filtered version of the annotated image based on confidence threshold
    """
    data = request.json

    if not data or 'image_id' not in data or 'threshold' not in data:
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        # Get parameters
        image_id = data['image_id']
        threshold = float(data['threshold']) / 100.0  # Convert percentage to decimal

        # Paths to original image
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], image_id)

        # Check if original image exists
        if not os.path.exists(original_path):
            return jsonify({'error': f'Original image not found: {image_id}'}), 404

        # Path for filtered image
        filtered_filename = f"filtered_{threshold:.2f}_{image_id}"
        filtered_path = os.path.join(app.config['UPLOAD_FOLDER'], filtered_filename)

        # Open original image
        image = Image.open(original_path)

        # Create a clean copy for drawing
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        # Load cached predictions for this image (stored in a JSON file)
        predictions_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{image_id}.json")

        # If predictions aren't cached, run the model again
        if not os.path.exists(predictions_path):
            # Run model and save predictions to JSON
            predictions, _ = predict_with_location(model, original_path, class_names)

            # Save predictions to JSON
            with open(predictions_path, 'w') as f:
                json.dump(predictions, f)
        else:
            # Load cached predictions
            with open(predictions_path, 'r') as f:
                predictions = json.load(f)

        # Filter predictions based on threshold
        filtered_predictions = [p for p in predictions if p.get('probability', 0) >= threshold]

        # Get image size for teeth numbering
        image_size = (image.width, image.height)

        # Add teeth numbers to predictions if not already present
        if filtered_predictions and 'teeth_numbers' not in filtered_predictions[0]:
            filtered_predictions = add_teeth_numbers_to_predictions(filtered_predictions, image_size)

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

        # Sort filtered predictions by probability for consistent numbering
        filtered_predictions = sorted(filtered_predictions, key=lambda x: x.get('probability', 0), reverse=True)

        # Draw bounding boxes for filtered predictions
        for i, pred in enumerate(filtered_predictions):
            if 'bbox' in pred:
                condition = pred['condition']
                color = condition_colors.get(condition, (255, 255, 255))
                draw_annotation_with_teeth_numbers(draw, pred, i, color)

        # Save the filtered image
        draw_image.save(filtered_path)

        # Return the URL of the filtered image
        return jsonify({
            'filtered_image': f"/uploads/{filtered_filename}",
            'count': len(filtered_predictions)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/clean-view', methods=['POST'])
def clean_view():
    """
    Create a clean version of the X-ray with only teeth numbers
    """
    data = request.json

    if not data or 'image_id' not in data:
        return jsonify({'error': 'Missing image_id'}), 400

    try:
        # Get image ID
        image_id = data['image_id']

        # Original image path
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], image_id)

        # Check if original image exists
        if not os.path.exists(original_path):
            return jsonify({'error': f'Original image not found: {image_id}'}), 404

        # Path for clean view image
        clean_filename = f"clean_{image_id}"
        clean_path = os.path.join(app.config['UPLOAD_FOLDER'], clean_filename)

        # Open original image
        image = Image.open(original_path)

        # Create a clean copy
        clean_image = image.copy()
        draw = ImageDraw.Draw(clean_image)

        # Get image size
        image_size = (image.width, image.height)

        # Draw only tooth numbers - no deficiency annotations
        try:
            font = ImageFont.truetype("arial.ttf", 10)  # Smaller font for teeth numbers
        except IOError:
            font = ImageFont.load_default()

        # Define positions for all teeth in a standard dental panoramic X-ray
        teeth_positions = {
            # Upper teeth numbering (18-11, 21-28)
            18: (int(image.width * 0.15), int(image.height * 0.3)),
            17: (int(image.width * 0.2), int(image.height * 0.3)),
            16: (int(image.width * 0.25), int(image.height * 0.3)),
            15: (int(image.width * 0.29), int(image.height * 0.3)),
            14: (int(image.width * 0.33), int(image.height * 0.3)),
            13: (int(image.width * 0.37), int(image.height * 0.3)),
            12: (int(image.width * 0.41), int(image.height * 0.3)),
            11: (int(image.width * 0.45), int(image.height * 0.3)),

            21: (int(image.width * 0.55), int(image.height * 0.3)),
            22: (int(image.width * 0.59), int(image.height * 0.3)),
            23: (int(image.width * 0.63), int(image.height * 0.3)),
            24: (int(image.width * 0.67), int(image.height * 0.3)),
            25: (int(image.width * 0.71), int(image.height * 0.3)),
            26: (int(image.width * 0.75), int(image.height * 0.3)),
            27: (int(image.width * 0.8), int(image.height * 0.3)),
            28: (int(image.width * 0.85), int(image.height * 0.3)),

            # Lower teeth numbering (48-41, 31-38)
            48: (int(image.width * 0.15), int(image.height * 0.7)),
            47: (int(image.width * 0.2), int(image.height * 0.7)),
            46: (int(image.width * 0.25), int(image.height * 0.7)),
            45: (int(image.width * 0.29), int(image.height * 0.7)),
            44: (int(image.width * 0.33), int(image.height * 0.7)),
            43: (int(image.width * 0.37), int(image.height * 0.7)),
            42: (int(image.width * 0.41), int(image.height * 0.7)),
            41: (int(image.width * 0.45), int(image.height * 0.7)),

            31: (int(image.width * 0.55), int(image.height * 0.7)),
            32: (int(image.width * 0.59), int(image.height * 0.7)),
            33: (int(image.width * 0.63), int(image.height * 0.7)),
            34: (int(image.width * 0.67), int(image.height * 0.7)),
            35: (int(image.width * 0.71), int(image.height * 0.7)),
            36: (int(image.width * 0.75), int(image.height * 0.7)),
            37: (int(image.width * 0.8), int(image.height * 0.7)),
            38: (int(image.width * 0.85), int(image.height * 0.7)),
        }

        # Draw each tooth number
        for tooth_num, position in teeth_positions.items():
            # Create background for better visibility
            text = str(tooth_num)
            text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (
                len(text) * 8, 10)

            # Draw background box
            draw.rectangle([
                position[0] - 2,
                position[1] - 2,
                position[0] + text_width + 2,
                position[1] + text_height + 2
            ], fill=(0, 0, 0, 180))  # Black with transparency

            # Draw text
            draw.text(position, text, fill=(255, 255, 255), font=font)

        # Save the clean image
        clean_image.save(clean_path)

        # Return the URL of the clean image
        return jsonify({
            'clean_image': f"/uploads/{clean_filename}",
            'message': 'Clean view with teeth numbers created successfully'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        # Print raw data for debugging
        raw_data = request.get_data().decode('utf-8')
        print("Raw feedback data:", raw_data)

        # Parse JSON data
        data = request.json

        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        print("Parsed data:", data)

        # Check required fields
        if 'image_id' not in data or 'corrected_predictions' not in data:
            return jsonify({'error': 'Missing required fields'}), 400

        # Sanitize and validate the data
        image_id = str(data['image_id'])

        # Ensure corrected_predictions is a list
        if not isinstance(data['corrected_predictions'], list):
            return jsonify({'error': 'corrected_predictions must be a list'}), 400

        # Clean up any null values
        corrected_predictions = [p for p in data['corrected_predictions'] if p is not None]

        # Validate each prediction
        for i, pred in enumerate(corrected_predictions):
            if not isinstance(pred, dict):
                return jsonify({'error': f'Prediction {i} is not a valid object'}), 400

            # Ensure required fields exist
            if 'condition' not in pred:
                return jsonify({'error': f'Prediction {i} missing condition field'}), 400

        # Get original predictions (if they exist)
        original_predictions = data.get('original_predictions', [])
        if not isinstance(original_predictions, list):
            original_predictions = []

        # Get dentist ID
        dentist_id = str(data.get('dentist_id', 'anonymous'))

        # Save to database
        feedback_db.save_feedback(
            image_id=image_id,
            original_predictions=original_predictions,
            corrected_predictions=corrected_predictions,
            dentist_id=dentist_id
        )

        return jsonify({'status': 'success', 'message': 'Feedback saved successfully'})
    except Exception as e:
        import traceback
        print("Error in submit_feedback:", str(e))
        traceback.print_exc()
        return jsonify({'error': f'Error processing feedback: {str(e)}'}), 500

@app.route('/admin/feedback')
def admin_feedback():
    # In a real app, you'd add authentication here
    feedback_data = feedback_db.get_all_feedback()
    return render_template('admin_feedback.html', feedback=feedback_data)


@app.route('/admin/learning-data')
def admin_learning_data():
    """View the learning data collected from dentist feedback"""
    learning_data = feedback_db.get_learning_data()
    return render_template('admin_learning_data.html', learning_data=learning_data)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)