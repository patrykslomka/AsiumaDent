import os
import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import tempfile
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your existing modules
# These imports assume your original files are in the same directory
from model import DentalClassificationModel, DentalDetectionModel
from inference import load_model, predict_with_location
from dental_ontology import DentalOntology

# Initialize dental ontology
try:
    dental_ontology = DentalOntology()
    ontology_loaded = True
except Exception as e:
    print(f"Warning: Could not load dental ontology: {e}")
    ontology_loaded = False
    dental_ontology = None

# Initialize Claude assistant if API key is available
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
claude_available = ANTHROPIC_API_KEY is not None

if claude_available:
    from claude_integration import ClaudeAssistant

    claude_assistant = ClaudeAssistant(api_key=ANTHROPIC_API_KEY)
    print("Claude integration enabled")
else:
    print("Claude integration disabled. Set ANTHROPIC_API_KEY to enable.")

# Create directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load model
MODEL_PATH = 'models/best_model.pth'
CLASS_NAMES_PATH = 'models/class_names.json'

# Load class names directly if model isn't available yet
try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} class names")
except FileNotFoundError:
    print("Class names file not found. Using placeholder.")
    class_names = ["Caries", "Crown", "Filling", "Implant", "Malaligned"]

# Try to load model, but continue even if it fails (for UI development)
try:
    model, _ = load_model(MODEL_PATH, CLASS_NAMES_PATH)
    model_loaded = True
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("Continuing with UI only. Upload model files to enable analysis.")
    model_loaded = False

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


def analyze_xray(image):
    """Process the uploaded X-ray image and return analysis results"""
    if image is None:
        return None, "Please upload an image to analyze.", None, []

    if not model_loaded:
        return (
            image,
            "Model not loaded. Please upload model files to the 'models' directory.",
            None,
            []
        )

    # Save image to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_path = temp_file.name
        image.save(temp_path)

    try:
        # Make prediction with locations
        predictions, pil_image = predict_with_location(model, temp_path, class_names)

        # Draw bounding boxes on a copy of the image
        draw_image = pil_image.copy()
        draw = ImageDraw.Draw(draw_image)

        # Draw precise bounding boxes
        for i, pred in enumerate(predictions):
            if 'bbox' in pred:
                condition = pred['condition']
                bbox = pred['bbox']
                color = condition_colors.get(condition, (255, 255, 255))

                # Draw rectangle
                draw.rectangle(
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    outline=color, width=2
                )

                # Add number and label
                text = f"{i + 1}. {condition}: {pred['probability']:.2f}"
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except IOError:
                    font = ImageFont.load_default()

                # Handle different versions of PIL
                if hasattr(draw, 'textsize'):
                    text_width, text_height = draw.textsize(text, font=font)
                else:
                    text_width, text_height = (len(text) * 7, 12)

                draw.rectangle(
                    [bbox[0], bbox[1] - text_height - 2, bbox[0] + text_width, bbox[1]],
                    fill=color
                )
                draw.text((bbox[0], bbox[1] - text_height - 2), text, fill="white", font=font)

        # Get detailed information from the ontology for each prediction
        detailed_predictions = []
        if ontology_loaded:
            for pred in predictions:
                condition = pred['condition']
                detailed_pred = {
                    **pred,
                    "details": dental_ontology.get_condition_info(condition)
                }
                detailed_predictions.append(detailed_pred)
        else:
            detailed_predictions = predictions

        # Generate AI report if Claude is available
        ai_report = None
        if claude_available and len(predictions) > 0:
            try:
                report_result = claude_assistant.generate_dental_report(
                    predictions,
                    dental_ontology if ontology_loaded else {}
                )
                ai_report = report_result.get("report")
            except Exception as e:
                ai_report = f"Error generating report: {str(e)}"

        # Format predictions for display
        formatted_predictions = []
        for i, pred in enumerate(predictions):
            formatted_predictions.append(
                f"{i + 1}. {pred['condition']}: {pred['probability'] * 100:.1f}%"
            )

        # Clean up
        os.unlink(temp_path)

        return draw_image, ai_report, create_chart_data(predictions), formatted_predictions

    except Exception as e:
        import traceback
        error_msg = f"Error analyzing image: {str(e)}\n{traceback.format_exc()}"
        return image, error_msg, None, []
    finally:
        # Clean up temp file if it still exists
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def create_chart_data(predictions):
    """Create data for the bar chart visualization"""
    if not predictions:
        return None

    # Limit to top 10 predictions
    top_predictions = sorted(
        predictions,
        key=lambda x: x['probability'],
        reverse=True
    )[:10]

    # Create labels and values
    labels = [p['condition'] for p in top_predictions]
    values = [p['probability'] * 100 for p in top_predictions]

    return (labels, values)


# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Dental X-ray Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Dental X-ray Analysis Platform

            Upload a dental X-ray to detect and locate dental conditions using AI.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="Upload X-ray Image",
                    elem_id="xray-input"
                )
                analyze_button = gr.Button(
                    "Analyze X-ray",
                    variant="primary",
                    elem_id="analyze-button"
                )
                model_status = gr.Markdown(
                    f"**Model Status**: {'✅ Loaded' if model_loaded else '❌ Not Loaded'}"
                )

            with gr.Column(scale=2):
                with gr.Tab("Results"):
                    output_image = gr.Image(
                        type="pil",
                        label="Annotated X-ray",
                        elem_id="annotated-image"
                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            predictions_list = gr.Dataframe(
                                headers=["Detected Conditions"],
                                datatype=["str"],
                                label="Detected Conditions",
                                elem_id="predictions-list"
                            )

                        with gr.Column(scale=1):
                            chart = gr.BarPlot(
                                x="Condition",
                                y="Confidence (%)",
                                title="Confidence Scores",
                                tooltip=["Condition", "Confidence (%)"],
                                y_lim=[0, 100],
                                label="Prediction Confidence",
                                elem_id="prediction-chart"
                            )

                with gr.Tab("AI Report"):
                    ai_report = gr.Markdown(
                        label="AI Analysis Report",
                        elem_id="ai-report"
                    )

        analyze_button.click(
            analyze_xray,
            inputs=[input_image],
            outputs=[output_image, ai_report, chart, predictions_list]
        )

        gr.Markdown("""
        ## How to Use

        1. Upload a dental X-ray image using the upload box
        2. Click "Analyze X-ray" to process the image
        3. View the annotated image with detected conditions
        4. Check the "AI Report" tab for clinical insights

        ## About

        This application uses a deep learning model to detect various dental conditions in X-ray images.
        The model is based on EfficientNet architecture and trained on dental radiographs.
        """)

        if not model_loaded:
            gr.Markdown("""
            ## ⚠️ Model Not Loaded

            The analysis model is not currently loaded. To enable analysis:

            1. Upload the model file (`best_model.pth`) to the `models` directory
            2. Upload the class names file (`class_names.json`) to the `models` directory
            3. Restart the Space
            """)

        if not claude_available:
            gr.Markdown("""
            ## ℹ️ AI Report Disabled

            The AI report generation is disabled because the ANTHROPIC_API_KEY is not set.
            To enable AI reports, add your Claude API key as an environment variable.
            """)

    return demo


# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()