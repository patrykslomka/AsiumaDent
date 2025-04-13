# src/export_feedback.py
import json
import argparse
from feedback_db import FeedbackDatabase


def export_feedback_for_training(output_file):
    """Export feedback data in a format suitable for model retraining"""
    db = FeedbackDatabase()
    feedback_data = db.get_all_feedback()

    # Process feedback data into training format
    training_data = []

    for entry in feedback_data:
        # Skip entries with no corrections
        if not entry['corrected_predictions']:
            continue

        # Create training sample with corrections
        training_sample = {
            'image_id': entry['image_id'],
            'predictions': entry['corrected_predictions']
        }

        training_data.append(training_sample)

    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"Exported {len(training_data)} training samples to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export feedback data for model retraining")
    parser.add_argument("--output", type=str, default="feedback_training_data.json",
                        help="Output file for training data")

    args = parser.parse_args()
    export_feedback_for_training(args.output)