# src/feedback_db.py
import os
import json
import sqlite3
import datetime


class FeedbackDatabase:
    """Enhanced feedback database with learning capabilities"""

    def __init__(self, db_path='feedback.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database with necessary tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create feedback table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL,
            original_predictions TEXT NOT NULL,
            corrected_predictions TEXT NOT NULL,
            dentist_id TEXT,
            timestamp TEXT NOT NULL
        )
        ''')

        # Create learning table to store corrected predictions for improving the model
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition TEXT NOT NULL,
            bbox TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_path TEXT NOT NULL,
            correction_count INTEGER NOT NULL,
            last_update TEXT NOT NULL
        )
        ''')

        conn.commit()
        conn.close()

    def save_feedback(self, image_id, original_predictions, corrected_predictions, dentist_id=None):
        """Save feedback and update learning data"""
        timestamp = datetime.datetime.now().isoformat()

        # Save the feedback record
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'INSERT INTO feedback (image_id, original_predictions, corrected_predictions, dentist_id, timestamp) VALUES (?, ?, ?, ?, ?)',
            (
                image_id,
                json.dumps(original_predictions),
                json.dumps(corrected_predictions),
                dentist_id,
                timestamp
            )
        )

        # Process corrections for learning
        self._process_corrections(cursor, image_id, original_predictions, corrected_predictions)

        conn.commit()
        conn.close()

    def _process_corrections(self, cursor, image_id, original_predictions, corrected_predictions):
        """Process corrections to improve the model over time"""

        # 1. Find removed predictions (predictions in original but not in corrected)
        original_dict = {self._get_pred_key(pred): pred for pred in original_predictions if pred is not None}
        corrected_dict = {self._get_pred_key(pred): pred for pred in corrected_predictions if pred is not None}

        # Find items in original but not in corrected - these were removed/rejected
        removed_keys = set(original_dict.keys()) - set(corrected_dict.keys())

        # Find items that were modified (same key but different condition)
        modified_keys = []
        for key in set(original_dict.keys()) & set(corrected_dict.keys()):
            if original_dict[key]['condition'] != corrected_dict[key]['condition']:
                modified_keys.append(key)

        # 2. Store learning data
        for key in removed_keys:
            pred = original_dict[key]
            self._update_learning(cursor,
                                  pred['condition'],
                                  pred.get('bbox', [0, 0, 0, 0]),
                                  pred.get('probability', 0),
                                  image_id,
                                  False  # This was rejected, so it's a negative example
                                  )

        # 3. Store modified predictions
        for key in modified_keys:
            # The original prediction was wrong
            orig_pred = original_dict[key]
            self._update_learning(cursor,
                                  orig_pred['condition'],
                                  orig_pred.get('bbox', [0, 0, 0, 0]),
                                  orig_pred.get('probability', 0),
                                  image_id,
                                  False  # This was corrected, so original was wrong
                                  )

            # The corrected prediction is right
            corr_pred = corrected_dict[key]
            self._update_learning(cursor,
                                  corr_pred['condition'],
                                  corr_pred.get('bbox', [0, 0, 0, 0]),
                                  corr_pred.get('probability', 0),
                                  image_id,
                                  True  # This is the correction, so it's positive
                                  )

        # 4. Store added predictions
        added_keys = set(corrected_dict.keys()) - set(original_dict.keys())
        for key in added_keys:
            pred = corrected_dict[key]
            self._update_learning(cursor,
                                  pred['condition'],
                                  pred.get('bbox', [0, 0, 0, 0]),
                                  pred.get('probability', 0),
                                  image_id,
                                  True  # This was added by dentist, so it's a positive example
                                  )

    def _update_learning(self, cursor, condition, bbox, confidence, image_path, is_positive):
        """Update learning data with a new example"""
        bbox_str = json.dumps(bbox)
        timestamp = datetime.datetime.now().isoformat()

        # Check if this example already exists
        cursor.execute(
            'SELECT id, correction_count FROM learning_data WHERE condition = ? AND bbox = ? AND image_path = ?',
            (condition, bbox_str, image_path)
        )
        result = cursor.fetchone()

        if result:
            # Update existing example
            example_id, count = result
            new_count = count + (1 if is_positive else -1)  # Increment if positive, decrement if negative

            cursor.execute(
                'UPDATE learning_data SET correction_count = ?, last_update = ? WHERE id = ?',
                (new_count, timestamp, example_id)
            )
        else:
            # Insert new example
            cursor.execute(
                'INSERT INTO learning_data (condition, bbox, confidence, image_path, correction_count, last_update) VALUES (?, ?, ?, ?, ?, ?)',
                (condition, bbox_str, confidence, image_path, 1 if is_positive else -1, timestamp)
            )

    def _get_pred_key(self, pred):
        """Generate a unique key for a prediction based on its bounding box"""
        if not pred or 'bbox' not in pred:
            return None

        bbox = pred['bbox']
        # Round to nearest 5 pixels to allow for small differences
        rounded_bbox = [round(coord / 5) * 5 for coord in bbox]
        return f"{rounded_bbox[0]}_{rounded_bbox[1]}_{rounded_bbox[2]}_{rounded_bbox[3]}"

    def get_all_feedback(self):
        """Get all feedback records"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM feedback ORDER BY timestamp DESC')
        rows = cursor.fetchall()

        feedback = []
        for row in rows:
            feedback.append({
                'id': row['id'],
                'image_id': row['image_id'],
                'original_predictions': json.loads(row['original_predictions']),
                'corrected_predictions': json.loads(row['corrected_predictions']),
                'dentist_id': row['dentist_id'],
                'timestamp': row['timestamp']
            })

        conn.close()
        return feedback

    def get_learning_data(self):
        """Get all learning data for model improvement"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM learning_data ORDER BY correction_count DESC')
        rows = cursor.fetchall()

        learning_data = []
        for row in rows:
            learning_data.append({
                'id': row['id'],
                'condition': row['condition'],
                'bbox': json.loads(row['bbox']),
                'confidence': row['confidence'],
                'image_path': row['image_path'],
                'correction_count': row['correction_count'],
                'last_update': row['last_update']
            })

        conn.close()
        return learning_data

    # Add to your feedback_db.py file

    def apply_learning_to_predictions(self, predictions, image_id=None, dentist_id=None):
        """
        Apply learning from previous feedback to adjust prediction confidences.

        Args:
            predictions: List of prediction dictionaries
            image_id: Optional image ID for context-specific adjustments
            dentist_id: Optional dentist ID for personalized adjustments

        Returns:
            Adjusted predictions with weighted confidence values
        """
        if not predictions:
            return predictions

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get learning data for conditions in the predictions
        conditions = set(pred['condition'] for pred in predictions)
        placeholders = ','.join(['?'] * len(conditions))

        # Prepare a dictionary to store the adjustments
        global_adjustments = {}  # Adjustments based on all feedback
        dentist_adjustments = {}  # Adjustments specific to this dentist

        # Get global adjustments (from all dentists)
        if conditions:
            cursor.execute(
                f"""
                SELECT condition, 
                       SUM(CASE WHEN correction_count > 0 THEN 1 ELSE 0 END) as confirmations,
                       SUM(CASE WHEN correction_count < 0 THEN 1 ELSE 0 END) as rejections,
                       COUNT(*) as total
                FROM learning_data 
                WHERE condition IN ({placeholders})
                GROUP BY condition
                """,
                tuple(conditions)
            )

            # Calculate adjustment factors
            for row in cursor.fetchall():
                condition, confirmations, rejections, total = row

                # Calculate confidence factor (-0.15 to +0.15 range)
                if total > 0:
                    # More confirmations than rejections = positive adjustment
                    # More rejections than confirmations = negative adjustment
                    confidence_factor = (confirmations - rejections) / total
                    adjustment = min(max(confidence_factor * 0.15, -0.15), 0.15)
                    global_adjustments[condition] = adjustment

        # Get dentist-specific adjustments if dentist_id provided
        if dentist_id:
            cursor.execute(
                f"""
                SELECT f.corrected_predictions, f.original_predictions
                FROM feedback f
                WHERE f.dentist_id = ?
                ORDER BY f.id DESC
                LIMIT 10
                """,
                (dentist_id,)
            )

            # Process this dentist's recent feedback
            dentist_feedback = cursor.fetchall()
            for corrected, original in dentist_feedback:
                try:
                    corr_preds = json.loads(corrected)
                    orig_preds = json.loads(original)

                    # Find changed predictions
                    for c_pred in corr_preds:
                        condition = c_pred.get('condition')
                        if not condition:
                            continue

                        # Check if this condition was modified
                        orig_match = next(
                            (o for o in orig_preds if self._get_pred_key(o) == self._get_pred_key(c_pred)), None)

                        if orig_match and orig_match.get('condition') != condition:
                            # This condition was changed by the dentist
                            # Decrease confidence in original condition
                            dentist_adjustments[orig_match['condition']] = dentist_adjustments.get(
                                orig_match['condition'], 0) - 0.05

                            # Increase confidence in corrected condition
                            dentist_adjustments[condition] = dentist_adjustments.get(condition, 0) + 0.1

                        elif not orig_match:
                            # This is a newly added condition
                            dentist_adjustments[condition] = dentist_adjustments.get(condition, 0) + 0.1
                except:
                    # Skip if there's an error parsing the feedback
                    pass

        # Apply adjustments to predictions
        for pred in predictions:
            condition = pred['condition']
            original_probability = pred['probability']
            adjustment = 0

            # Apply global adjustment (lower weight)
            if condition in global_adjustments:
                adjustment += global_adjustments[condition] * 0.7

            # Apply dentist-specific adjustment (higher weight)
            if condition in dentist_adjustments:
                adjustment += dentist_adjustments[condition]

            # Apply the total adjustment
            if adjustment != 0:
                # Ensure probability stays between 0.01 and 0.99
                adjusted_probability = min(max(original_probability + adjustment, 0.01), 0.99)

                # Store adjustment info
                pred['adjusted'] = True
                pred['confidence_adjustment'] = {
                    'original': original_probability,
                    'adjusted': adjusted_probability,
                    'adjustment': adjustment
                }

                # Update the probability
                pred['probability'] = adjusted_probability

        conn.close()
        return predictions