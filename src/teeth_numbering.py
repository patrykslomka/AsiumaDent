class TeethNumbering:
    """
    Handles dental numbering systems for teeth identification.
    """

    def __init__(self):
        # FDI World Dental Federation notation (ISO 3950)
        self.fdi_system = {
            # Upper right quadrant (1)
            'upper_right_third_molar': 18,
            'upper_right_second_molar': 17,
            'upper_right_first_molar': 16,
            'upper_right_second_premolar': 15,
            'upper_right_first_premolar': 14,
            'upper_right_canine': 13,
            'upper_right_lateral_incisor': 12,
            'upper_right_central_incisor': 11,

            # Upper left quadrant (2)
            'upper_left_central_incisor': 21,
            'upper_left_lateral_incisor': 22,
            'upper_left_canine': 23,
            'upper_left_first_premolar': 24,
            'upper_left_second_premolar': 25,
            'upper_left_first_molar': 26,
            'upper_left_second_molar': 27,
            'upper_left_third_molar': 28,

            # Lower left quadrant (3)
            'lower_left_third_molar': 38,
            'lower_left_second_molar': 37,
            'lower_left_first_molar': 36,
            'lower_left_second_premolar': 35,
            'lower_left_first_premolar': 34,
            'lower_left_canine': 33,
            'lower_left_lateral_incisor': 32,
            'lower_left_central_incisor': 31,

            # Lower right quadrant (4)
            'lower_right_central_incisor': 41,
            'lower_right_lateral_incisor': 42,
            'lower_right_canine': 43,
            'lower_right_first_premolar': 44,
            'lower_right_second_premolar': 45,
            'lower_right_first_molar': 46,
            'lower_right_second_molar': 47,
            'lower_right_third_molar': 48
        }

        # Regions of the mouth for approximating tooth numbers when exact location isn't known
        self.regions = {
            'upper_anterior': [13, 12, 11, 21, 22, 23],
            'upper_right_posterior': [18, 17, 16, 15, 14],
            'upper_left_posterior': [24, 25, 26, 27, 28],
            'lower_anterior': [43, 42, 41, 31, 32, 33],
            'lower_right_posterior': [48, 47, 46, 45, 44],
            'lower_left_posterior': [34, 35, 36, 37, 38]
        }

    def get_tooth_number(self, position, bbox=None, image_size=None):
        """
        Estimates the tooth number based on the position in the image.

        Args:
            position: A string like "upper_right", "lower_left", etc.
            bbox: The bounding box coordinates [x, y, width, height]
            image_size: Tuple of (width, height) of the original image

        Returns:
            A list of possible tooth numbers in FDI notation
        """
        # If we don't have precise information, return a list of possible numbers
        if position in self.regions:
            return self.regions[position]

        # If we have a bounding box and image size, we can estimate based on position
        if bbox and image_size:
            x, y, w, h = bbox
            img_w, img_h = image_size

            # Get center of the bbox
            center_x = x + (w / 2)
            center_y = y + (h / 2)

            # Determine if upper or lower (based on y-coordinate)
            upper = center_y < img_h / 2

            # Determine if left or right (based on x-coordinate)
            left = center_x > img_w / 2

            # Determine if anterior or posterior
            # Anterior teeth are typically in the center third of the image
            anterior = img_w / 3 < center_x < 2 * img_w / 3

            # Estimate general region
            if upper:
                if anterior:
                    return self.regions['upper_anterior']
                elif left:
                    return self.regions['upper_left_posterior']
                else:
                    return self.regions['upper_right_posterior']
            else:
                if anterior:
                    return self.regions['lower_anterior']
                elif left:
                    return self.regions['lower_left_posterior']
                else:
                    return self.regions['lower_right_posterior']

        # Default if we can't determine
        return []