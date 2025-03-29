import os
import cv2
import json
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def preprocess_image(src_path, dst_path, target_size=(224, 224)):
    """Resize a single image to the target size for EfficientNet"""
    try:
        # Read image
        image = cv2.imread(src_path)
        if image is None:
            print(f"Failed to read image: {src_path}")
            return False

        # Resize image
        resized = cv2.resize(image, target_size)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Save the resized image
        cv2.imwrite(dst_path, resized)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def preprocess_coco_dataset(data_dir, output_dir, subset, target_size=(224, 224)):
    """Process images in COCO format dataset"""
    # Source directories
    images_dir = os.path.join(data_dir, "COCO", subset)
    annotations_file = os.path.join(data_dir, "COCO", "annotations", f"{subset}_coco.json")

    # Output directories
    output_images_dir = os.path.join(output_dir, subset, "images")
    os.makedirs(output_images_dir, exist_ok=True)

    # Output annotations directory
    output_anno_dir = os.path.join(output_dir, subset, "annotations")
    os.makedirs(output_anno_dir, exist_ok=True)

    # Copy and adapt annotations file
    print(f"Processing annotations from {annotations_file}")
    if os.path.exists(annotations_file):
        # Read original annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        # Copy the annotations file
        output_anno_file = os.path.join(output_anno_dir, f"{subset}_coco.json")
        shutil.copy(annotations_file, output_anno_file)
        print(f"Copied annotations to {output_anno_file}")
    else:
        print(f"Warning: Annotations file {annotations_file} not found")
        annotations = None

    # Process images
    if os.path.exists(images_dir):
        # Get all image files
        image_files = []
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))

        print(f"Found {len(image_files)} images in {images_dir}")

        # Process images in parallel
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
            futures = []
            for img_path in image_files:
                # Get relative path from images_dir
                rel_path = os.path.relpath(img_path, images_dir)
                dst_path = os.path.join(output_images_dir, rel_path)

                futures.append(executor.submit(
                    preprocess_image, img_path, dst_path, target_size))

            # Monitor progress
            processed = 0
            failed = 0
            for future in tqdm(futures, desc=f"Processing {subset} images"):
                if future.result():
                    processed += 1
                else:
                    failed += 1

        print(f"Processed {processed} images, failed {failed}")
        return processed, failed
    else:
        print(f"Warning: Images directory {images_dir} not found")
        return 0, 0


def main():
    parser = argparse.ArgumentParser(description="Preprocess COCO format dental X-ray dataset")
    parser.add_argument("--data_dir", type=str, default=".", help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Output directory for processed data")
    parser.add_argument("--target_size", type=int, nargs=2, default=[224, 224],
                        help="Target image size (width, height)")
    parser.add_argument("--subset", type=str, default="train", choices=["train", "valid", "test"],
                        help="Dataset subset to process")

    args = parser.parse_args()

    # Process the specified subset
    preprocess_coco_dataset(
        args.data_dir,
        args.output_dir,
        args.subset,
        tuple(args.target_size)
    )


if __name__ == "__main__":
    main()