import os
import argparse
import subprocess
import time


def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True)

    for line in process.stdout:
        print(line.strip())

    process.wait()
    return process.returncode


def setup_environment():
    """Set up the environment on vast.ai instance"""
    # Install dependencies
    print("Installing dependencies...")
    run_command(
        "pip install torch==2.0.1 torchvision==0.15.2 timm==0.9.2 pycocotools pandas matplotlib tqdm opencv-python scikit-learn")

    # Create directories
    run_command("mkdir -p logs")


def main(args):
    start_time = time.time()

    # Set up environment
    if not args.skip_setup:
        setup_environment()

    # Preprocess data
    if not args.skip_preprocessing:
        print("\nPreprocessing training data...")
        run_command(
            f"python src/preprocess_coco.py --data_dir={args.data_dir} --output_dir={args.output_dir} --subset=train")

        print("\nPreprocessing validation data...")
        run_command(
            f"python src/preprocess_coco.py --data_dir={args.data_dir} --output_dir={args.output_dir} --subset=valid")

        if args.process_test:
            print("\nPreprocessing test data...")
            run_command(
                f"python src/preprocess_coco.py --data_dir={args.data_dir} --output_dir={args.output_dir} --subset=test")

    # Start training
    print("\nStarting training...")
    training_cmd = (
        f"python src/train_coco.py "
        f"--data_dir={args.output_dir} "
        f"--output_dir={args.model_dir} "
        f"--batch_size={args.batch_size} "
        f"--num_workers={args.workers} "
        f"--num_epochs={args.epochs} "
        f"--learning_rate={args.lr} "
        f"--weight_decay={args.weight_decay} "
        f"--save_freq={args.save_freq} "
        f"--seed={args.seed}"
    )

    if args.resume:
        training_cmd += f" --resume={args.resume}"

    # Run with logging
    log_file = os.path.join("logs", "training_log.txt")
    run_command(f"{training_cmd} 2>&1 | tee {log_file}")

    # Print total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup and train dental X-ray model on vast.ai")

    # Setup options
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip data preprocessing steps")
    parser.add_argument("--process-test", action="store_true", help="Also process test data")

    # Data paths
    parser.add_argument("--data-dir", type=str, default=".", help="Path to raw data directory with COCO format")
    parser.add_argument("--output-dir", type=str, default="processed_data", help="Output directory for processed data")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models and results")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--save-freq", type=int, default=5, help="Save checkpoint frequency (epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")

    args = parser.parse_args()
    main(args)
