import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import json
import argparse
import time
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

from model import DentalClassificationModel
from coco_dataset import get_coco_dataloaders, save_class_names


def train_model(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "metrics"), exist_ok=True)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, class_names = get_coco_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Save class names for inference
    save_class_names(class_names, os.path.join(args.output_dir, "class_names.json"))

    # Define model
    print("Initializing model...")
    model = DentalClassificationModel(
        num_classes=len(class_names),
        pretrained=True
    )
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Define scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
            best_val_loss = float('inf')
    else:
        best_val_loss = float('inf')

    # Training metrics
    train_losses = []
    val_losses = []
    learning_rates = []

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0

        train_progress = tqdm(train_loader, desc="Training")
        for batch in train_progress:
            # Get data
            images = batch['image'].to(device)
            targets = batch['target'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs['logits'], targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights
            optimizer.step()

            # Update statistics
            train_loss += loss.item()

            # Update progress bar
            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})

        # Calculate average loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        print(f"Train Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0

        # For computing metrics
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="Validation")
            for batch in val_progress:
                # Get data
                images = batch['image'].to(device)
                targets = batch['target'].to(device)

                # Forward pass
                outputs = model(images)

                # Calculate loss
                loss = criterion(outputs['logits'], targets)

                # Update statistics
                val_loss += loss.item()

                # Store predictions and targets for metrics
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs['output'].cpu().numpy())

                # Update progress bar
                val_progress.set_postfix({"loss": f"{loss.item():.4f}"})

        # Calculate average loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Concatenate all targets and outputs
        all_targets = np.vstack(all_targets)
        all_outputs = np.vstack(all_outputs)

        # Calculate average precision for each class
        ap_scores = []
        for i in range(len(class_names)):
            try:
                ap = average_precision_score(all_targets[:, i], all_outputs[:, i])
                ap_scores.append(ap)
            except Exception as e:
                print(f"Error calculating AP for class {i}: {e}")
                ap_scores.append(0)

        mean_ap = np.mean(ap_scores)

        print(f"Val Loss: {val_loss:.4f}, mAP: {mean_ap:.4f}")

        # Top classes by AP
        top_indices = np.argsort(ap_scores)[-5:]  # Top 5 classes
        print("Top 5 classes by AP:")
        for idx in reversed(top_indices):
            print(f"  {class_names[idx]}: {ap_scores[idx]:.4f}")

        # Save metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_ap": float(mean_ap),
            "class_ap": {class_names[i]: float(ap_scores[i]) for i in range(len(class_names))},
            "learning_rate": optimizer.param_groups[0]['lr'],
            "time": time.time() - start_time
        }

        # Save metrics to JSON
        with open(os.path.join(args.output_dir, "metrics", f"epoch_{epoch + 1}.json"), 'w') as f:
            json.dump(epoch_metrics, f, indent=2)

        # Update scheduler
        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'mean_ap': mean_ap
            }, save_path)
            print(f"New best model saved to {save_path}")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.num_epochs:
            save_path = os.path.join(args.output_dir, "checkpoints", f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'mean_ap': mean_ap
            }, save_path)
            print(f"Checkpoint saved to {save_path}")

        # Plot and save learning curves
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.num_epochs:
            plot_learning_curves(train_losses, val_losses, learning_rates,
                                 os.path.join(args.output_dir, "learning_curves.png"))

    print("Training complete!")


def plot_learning_curves(train_losses, val_losses, learning_rates, save_path):
    """Plot and save learning curves"""
    plt.figure(figsize=(12, 10))

    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)

    # Plot learning rate
    plt.subplot(2, 1, 2)
    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dental X-ray classification model on COCO format data")
    parser.add_argument("--data_dir", type=str, default="processed_data",
                        help="Path to processed data directory with COCO format")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save models and results")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    parser.add_argument("--save_freq", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Train the model
    train_model(args)