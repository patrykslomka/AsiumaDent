import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pycocotools.coco import COCO


class DentalCOCODataset(Dataset):
    def __init__(self, root_dir, subset="train", transform=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.images_dir = os.path.join(root_dir, subset, "images")

        # Load COCO annotations
        annotation_file = os.path.join(root_dir, subset, "annotations", f"{subset}_coco.json")
        self.coco = COCO(annotation_file)

        # Get all image IDs
        self.image_ids = list(self.coco.imgs.keys())

        # Get category information
        self.categories = self.coco.cats
        self.num_classes = len(self.categories)

        print(f"Loaded {subset} dataset with {len(self.image_ids)} images and {self.num_classes} categories")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='black')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Create multi-label target tensor (one-hot encoding)
        target = torch.zeros(self.num_classes)

        # Fill in labels for detected classes
        for ann in annotations:
            category_id = ann['category_id']
            # Map from category_id to consecutive index (0-30)
            # This assumes category_ids are 1-indexed (common in COCO)
            if category_id in self.categories:
                class_idx = self.categories[category_id]['id']
                if 0 <= class_idx < self.num_classes:
                    target[class_idx] = 1

        return {
            'image': image,
            'target': target,
            'image_id': img_id,
            'file_name': img_info['file_name']
        }


def get_coco_dataloaders(data_dir, batch_size=32, num_workers=4):
    """Create DataLoaders for COCO format dental dataset"""
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets
    train_dataset = DentalCOCODataset(
        root_dir=data_dir,
        subset="train",
        transform=train_transform
    )

    val_dataset = DentalCOCODataset(
        root_dir=data_dir,
        subset="valid",
        transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Get class names
    class_names = [train_dataset.categories[cat_id]['name'] for cat_id in sorted(train_dataset.categories.keys())]

    return train_loader, val_loader, class_names


def save_class_names(class_names, output_file):
    """Save class names to a JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(class_names, f, indent=2)