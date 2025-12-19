"""
Dataset classes for few-shot remote sensing scene classification.
Supports NWPU-RESISC45, UCMerced, AID, and WHU-RS19 datasets.
"""

import os
import random
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class RSDataset(Dataset):
    """
    Remote Sensing Dataset for few-shot learning.
    
    Organizes images by class for episodic training.
    Expected directory structure:
        root/
            class1/
                image1.jpg
                image2.jpg
                ...
            class2/
                ...
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        seed: int = 42
    ):
        """
        Args:
            root: Root directory of the dataset
            split: One of 'train', 'val', or 'test'
            transform: Image transformations
            train_ratio: Ratio of classes for training
            val_ratio: Ratio of classes for validation
            seed: Random seed for reproducibility
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        # Get all classes (subdirectories)
        all_classes = sorted([
            d for d in os.listdir(root) 
            if os.path.isdir(os.path.join(root, d))
        ])
        
        # Split classes into train/val/test
        random.seed(seed)
        shuffled_classes = all_classes.copy()
        random.shuffle(shuffled_classes)
        
        n_train = int(len(all_classes) * train_ratio)
        n_val = int(len(all_classes) * val_ratio)
        
        if split == 'train':
            self.classes = shuffled_classes[:n_train]
        elif split == 'val':
            self.classes = shuffled_classes[n_train:n_train + n_val]
        else:  # test
            self.classes = shuffled_classes[n_train + n_val:]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build image list per class
        self.class_images: Dict[int, List[str]] = {}
        self.all_images: List[Tuple[str, int]] = []
        
        for cls in self.classes:
            cls_idx = self.class_to_idx[cls]
            cls_dir = os.path.join(root, cls)
            images = [
                os.path.join(cls_dir, img) 
                for img in os.listdir(cls_dir)
                if img.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
            ]
            self.class_images[cls_idx] = images
            self.all_images.extend([(img, cls_idx) for img in images])
    
    def __len__(self) -> int:
        return len(self.all_images)
    
    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.all_images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_images(self, class_idx: int) -> List[str]:
        """Get all image paths for a specific class."""
        return self.class_images[class_idx]
    
    def sample_episode(
        self,
        n_way: int,
        n_shot: int,
        n_query: int
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Sample a single episode (task) for meta-learning.
        
        Args:
            n_way: Number of classes per episode
            n_shot: Number of support samples per class
            n_query: Number of query samples per class
            
        Returns:
            support_set: List of (image_path, label) tuples
            query_set: List of (image_path, label) tuples
        """
        # Sample n_way classes
        episode_classes = random.sample(list(self.class_images.keys()), n_way)
        
        support_set = []
        query_set = []
        
        for new_label, cls_idx in enumerate(episode_classes):
            # Get all images for this class
            cls_images = self.class_images[cls_idx].copy()
            random.shuffle(cls_images)
            
            # Split into support and query
            support_images = cls_images[:n_shot]
            query_images = cls_images[n_shot:n_shot + n_query]
            
            support_set.extend([(img, new_label) for img in support_images])
            query_set.extend([(img, new_label) for img in query_images])
        
        return support_set, query_set


class EpisodeDataset(Dataset):
    """
    Dataset that yields episodes for meta-learning.
    Each item is a complete episode with support and query sets.
    """
    
    def __init__(
        self,
        base_dataset: RSDataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_episodes: int,
        transform: Optional[Callable] = None
    ):
        self.base_dataset = base_dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.transform = transform
    
    def __len__(self) -> int:
        return self.n_episodes
    
    def __getitem__(self, idx: int) -> Dict:
        support_set, query_set = self.base_dataset.sample_episode(
            self.n_way, self.n_shot, self.n_query
        )
        
        # Load and transform images
        support_images = []
        support_labels = []
        for img_path, label in support_set:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            support_images.append(img)
            support_labels.append(label)
        
        query_images = []
        query_labels = []
        for img_path, label in query_set:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            query_images.append(img)
            query_labels.append(label)
        
        import torch
        return {
            'support_images': torch.stack(support_images),
            'support_labels': torch.tensor(support_labels),
            'query_images': torch.stack(query_images),
            'query_labels': torch.tensor(query_labels)
        }


# Dataset configurations
DATASET_CONFIGS = {
    'NWPU-RESISC45': {
        'num_classes': 45,
        'images_per_class': 700,
        'image_size': 256
    },
    'UCMerced': {
        'num_classes': 21,
        'images_per_class': 100,
        'image_size': 256
    },
    'AID': {
        'num_classes': 30,
        'images_per_class': 200,  # varies 220-420
        'image_size': 600
    },
    'WHU-RS19': {
        'num_classes': 19,
        'images_per_class': 50,
        'image_size': 600
    }
}


def get_dataset(
    name: str,
    root: str,
    split: str,
    transform: Optional[Callable] = None,
    **kwargs
) -> RSDataset:
    """
    Factory function to create dataset instances.
    
    Args:
        name: Dataset name (NWPU-RESISC45, UCMerced, AID, WHU-RS19)
        root: Root directory containing the dataset
        split: Data split ('train', 'val', 'test')
        transform: Image transformations
        
    Returns:
        RSDataset instance
    """
    dataset_path = os.path.join(root, name)
    
    if not os.path.exists(dataset_path):
        raise ValueError(
            f"Dataset {name} not found at {dataset_path}. "
            f"Please download and extract the dataset first."
        )
    
    return RSDataset(
        root=dataset_path,
        split=split,
        transform=transform,
        **kwargs
    )
