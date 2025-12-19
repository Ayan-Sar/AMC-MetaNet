"""
Episodic batch sampler for few-shot learning.
Creates N-way K-shot episodes for meta-training.
"""

import random
from typing import List, Iterator

import torch
from torch.utils.data import Sampler


class EpisodicBatchSampler(Sampler):
    """
    Samples batches in the form of episodes for meta-learning.
    
    Each episode consists of:
    - N classes (ways)
    - K support samples per class (shots)
    - Q query samples per class
    """
    
    def __init__(
        self,
        labels: List[int],
        n_way: int,
        n_shot: int,
        n_query: int,
        n_episodes: int
    ):
        """
        Args:
            labels: List of labels for all samples in the dataset
            n_way: Number of classes per episode
            n_shot: Number of support samples per class
            n_query: Number of query samples per class
            n_episodes: Number of episodes per epoch
        """
        self.labels = labels
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # Build index mapping: class -> list of sample indices
        self.class_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.classes = list(self.class_indices.keys())
        
        # Validate
        if len(self.classes) < n_way:
            raise ValueError(
                f"Dataset has {len(self.classes)} classes, "
                f"but n_way is {n_way}"
            )
        
        for cls, indices in self.class_indices.items():
            if len(indices) < n_shot + n_query:
                raise ValueError(
                    f"Class {cls} has only {len(indices)} samples, "
                    f"but need {n_shot + n_query}"
                )
    
    def __len__(self) -> int:
        return self.n_episodes
    
    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_episodes):
            # Sample n_way classes
            episode_classes = random.sample(self.classes, self.n_way)
            
            batch_indices = []
            
            for cls in episode_classes:
                # Sample support + query indices for this class
                cls_indices = self.class_indices[cls].copy()
                random.shuffle(cls_indices)
                
                # First n_shot are support, next n_query are query
                selected = cls_indices[:self.n_shot + self.n_query]
                batch_indices.extend(selected)
            
            yield batch_indices


class PrototypicalBatchSampler(Sampler):
    """
    Batch sampler for Prototypical Networks.
    
    Ensures that each batch contains enough samples per class
    to form support and query sets.
    """
    
    def __init__(
        self,
        labels: List[int],
        classes_per_batch: int,
        samples_per_class: int,
        n_batches: int
    ):
        """
        Args:
            labels: List of labels for all samples
            classes_per_batch: Number of classes per batch (N-way)
            samples_per_class: Total samples per class (support + query)
            n_batches: Number of batches (episodes) per epoch
        """
        self.labels = labels
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.n_batches = n_batches
        
        # Build class to indices mapping
        self.class_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.classes = list(self.class_indices.keys())
    
    def __len__(self) -> int:
        return self.n_batches
    
    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_batches):
            batch = []
            
            # Sample classes for this batch
            selected_classes = random.sample(
                self.classes, self.classes_per_batch
            )
            
            for cls in selected_classes:
                # Sample indices for this class
                cls_indices = self.class_indices[cls]
                if len(cls_indices) >= self.samples_per_class:
                    selected = random.sample(
                        cls_indices, self.samples_per_class
                    )
                else:
                    # Sample with replacement if not enough samples
                    selected = random.choices(
                        cls_indices, k=self.samples_per_class
                    )
                batch.extend(selected)
            
            yield batch


def collate_episode(batch: List) -> dict:
    """
    Collate function for episodic training.
    
    Reorganizes batch into support and query sets.
    
    Args:
        batch: List of (image, label) tuples
        
    Returns:
        Dictionary with support and query tensors
    """
    # Determine n_way and samples per class from batch structure
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    
    return {
        'images': images,
        'labels': labels
    }
