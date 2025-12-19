"""
AMC-MetaNet: Adaptive Metric Classifier Meta-Network.

Main model combining backbone feature extraction with adaptive metric learning
for few-shot remote sensing scene classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .backbone import get_backbone
from .metric_module import AdaptiveMetricModule


class AMCMetaNet(nn.Module):
    """
    Adaptive Metric Classifier Meta-Network (AMC-MetaNet).
    
    A meta-learning model for few-shot remote sensing classification that:
    1. Extracts features using a ResNet-12 backbone
    2. Applies adaptive metric learning with attention
    3. Performs prototypical classification with learned metrics
    
    Architecture:
        Input -> Backbone -> Adaptive Metric Module -> Prototypical Classifier
    """
    
    def __init__(
        self,
        backbone: str = 'resnet12',
        feature_dim: int = 640,
        attention_dim: int = 128,
        temperature: float = 1.0,
        use_attention: bool = True,
        use_task_conditioning: bool = True,
        metric: str = 'euclidean',
        dropout: float = 0.5
    ):
        """
        Args:
            backbone: Backbone network name ('resnet12' or 'conv4')
            feature_dim: Feature embedding dimension
            attention_dim: Attention module hidden dimension
            temperature: Temperature for softmax scaling
            use_attention: Whether to use channel attention
            use_task_conditioning: Whether to use task conditioning
            metric: Distance metric type
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Backbone for feature extraction
        self.backbone = get_backbone(backbone, feature_dim=feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Adaptive metric module
        self.metric_module = AdaptiveMetricModule(
            feature_dim=feature_dim,
            attention_dim=attention_dim,
            temperature=temperature,
            use_attention=use_attention,
            use_task_conditioning=use_task_conditioning,
            metric=metric
        )
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            images: Input images (B, C, H, W)
            
        Returns:
            Feature embeddings (B, feature_dim)
        """
        features = self.backbone(images)
        features = self.dropout(features)
        return features
    
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
        n_shot: int
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for episodic training/evaluation.
        
        Args:
            support_images: Support set images (N*K, C, H, W)
            support_labels: Support set labels (N*K,)
            query_images: Query set images (Q, C, H, W)
            n_way: Number of classes (N)
            n_shot: Number of shots per class (K)
            
        Returns:
            Dictionary containing:
                - logits: Classification logits (Q, N)
                - prototypes: Class prototypes (N, D)
                - support_features: Support features (N*K, D)
                - query_features: Query features (Q, D)
        """
        # Extract features
        support_features = self.extract_features(support_images)
        query_features = self.extract_features(query_images)
        
        # Compute logits using adaptive metric module
        logits, prototypes = self.metric_module(
            support_features, support_labels,
            query_features, n_way
        )
        
        return {
            'logits': logits,
            'prototypes': prototypes,
            'support_features': support_features,
            'query_features': query_features
        }
    
    def forward_from_features(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        n_way: int
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using pre-extracted features.
        
        Useful for two-stage training or when features are cached.
        
        Args:
            support_features: Support features (N*K, D)
            support_labels: Support labels (N*K,)
            query_features: Query features (Q, D)
            n_way: Number of classes
            
        Returns:
            Dictionary with logits and prototypes
        """
        logits, prototypes = self.metric_module(
            support_features, support_labels,
            query_features, n_way
        )
        
        return {
            'logits': logits,
            'prototypes': prototypes
        }


class PrototypicalLoss(nn.Module):
    """
    Loss function for prototypical networks.
    
    Combines cross-entropy loss with optional regularization terms.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prototypical loss.
        
        Args:
            logits: Classification logits (Q, N)
            labels: Ground truth labels (Q,)
            
        Returns:
            Loss value
        """
        return F.cross_entropy(
            logits / self.temperature,
            labels,
            label_smoothing=self.label_smoothing
        )


class BalancedLoss(nn.Module):
    """
    Balanced loss for few-shot learning.
    
    Combines classification loss with:
    - Prototype diversity loss
    - Feature alignment loss
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        diversity_weight: float = 0.1,
        alignment_weight: float = 0.1
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.diversity_weight = diversity_weight
        self.alignment_weight = alignment_weight
    
    def prototype_diversity_loss(
        self,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage prototypes to be diverse/separated.
        
        Args:
            prototypes: Class prototypes (N, D)
            
        Returns:
            Diversity loss
        """
        # Pairwise distances between prototypes
        n_prototypes = prototypes.size(0)
        if n_prototypes < 2:
            return torch.tensor(0.0, device=prototypes.device)
        
        distances = torch.cdist(prototypes, prototypes, p=2)
        
        # Mask diagonal
        mask = torch.eye(n_prototypes, device=prototypes.device).bool()
        distances = distances.masked_fill(mask, float('inf'))
        
        # Minimize negative of minimum distance (maximize separation)
        min_distances = distances.min(dim=1)[0]
        loss = -min_distances.mean()
        
        return loss
    
    def feature_alignment_loss(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage features to be close to their prototypes.
        
        Args:
            support_features: Support features (N*K, D)
            support_labels: Support labels (N*K,)
            prototypes: Class prototypes (N, D)
            
        Returns:
            Alignment loss
        """
        # Gather corresponding prototypes
        target_prototypes = prototypes[support_labels]
        
        # L2 distance between features and prototypes
        distances = (support_features - target_prototypes).pow(2).sum(dim=1)
        
        return distances.mean()
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        prototypes: torch.Tensor,
        support_features: Optional[torch.Tensor] = None,
        support_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute balanced loss.
        
        Args:
            logits: Classification logits (Q, N)
            labels: Ground truth labels (Q,)
            prototypes: Class prototypes (N, D)
            support_features: Optional support features for alignment
            support_labels: Optional support labels for alignment
            
        Returns:
            Dictionary with total loss and individual components
        """
        # Classification loss
        cls_loss = F.cross_entropy(logits, labels)
        
        # Diversity loss
        div_loss = self.prototype_diversity_loss(prototypes)
        
        # Alignment loss
        if support_features is not None and support_labels is not None:
            align_loss = self.feature_alignment_loss(
                support_features, support_labels, prototypes
            )
        else:
            align_loss = torch.tensor(0.0, device=logits.device)
        
        # Total loss
        total_loss = (
            self.classification_weight * cls_loss +
            self.diversity_weight * div_loss +
            self.alignment_weight * align_loss
        )
        
        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'div_loss': div_loss,
            'align_loss': align_loss
        }
