"""
Data augmentation and transforms for remote sensing images.
"""

from typing import Tuple

from torchvision import transforms


def get_train_transforms(image_size: int = 84) -> transforms.Compose:
    """
    Get training augmentations for remote sensing images.
    
    Includes random cropping, flipping, rotation, and color jitter
    to improve model robustness.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size + 8, image_size + 8)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_transforms(image_size: int = 84) -> transforms.Compose:
    """
    Get test/evaluation transforms.
    
    Uses center crop without augmentation for consistent evaluation.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size + 8, image_size + 8)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_simple_transforms(image_size: int = 84) -> transforms.Compose:
    """
    Get simple transforms without augmentation.
    
    Useful for visualization or debugging.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def get_inverse_normalize() -> transforms.Normalize:
    """
    Get inverse normalization transform for visualization.
    
    Returns:
        Inverse normalize transform
    """
    return transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )


class MultiScaleTransform:
    """
    Multi-scale augmentation for remote sensing images.
    
    Applies transforms at multiple scales to capture
    objects at different resolutions.
    """
    
    def __init__(
        self,
        base_size: int = 84,
        scales: Tuple[float, ...] = (0.8, 1.0, 1.2)
    ):
        self.base_size = base_size
        self.scales = scales
        
        self.transforms = {}
        for scale in scales:
            size = int(base_size * scale)
            self.transforms[scale] = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.CenterCrop(base_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __call__(self, img):
        """Apply transform at a randomly selected scale."""
        import random
        scale = random.choice(self.scales)
        return self.transforms[scale](img)
