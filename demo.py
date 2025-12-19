"""
Demo script for AMC-MetaNet.

Demonstrates few-shot classification on sample images.

Usage:
    python demo.py --checkpoint checkpoints/best_model.pth --support_dir ./demo/support --query_dir ./demo/query
"""

import argparse
import os
from typing import List, Tuple

import torch
import matplotlib.pyplot as plt
from PIL import Image

from models import AMCMetaNet
from data import get_test_transforms
from utils import get_device, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AMC-MetaNet Demo')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--support_dir', type=str, required=True,
                        help='Directory with support images (subdirs = classes)')
    parser.add_argument('--query_dir', type=str, required=True,
                        help='Directory with query images to classify')
    parser.add_argument('--image_size', type=int, default=84,
                        help='Input image size')
    parser.add_argument('--output', type=str, default='demo_results.png',
                        help='Output visualization path')
    
    return parser.parse_args()


def load_support_set(
    support_dir: str,
    transform,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Load support set from directory.
    
    Expected structure:
        support_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                ...
    
    Returns:
        Tuple of (images, labels, class_names)
    """
    images = []
    labels = []
    class_names = []
    
    # Get class directories
    class_dirs = sorted([
        d for d in os.listdir(support_dir)
        if os.path.isdir(os.path.join(support_dir, d))
    ])
    
    for cls_idx, cls_name in enumerate(class_dirs):
        class_names.append(cls_name)
        cls_dir = os.path.join(support_dir, cls_name)
        
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                img_path = os.path.join(cls_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                labels.append(cls_idx)
    
    images = torch.stack(images).to(device)
    labels = torch.tensor(labels).to(device)
    
    return images, labels, class_names


def load_query_images(
    query_dir: str,
    transform,
    device: torch.device
) -> Tuple[torch.Tensor, List[str]]:
    """
    Load query images from directory.
    
    Returns:
        Tuple of (images, image_names)
    """
    images = []
    names = []
    
    for img_name in sorted(os.listdir(query_dir)):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
            img_path = os.path.join(query_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
            names.append(img_name)
    
    images = torch.stack(images).to(device)
    
    return images, names


def visualize_results(
    query_images: torch.Tensor,
    query_names: List[str],
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    class_names: List[str],
    output_path: str
):
    """Visualize classification results."""
    n_queries = len(query_names)
    cols = min(4, n_queries)
    rows = (n_queries + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1) if n_queries > 1 else [[axes]]
    
    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for idx in range(n_queries):
        row, col = idx // cols, idx % cols
        ax = axes[row][col]
        
        # Denormalize and convert to numpy
        img = query_images[idx].cpu()
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = img.clip(0, 1)
        
        ax.imshow(img)
        
        pred_class = class_names[predictions[idx].item()]
        conf = confidences[idx].item() * 100
        
        ax.set_title(f'{pred_class}\n({conf:.1f}%)', fontsize=12)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_queries, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to: {output_path}")


def main():
    """Main demo function."""
    args = parse_args()
    
    set_seed(42)
    device = get_device()
    
    # Load checkpoint
    print(f"Loading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create transform
    transform = get_test_transforms(args.image_size)
    
    # Load support set
    print(f"Loading support set from: {args.support_dir}")
    support_images, support_labels, class_names = load_support_set(
        args.support_dir, transform, device
    )
    n_way = len(class_names)
    n_shot = support_images.size(0) // n_way
    
    print(f"  Classes: {class_names}")
    print(f"  {n_way}-way {n_shot}-shot")
    
    # Load query images
    print(f"Loading query images from: {args.query_dir}")
    query_images, query_names = load_query_images(
        args.query_dir, transform, device
    )
    print(f"  {len(query_names)} query images")
    
    # Create model
    model = AMCMetaNet(
        backbone=config.get('model', {}).get('backbone', 'resnet12'),
        feature_dim=config.get('model', {}).get('feature_dim', 640),
        attention_dim=config.get('metric', {}).get('attention_dim', 128),
        temperature=config.get('metric', {}).get('temperature', 1.0),
        use_attention=True,
        use_task_conditioning=True,
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(
            support_images, support_labels,
            query_images, n_way, n_shot
        )
        
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        confidences = probabilities.max(dim=1)[0]
    
    # Print results
    print("\nClassification Results:")
    print("-" * 50)
    for i, name in enumerate(query_names):
        pred_class = class_names[predictions[i].item()]
        conf = confidences[i].item() * 100
        print(f"  {name}: {pred_class} ({conf:.1f}%)")
    
    # Visualize
    visualize_results(
        query_images, query_names,
        predictions, confidences,
        class_names, args.output
    )


if __name__ == '__main__':
    main()
