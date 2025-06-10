import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def detransform(
    img: torch.Tensor,
    mean: list[int] = [0.485, 0.456, 0.406],
    std: list[int] = [0.229, 0.224, 0.225],
) -> torch.Tensor:
    """
    Reverse the transformations applied to the image.
    Args:
        img: Transformed image tensor (C, H, W)
    Returns:
        detransformed image tensor (C, H, W) in uint8 format
    """
    # Denormalize
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img_denorm = img * std + mean

    # Scale back to [0, 255] and convert to uint8
    img_scaled = (img_denorm * 255.0).clamp(0, 255)
    img_uint8 = img_scaled.to(torch.uint8)

    return img_uint8


def visualize_sample(dataset, idx) -> None:
    """
    Visualizes a sample from the KeypointDataset.

    Args:
        dataset: Initialized KeypointDataset instance
        idx: Index of the sample to visualize
    """
    # Fetch sample
    img, heatmaps, _ = dataset[idx]

    # Prepare image: (C, H, W) -> (H, W, C)
    img = detransform(img)
    image_np = img.cpu().numpy().transpose(1, 2, 0)

    # Combine heatmaps (max across keypoints)
    combined_heatmap = torch.max(heatmaps, dim=0)[0]

    # Resize heatmap to match image dimensions
    combined_heatmap = combined_heatmap.unsqueeze(0).unsqueeze(
        0
    )  # Add batch and channel dims
    heatmap_resized = (
        F.interpolate(
            combined_heatmap,
            size=image_np.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )  # Remove extra dims

    # Create plot
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original Image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(heatmap_resized, cmap="hot")
    axes[1].set_title("Keypoint Heatmap")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(image_np)
    overlay = axes[2].imshow(heatmap_resized, cmap="jet", alpha=0.5)
    axes[2].set_title("Heatmap Overlay")
    axes[2].axis("off")

    plt.colorbar(overlay, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
