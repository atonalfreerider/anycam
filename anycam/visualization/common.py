import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_image_grid(
    images, rows, cols, directions=None, imsize=(2, 2), title=None, show=True
):
    fig, axs = plt.subplots(
        rows,
        cols,
        gridspec_kw={"wspace": 0, "hspace": 0},
        squeeze=True,
        figsize=(rows * imsize[0], cols * imsize[1]),
    )
    for i, image in enumerate(images):
        axs[i % rows][i // rows].axis("off")
        if directions is not None:
            axs[i % rows][i // rows].arrow(
                32,
                32,
                directions[i][0] * 16,
                directions[i][1] * 16,
                color="red",
                length_includes_head=True,
                head_width=2.0,
                head_length=1.0,
            )
        axs[i % rows][i // rows].imshow(image, aspect="auto")
    plt.subplots_adjust(hspace=0, wspace=0)
    if title is not None:
        fig.suptitle(title, fontsize=12)
    if show:
        plt.show()
    return fig


def show_save(save_path, show=True, save=False):
    if show:
        plt.show()
    if save:
        plt.savefig(save_path)


def color_tensor(tensor, cmap="viridis", norm=True, vmin=None, vmax=None):
    """
    Color a tensor using a matplotlib colormap.

    Args:
        tensor: Input tensor of shape (H, W) or (1, H, W) or (B, H, W)
        cmap: Colormap name (string) or matplotlib colormap object
        norm: Whether to normalize the tensor to [0, 1] range
        vmin: Minimum value for normalization (if None, uses tensor min)
        vmax: Maximum value for normalization (if None, uses tensor max)

    Returns:
        List of colored tensors with shape (3, H, W) in RGB format
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()

    # Handle different input shapes
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    elif tensor.dim() == 4:
        tensor = tensor.squeeze(1)  # Remove channel dimension if present

    batch_size = tensor.shape[0]

    # Get colormap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    colored_tensors = []

    for i in range(batch_size):
        img = tensor[i].numpy()

        # Normalize if requested
        if norm:
            if vmin is None:
                vmin_val = img.min()
            else:
                vmin_val = vmin

            if vmax is None:
                vmax_val = img.max()
            else:
                vmax_val = vmax

            if vmax_val > vmin_val:
                img = (img - vmin_val) / (vmax_val - vmin_val)
            else:
                img = np.zeros_like(img)

        # Apply colormap
        colored_img = cmap(img)  # Returns RGBA
        colored_img = colored_img[..., :3]  # Take only RGB

        # Convert to tensor and reorder to (C, H, W)
        colored_tensor = torch.from_numpy(colored_img).permute(2, 0, 1).float()
        colored_tensors.append(colored_tensor)

    return colored_tensors


def apply_colormap(depth, colormap="viridis", min_depth=None, max_depth=None):
    """
    Apply a colormap to depth values.

    Args:
        depth: Depth tensor of shape (H, W) or (B, H, W)
        colormap: Name of matplotlib colormap
        min_depth: Minimum depth for normalization
        max_depth: Maximum depth for normalization

    Returns:
        Colored depth tensor(s) with shape (3, H, W) or (B, 3, H, W)
    """
    return color_tensor(depth, cmap=colormap, norm=True, vmin=min_depth, vmax=max_depth)


def normalize_depth_for_visualization(depth, percentile=95):
    """
    Normalize depth for visualization by clipping outliers.

    Args:
        depth: Input depth tensor
        percentile: Percentile for clipping (e.g., 95 means clip top 5% of values)

    Returns:
        Normalized depth tensor
    """
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu()

    # Flatten for percentile calculation
    depth_flat = depth.flatten()

    # Remove invalid values (inf, nan, negative)
    valid_mask = torch.isfinite(depth_flat) & (depth_flat > 0)
    valid_depths = depth_flat[valid_mask]

    if len(valid_depths) == 0:
        return torch.zeros_like(depth)

    # Calculate percentile bounds
    min_depth = torch.quantile(valid_depths, 0.05)
    max_depth = torch.quantile(valid_depths, percentile / 100.0)

    # Clip and normalize
    depth_clipped = torch.clamp(depth, min_depth, max_depth)
    depth_normalized = (depth_clipped - min_depth) / (max_depth - min_depth + 1e-8)

    return depth_normalized
