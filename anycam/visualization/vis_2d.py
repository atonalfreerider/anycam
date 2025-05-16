import logging
import math
from typing import Any, Callable

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torchvision.utils import make_grid
from torchvision.utils import flow_to_image


from anycam.visualization.common import color_tensor

import numpy as np
import cv2

# TODO: configure logger somewhere else
logger = logging.getLogger("Visualization")



def get_input_imgs(data) -> torch.Tensor | None:
    if "imgs" in data and type(data["imgs"]) == list:
        return torch.stack(data["imgs"], dim=1).detach()[0] * 0.5 + 0.5
    elif "imgs" in data:
        return data["imgs"].detach()[0]
    logger.warning(
        "No images found in model output. Not creating a input image visualization."
    )
    return None




def get_depth(data) -> torch.Tensor | None:
    if "pred_depths" in data:
        depth = data["pred_depths"].detach()[0]
        z_near = data["z_near"]
        z_far = data["z_far"]

        depth = (1 / depth - 1 / z_far) / (1 / z_near - 1 / z_far)

        return color_tensor(depth.squeeze(1).clamp(0, 1), cmap="plasma").permute(
            0, 3, 1, 2
        )

    logger.warning(
        "No reconstructed depth found in model output. Not creating a depth visualization."
    )
    return None


def get_uncertainty(data) -> torch.Tensor | None:
    if "uncertainties" in data:
        uncert = data["uncertainties"][0, :, 0, :, :].detach()

        return color_tensor(uncert, cmap="plasma", norm=True).permute(0, 3, 1, 2)

    logger.warning(
        "No uncertainty found in model output. Not creating a uncertainty visualization."
    )
    return None


def get_rendered_flow(data) -> torch.Tensor | None:
    if "induced_flow" in data:
        flow = data["induced_flow"].detach()[0]

        h, w = flow.shape[-2:]
        nv = flow.shape[0]

        flow = flow.permute(0, 2, 3, 1).reshape(nv, h, w, 2).to(torch.float32)

        flow = torch.cat((flow[:, :, :, 0:1] / 2 * w , flow[:, :, :, 1:2] / 2 * h), dim=-1).permute(0, 3, 1, 2)

        flow_imgs = []
        for i in range(nv):
            flow_imgs.append(flow_to_image(flow[i].cpu().squeeze().clamp(-1000, 1000)).float() / 255)

        flow_imgs = torch.stack(flow_imgs, dim=0)
        return flow_imgs
    
    logger.warning(
        "No rendered flows found in model output. Not creating a rendered_flow visualization."
    )
    return None


def get_gt_flow(data) -> torch.Tensor | None:
    if "images_ip" in data:
        flow = data["images_ip"].detach()[0][:, 3:5]

        h, w = flow.shape[-2:]
        nv = flow.shape[0]

        flow = flow.permute(0, 2, 3, 1).reshape(nv, h, w, 2)

        flow = torch.cat((flow[:, :, :, 0:1] / 2 * w , flow[:, :, :, 1:2] / 2 * h), dim=-1).permute(0, 3, 1, 2)

        flow_imgs = []
        for i in range(nv):
            flow_imgs.append(flow_to_image(flow[i].cpu().squeeze()).float() / 255)

        flow_imgs = torch.stack(flow_imgs, dim=0)
        return flow_imgs
        
    logger.warning(
        "No gt flows found in model output. Not creating a rendered_flow visualization."
    )
    return None


def tb_visualize(model, dataset, config: dict[str, Any] | None = None):
    if config is None:
        vis_fns: dict[str, Callable[[Any], torch.Tensor | None]] = {
            "input_imgs": get_input_imgs,
            "depth": get_depth,
            "rendered_flow": get_rendered_flow,
            "gt_flow": get_gt_flow,
            "uncertainty": get_uncertainty,
        }
    else:
        # TODO: inform user about not found functions
        vis_fns = {
            name: globals()[f"get_{name}"]
            for name, _ in config.items()
            if [globals().get(f"get_{name}", None)]
        }

    def _visualize(engine: Engine, tb_logger: TensorboardLogger, step: int, tag: str):
        data = engine.state.output["output"]

        writer = tb_logger.writer
        for name, vis_fn in vis_fns.items():
            output = vis_fn(data)
            if output is not None:
                if name == "profiles":
                    grid = make_grid(output)
                else:
                    grid = make_grid(output, nrow=int(math.sqrt(output.shape[0])))
                writer.add_image(f"{tag}/{name}", grid.cpu(), global_step=step)

    return _visualize


def plot_optical_flow(flow, image=None, step=10, scale=1.0, color=(0, 255, 0)):
    """
    Visualize optical flow as arrows on an image.
    
    Args:
        flow: Optical flow tensor of shape (2, H, W) or (H, W, 2)
        image: Optional background image
        step: Step size for arrow grid
        scale: Scale factor for arrow length
        color: Arrow color in BGR format
    
    Returns:
        Visualization image as numpy array
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    
    # Handle different flow formats
    if flow.shape[0] == 2:  # (2, H, W)
        flow = flow.transpose(1, 2, 0)  # (H, W, 2)
    
    h, w = flow.shape[:2]
    
    # Create background image
    if image is None:
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        vis_img = image.copy()
    
    # Draw flow arrows
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            if np.sqrt(fx*fx + fy*fy) > 0.5:  # Only draw significant flow
                end_x = int(x + fx * scale)
                end_y = int(y + fy * scale)
                cv2.arrowedLine(vis_img, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
    
    return vis_img


def plot_keypoints(image, keypoints, radius=3, color=(255, 0, 0)):
    """
    Plot keypoints on an image.
    
    Args:
        image: Background image
        keypoints: List of (x, y) keypoint coordinates
        radius: Circle radius for keypoints
        color: Keypoint color in BGR format
    
    Returns:
        Image with keypoints drawn
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    vis_img = image.copy()
    
    for (x, y) in keypoints:
        cv2.circle(vis_img, (int(x), int(y)), radius, color, -1)
    
    return vis_img


def create_depth_heatmap(depth, colormap='plasma'):
    """
    Create a heatmap visualization of depth values.
    
    Args:
        depth: Depth tensor of shape (H, W)
        colormap: Matplotlib colormap name
    
    Returns:
        Colored depth heatmap as numpy array
    """
    from .common import color_tensor
    
    colored_depth = color_tensor(depth, cmap=colormap, norm=True)[0]
    colored_depth = colored_depth.permute(1, 2, 0).numpy()
    colored_depth = (colored_depth * 255).astype(np.uint8)
    
    return colored_depth


def plot_trajectory_2d(trajectory, image_shape, color=(0, 255, 0), thickness=2):
    """
    Plot camera trajectory on a 2D plane (top-down view).
    
    Args:
        trajectory: List of 4x4 pose matrices
        image_shape: (height, width) of output image
        color: Line color in BGR format
        thickness: Line thickness
    
    Returns:
        2D trajectory visualization
    """
    h, w = image_shape
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    if len(trajectory) < 2:
        return vis_img
    
    # Extract positions
    positions = []
    for pose in trajectory:
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        pos = pose[:3, 3]
        positions.append([pos[0], pos[2]])  # Use X and Z for top-down view
    
    positions = np.array(positions)
    
    # Normalize positions to image coordinates
    if len(positions) > 1:
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        range_pos = max_pos - min_pos
        
        # Add margin
        margin = 0.1
        range_pos = range_pos * (1 + 2 * margin)
        min_pos = min_pos - range_pos * margin
        
        # Scale to image
        if range_pos[0] > 0 and range_pos[1] > 0:
            scale_x = (w - 20) / range_pos[0]
            scale_y = (h - 20) / range_pos[1]
            scale = min(scale_x, scale_y)
            
            # Convert to image coordinates
            img_positions = []
            for pos in positions:
                x = int((pos[0] - min_pos[0]) * scale + 10)
                y = int((pos[1] - min_pos[1]) * scale + 10)
                img_positions.append((x, y))
            
            # Draw trajectory
            for i in range(len(img_positions) - 1):
                cv2.line(vis_img, img_positions[i], img_positions[i+1], color, thickness)
            
            # Draw start and end points
            if len(img_positions) > 0:
                cv2.circle(vis_img, img_positions[0], 5, (0, 255, 0), -1)  # Green start
                cv2.circle(vis_img, img_positions[-1], 5, (0, 0, 255), -1)  # Red end
    
    return vis_img
