import numpy as np
import uuid

import rerun as rr
from anycam.common.geometry import get_grid_xy
from anycam.visualization.common import color_tensor

import torch

def plot_to_rerun(
        trajectory,
        depths,
        imgs,
        proj,
        uncertainties=None,
        subsample_pts=1,
        radii=1.5,
        uncertainty_thresh=-1,
        max_depth=-1,
        filter_depth_threshold=0.1,
        image_plane_distance=0.05,
        rerun_mode="spawn",
):

    h, w = imgs[0].shape[:2]

    def filter_depth(depth, threshold=0.1):
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        _, h, w = depth.shape

        depth = depth.clone()[None, ...]
        median = torch.median(depth)

        depth_grad = torch.stack(torch.gradient(depth, dim=(-2, -1))).norm(dim=0)

        mask = depth_grad < median * threshold

        return mask

    def lift_image(img, depth, pose, proj):
        h, w = img.shape[:2]
        device = depth.device

        # Ensure depth has the right dimensions and resize if needed
        depth = depth.unsqueeze(0)

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0) if depth.dim() == 3 else depth,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        proj = proj.clone().detach().to(device).float()

        proj_normalized = proj.clone()
        proj_normalized[0, 0] = proj_normalized[0, 0] / w * 2
        proj_normalized[1, 1] = proj_normalized[1, 1] / h * 2
        proj_normalized[0, 2] = proj_normalized[0, 2] / w * 2 - 1
        proj_normalized[1, 2] = proj_normalized[1, 2] / h * 2 - 1

        inv_proj = torch.inverse(proj_normalized)

        pts = get_grid_xy(h, w, homogeneous=True).reshape(3, h* w).to(device)
        pts = inv_proj @ pts
        pts = pts * depth.view(1, -1).to(device)
        pts = torch.cat((pts, torch.ones(1, h * w, device=device)), dim=0)
        pts = pose.to(pts.dtype) @ pts
        pts = pts[:3, :].T

        colors = torch.from_numpy(img.reshape(-1, 3)).to(device)

        return pts, colors

    imgs = np.array(imgs)

    # Initialize rerun with appropriate mode
    if rerun_mode == "spawn":
        rr.init("AnyCam Demo", recording_id=uuid.uuid4(), spawn=True)
    elif rerun_mode == "connect":
        rr.init("AnyCam Demo", recording_id=uuid.uuid4(), spawn=False)
        print(f"Connecting to existing Rerun server.")
        rr.connect()
    else:
        raise ValueError(f"Unsupported rerun mode: {rerun_mode}. Use 'spawn' or 'connect'.")

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log("world/scene", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Horizontal(
            rr.blueprint.Spatial3DView(origin="/world/scene"),
            rr.blueprint.Vertical(
                rr.blueprint.Spatial2DView(origin="/world/scene/active_cam/input"),
                rr.blueprint.Spatial2DView(origin="/world/scene/active_cam/uncertainty"),
            ),
        ),
    )
    rr.send_blueprint(blueprint, make_active=True)

    # Calculate mapping between trajectory indices and depth/keyframe indices
    print(f"Visualization info: {len(trajectory)} poses, {len(depths)} depths, {len(imgs)} images")

    for traj_id in range(len(trajectory)):
        rr.set_time_sequence("step", traj_id)

        pose = trajectory[traj_id]
        rot = pose[:3, :3].cpu().numpy()

        # Convert projection matrix for rerun logging
        if isinstance(proj, np.ndarray):
            proj_focal = float(proj[0, 0])
        else:
            proj_focal = float(proj[0, 0])

        rr.log(f"world/scene/active_cam", rr.Pinhole(
            resolution=[w, h],
            focal_length=proj_focal,
            image_plane_distance=image_plane_distance,
        ), static=True)
        rr.log(f"world/scene/active_cam", rr.Transform3D(translation=pose[:3, 3].cpu(), mat3x3=rot, axis_length=0.01))

        rr.log("world/scene/cam_traj",
               rr.LineStrips3D([pose[:3, 3].cpu().numpy().tolist() for pose in trajectory[:traj_id + 1]],
                               colors=[(0, 255, 0)]), static=False)

        # Map trajectory index to image and depth indices
        img_id = min(traj_id, len(imgs) - 1)  # Direct mapping for images

        # For depths, use the keyframe mapping if available
        depth_id = traj_id

        # Ensure depth_id is within bounds
        if depth_id is not None:
            depth_id = min(max(depth_id, 0), len(depths) - 1)

        rr.log("world/scene/active_cam/input",
               rr.Image((imgs[img_id] * 255).astype(np.uint8)).compress(jpeg_quality=95))

        rr.log("world/scene",
               rr.Transform3D(translation=pose[:3, 3].cpu(), mat3x3=rot, axis_length=0, from_parent=True))

        # Show depth if available
        if depth_id is not None and depth_id < len(depths):
            try:
                # Get the depth for this frame
                depth = depths[depth_id]

                # Ensure depth is on CUDA and has proper dimensions
                depth = depth.cuda()

                # Handle depth dimensions
                depth = depth.squeeze(0)

                # Resize depth to match image dimensions if needed
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)

                # Lift points from depth
                pts, colors = lift_image(imgs[img_id], depth, trajectory[traj_id].cuda(), proj)

                # Compute mask on the depth
                mask = filter_depth(depth, threshold=filter_depth_threshold)
                mask = mask.view(-1)

                if max_depth > 0:
                    depth_mask = depth.view(-1) < max_depth
                    mask = mask & depth_mask

                # Apply mask to points and colors
                if len(pts) > 0:
                    pts = pts[mask, :]
                    colors = colors[mask, :]

                    # Subsample points
                    if len(pts) > 0:
                        pts = pts[subsample_pts // 2::subsample_pts]
                        colors = colors[subsample_pts // 2::subsample_pts]
                        colors = (colors * 255).clamp(0, 255).to(torch.uint8)

                        rr.log(f"world/scene/active_points",
                               rr.Points3D(pts[:, :3].cpu().numpy(), colors=colors[:, :3].cpu().numpy(),
                                           radii=rr.Radius.ui_points([radii]), ))

            except Exception as e:
                print(f"Warning: Failed to lift points for frame {traj_id}: {e}")

        # Show uncertainty if available
        if uncertainties is not None and traj_id < len(uncertainties):
            try:
                uncert = uncertainties[traj_id]

                # Handle uncertainty dimensions
                uncert = uncert.squeeze(0)

                # Resize uncertainty to match image dimensions if needed
                uncert = torch.nn.functional.interpolate(
                    uncert.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)

                uncertainty_img = color_tensor((uncert / uncertainty_thresh).clamp(0, 1), cmap="plasma", norm=False)[0]

                uncertainty_img = uncertainty_img.cpu().numpy()
                uncertainty_img = (uncertainty_img * 255).astype(np.uint8)

                # Transpose to HWC format for rerun
                uncertainty_img = uncertainty_img.transpose(1, 2, 0)

                rr.log(f"world/scene/active_cam/uncertainty", rr.Image(uncertainty_img))

            except Exception as e:
                print(f"Warning: Failed to show uncertainty for frame {traj_id}: {e}")