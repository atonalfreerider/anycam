import numpy as np
import uuid
from pathlib import Path

import rerun as rr
from anycam.common.geometry import get_grid_xy

import torch

def plot_to_rerun(
        trajectory,
        depths,
        imgs,
        proj,
        subsample_pts=1,
        radii=1.5,
        max_depth=-1,
        filter_depth_threshold=0.1,
        image_plane_distance=0.05,
        rerun_mode="spawn",
        depth_dir=None,
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

        # Convert proj to tensor if it's a numpy array
        if isinstance(proj, np.ndarray):
            proj = torch.from_numpy(proj).to(device).float()
        else:
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
        
        # Convert pose to tensor if it's a numpy array
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose).to(device).float()
        else:
            pose = pose.to(device).float()
            
        pts = pose @ pts
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

    # Load depths from depth_dir if provided, otherwise use passed depths
    if depth_dir is not None and Path(depth_dir).exists():
        print(f"Loading depths from depth_dir: {depth_dir}")
        from anycam.models.depth_predictor_wrapper import NPZDepthWrapper
        
        depth_loader = NPZDepthWrapper({
            "depth_dir": depth_dir,
            "scaling": 1.0,
            "frame_pattern": "depth_{:06d}.npz",
            "depth_key": "depth"
        })
        
        # Load all depths for the trajectory
        loaded_depths = []
        for frame_idx in range(len(trajectory)):
            try:
                depth_np = depth_loader._load_depth_frame(frame_idx)
                depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)  # Add channel dimension
                loaded_depths.append(depth_tensor)
            except Exception as e:
                print(f"Warning: Could not load depth for frame {frame_idx}: {e}")
                # Create dummy depth
                dummy_depth = torch.ones(1, h, w) * 1.0
                loaded_depths.append(dummy_depth)
        
        depths = loaded_depths
        print(f"Loaded {len(depths)} depth maps from depth_dir")
    elif depths is None:
        depths = []
        print("Warning: No depths available for visualization")

    # Calculate mapping between trajectory indices and depth/keyframe indices
    print(f"Visualization info: {len(trajectory)} poses, {len(depths)} depths, {len(imgs)} images")
    
    # Validate that we have enough data for visualization
    if len(depths) == 0:
        print("Warning: No depths available for visualization")
    elif len(depths) != len(trajectory):
        print(f"Warning: Depth count ({len(depths)}) doesn't match trajectory count ({len(trajectory)})")
        print("This may cause visualization to stop early")
    
    if len(imgs) != len(trajectory):
        print(f"Info: Image count ({len(imgs)}) doesn't match trajectory count ({len(trajectory)})")
        print("Images will be reused/sampled to match trajectory length")

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

        # Map trajectory index to image and depth indices with bounds checking
        if len(imgs) > 0:
            # Sample image index proportionally if we have fewer images than trajectory poses
            img_id = min(int(traj_id * len(imgs) / len(trajectory)), len(imgs) - 1)
        else:
            img_id = 0

        # For depths, ensure we don't go out of bounds
        depth_id = min(traj_id, len(depths) - 1) if depths else None

        if len(imgs) > 0:
            rr.log("world/scene/active_cam/input",
                   rr.Image((imgs[img_id] * 255).astype(np.uint8)).compress(jpeg_quality=95))

        rr.log("world/scene",
               rr.Transform3D(translation=pose[:3, 3].cpu(), mat3x3=rot, axis_length=0, from_parent=True))

        # Show depth if available
        if depth_id is not None and depth_id < len(depths):
            try:
                # Get the depth for this frame
                depth = depths[depth_id]

                # Debug: Print depth info for first few frames
                if traj_id < 5:
                    print(f"Frame {traj_id}: Using depth {depth_id}, depth shape: {depth.shape}, mean: {depth.mean():.6f}")

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

                # Lift points from depth only if we have images
                if len(imgs) > 0:
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
        else:
            if traj_id < 5:  # Only print for first few frames to avoid spam
                print(f"Frame {traj_id}: No depth available (depth_id={depth_id}, depths_len={len(depths)})")
