import sys
import os

import cv2

sys.path.append(".")
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import os
import numpy as np
import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from moviepy import VideoFileClip
from dotdict import dotdict


from anycam.loss import make_loss
from anycam.trainer import AnyCamWrapper
from anycam.utils.geometry import se3_ensure_numerical_accuracy
from anycam.scripts.fit_video import fit_video
from anycam.scripts.plot import plot_to_rerun
from anycam.models.depth_predictor_wrapper import NPZDepthWrapper


def load_video(video_path):
    video = VideoFileClip(video_path)
    frames = [frame for frame in video.iter_frames()]
    frames = [frame.astype(np.float32) / 255.0 for frame in frames]
    fps = video.fps

    return frames, fps


def format_frames(frames, target_size=336):
    height, width = frames[0].shape[:2]

    if height < width:
        new_height = target_size
        new_width = int((target_size / height) * width)
    else:
        new_width = target_size
        new_height = int((target_size / width) * height)
    
    frames = [cv2.resize(frame, (new_width, new_height)) for frame in frames]

    return frames


def load_anycam(model_path, depth_dir, checkpoint=None):
    config = OmegaConf.load(model_path / "training_config.yaml")

    prefix = "training_checkpoint_"
    ckpts = Path(model_path).glob(f"{prefix}*.pt")

    config["model"]["depth_predictor"]["depth_dir"] = depth_dir
    model_conf = config["model"]
    model_conf["use_provided_flow"] = False
    model_conf["train_directions"] = "forward"

    model = AnyCamWrapper(model_conf)

    criterion = [make_loss(cfg) for cfg in config.get("loss", [])][0]

    training_steps = [int(ckpt.stem.split(prefix)[1]) for ckpt in ckpts]

    if training_steps:
        if checkpoint is None:
            ckpt_path = f"{prefix}{max(training_steps)}.pt"
        else:
            ckpt_path = checkpoint

        ckpt_path = Path(model_path) / ckpt_path

        print(ckpt_path)

        cp = torch.load(ckpt_path, map_location="cpu")

        model.load_state_dict(cp["model"], strict=False)

    return model, criterion


def process_video(model, criterion, frames, config=None, ba_refinement=True, depth_dir=None):
    """
    Process a video by fitting the AnyCam model to the provided frames.
    
    Args:
        model: The AnyCam model
        criterion: The loss criterion
        frames: List of frames as numpy arrays with shape (H,W,3) and values in [0,1]
        config: Optional configuration dictionary for the fit_video function
               If None, default configuration will be used
        ba_refinement: Whether to perform bundle adjustment refinement (default: True)
        depth_dir: Optional path to directory containing pre-computed NPZ depth files
    
    Returns:
        trajectory: The estimated camera trajectory
        proj: The camera projection matrix
        extras_dict: Additional information from the fitting process
        ba_extras: Bundle adjustment extra information
    """

    if depth_dir is None:
        raise ValueError("no depth dir")

    # Default configuration with memory optimizations
    if config is None:
        default_config = {
            "with_rerun": False,
            "do_ba_refinement": ba_refinement,
            "prediction": {
                "model_seq_len": 32,  # Reduced from 100 to 32 for memory
                "shift": 31,  # Reduced accordingly
                "square_crop": False,
                "return_all_uncerts": False,
            },
            "ba_refinement": {
                "with_rerun": False,
                "max_uncert": 0.05,
                "lambda_smoothness": 0.1,
                "long_tracks": True,
                "n_steps_last_global": 2000, # Reduced from 5000
            },
            "ba_refinement_level": 1,  # Increased from 0 to reduce frames processed
            "dataset": {
                "image_size": [224, None]  # Reduced from 336 to 224 for memory
            }
        }
        config = dotdict(default_config)
    elif not isinstance(config, dotdict):
        config = dotdict(config)
    
    # Ensure the BA refinement setting is applied to the config
    config.do_ba_refinement = ba_refinement

    # Replace depth predictor with NPZ loader if depth_dir is provided
    # Create NPZ depth wrapper with proper frame alignment
    npz_depth_predictor = NPZDepthWrapper({
        "depth_dir": depth_dir,
        "scaling": 1.0,
        "frame_pattern": "depth_{:06d}.npz",
        "depth_key": "depth"
    }).cuda()

    # Replace the depth predictor in the model
    model._depth_predictor = npz_depth_predictor

    # Reset frame index to ensure alignment with video frames
    npz_depth_predictor.reset_frame_index()

    # Verify frame count alignment
    print(f"Video frames: {len(frames)}, Available depth frames: {len(npz_depth_predictor.available_frames)}")
    if len(frames) > len(npz_depth_predictor.available_frames):
        raise Exception(f"Video has more frames ({len(frames)}) than depth maps ({len(npz_depth_predictor.available_frames)})")

    print(f"Processing {len(frames)} frames...")
    print(f"Bundle adjustment refinement: {'Enabled' if ba_refinement else 'Disabled'}")
    
    # Clear GPU memory before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial GPU memory usage: {initial_memory:.2f} GB")

    # Run fit_video function
    trajectory, proj, extras_dict, ba_extras = fit_video(
        config,
        model,
        criterion,
        frames,
        return_extras=True,
    )
    
    print("Finished processing video")
    return trajectory, proj, extras_dict, ba_extras


@hydra.main(version_base=None, config_name=None)
def main(cfg: DictConfig):
    """
    AnyCam demo script for processing videos and extracting 3D information.
    
    Example usage:
    - Process video: python anycam_demo.py input_path=/path/to/video output_path=/path/to/output
    - Process images: python anycam_demo.py input_path=/path/to/images_folder output_path=/path/to/output
    - Use pre-computed depths: python anycam_demo.py input_path=/path/to/video depth_dir=/path/to/npz_depths
    - Visualize with rerun: python anycam_demo.py input_path=/path/to/video visualize=true
    - Connect to existing rerun server: python anycam_demo.py input_path=/path/to/video visualize=true rerun_mode=connect rerun_address=localhost:8787
    - Export to COLMAP: python anycam_demo.py input_path=/path/to/video export_colmap=true output_path=/path/to/colmap_output
    - Disable BA refinement: python anycam_demo.py input_path=/path/to/video ba_refinement=false
    - Subsample frames: python anycam_demo.py input_path=/path/to/video fps=10
    
    Config parameters:
    - input_path: Path to video or directory of images
    - output_path: Path to save outputs
    - model_path: Path to model (optional)
    - checkpoint: Specific checkpoint to use (optional)
    - depth_dir: Path to directory containing pre-computed NPZ depth files (optional)
    - visualize: Whether to visualize results with rerun (boolean)
    - rerun_mode: Mode to use for rerun visualization ('spawn' or 'connect', default: 'spawn')
    - rerun_address: Address to connect to when using rerun_mode=connect (default: localhost:8787)
    - export_colmap: Whether to export to COLMAP format (boolean)
    - image_size: Target image size for processing (default: 336)
    - ba_refinement: Whether to perform bundle adjustment refinement (default: True)
    - fps: Target frames per second (default: 0, use all frames)
    - vis: Visualization parameters subconfig with the following options:
        - subsample_pts: Point sampling rate (default: 1)
        - radii: Point radius for visualization (default: 1.5)
        - uncertainty_thresh: Threshold for uncertainty visualization (default: 0.05)
        - max_depth: Maximum depth value to consider (default: -1, no limit)
        - filter_depth_threshold: Threshold for depth filtering (default: 0.1)
        - image_plane_distance: Distance of image plane in visualization (default: 0.05)
    """
    input_path = cfg.get("input_path", None)
    output_path = cfg.get("output_path", None)
    model_path = cfg.get("model_path", None)
    checkpoint = cfg.get("checkpoint", None)
    depth_dir = cfg.get("depth_dir", None)  # New parameter for NPZ depths
    visualize = cfg.get("visualize", False)
    rerun_mode = cfg.get("rerun_mode", "spawn")
    export_colmap = cfg.get("export_colmap", False)
    image_size = cfg.get("image_size", 336)
    ba_refinement = cfg.get("ba_refinement", True)
    
    if input_path is None:
        print("Error: input_path is required")
        return
        
    if model_path is None:
        print("Using default model path")
        model_path = Path(__file__).parent.parent.parent / "outputs"
    else:
        model_path = Path(model_path)
    
    # Validate depth_dir if provided
    if depth_dir and not os.path.exists(depth_dir):
        print(f"Error: depth_dir does not exist: {depth_dir}")
        return
    
    # Load input data
    print(f"Loading video from: {input_path}")
    frames, fps = load_video(input_path)
    
    if not frames:
        print("Error: No frames loaded")
        return
        
    print(f"Loaded {len(frames)} frames")

    # Format frames for processing with memory consideration
    frames = format_frames(frames, target_size=image_size)

    print(f"Resized frames to {frames[0].shape[:2]}")
    
    # Load model
    print(f"Loading model from {model_path}")
    model, criterion = load_anycam(model_path, depth_dir, checkpoint)
    model = model.cuda().eval()
    
    # Process frames with memory monitoring
    try:
        trajectory, proj, extras_dict, ba_extras = process_video(
            model, 
            criterion, 
            frames, 
            ba_refinement=ba_refinement,
            depth_dir=depth_dir
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA out of memory error: {e}")
        print("Try reducing image_size, max_frames, or model_seq_len parameters")
        torch.cuda.empty_cache()
        return

    trajectory = [se3_ensure_numerical_accuracy(pose.clone().detach()) for pose in trajectory]
    
    # Extract depth and uncertainty information
    best_candidate = extras_dict["best_candidate"]
    depths = extras_dict["seq_depths"]

    if not ba_refinement:
        processed_frames = extras_dict["images"].permute(0, 2, 3, 1).cpu().numpy()
        uncertainties = torch.stack(extras_dict["uncertainties"])[:, 0, best_candidate, :1, :, :]
        
        # For non-BA case, use processed frames for visualization
        vis_frames = processed_frames
    else:
        # For BA case, fix the depth-to-trajectory mapping
        ba_refinement_level = extras_dict.get("ba_refinement_level", 1)
        
        print(f"BA refinement debug:")
        print(f"  ba_refinement_level from extras: {ba_refinement_level}")
        print(f"  Total trajectory poses: {len(trajectory)}")
        print(f"  Total depth maps: {len(depths)}")
        
        # Calculate the correct ratio
        poses_per_depth = len(trajectory) / len(depths) if len(depths) > 0 else 1
        print(f"  Calculated poses per depth: {poses_per_depth}")
        
        uncertainties = extras_dict["ba_uncertainties"]
        
        # For BA case, use original frames for visualization
        vis_frames = frames

    # Print debugging information
    print(f"Frame mapping debug:")
    print(f"  Total trajectory poses: {len(trajectory)}")
    print(f"  Total depth maps: {len(depths)}")
    print(f"  Total images: {len(vis_frames)}")
    print(f"  BA refinement level: {ba_refinement_level if ba_refinement else 'N/A'}")

    # Extend uncertainties to match trajectory length if needed
    if uncertainties is not None and len(uncertainties) < len(trajectory):
        # For BA case, repeat uncertainties according to actual refinement level
        if ba_refinement:
            extended_uncertainties = []
            actual_refinement_level = len(trajectory) // len(uncertainties) if len(uncertainties) > 0 else 1
            
            for i, uncert in enumerate(uncertainties):
                # Each uncertainty applies to actual_refinement_level trajectory poses
                for _ in range(actual_refinement_level):
                    if len(extended_uncertainties) < len(trajectory):
                        extended_uncertainties.append(uncert)
            
            # Handle any remaining poses
            while len(extended_uncertainties) < len(trajectory):
                extended_uncertainties.append(uncertainties[-1])
                
            uncertainties = torch.stack(extended_uncertainties[:len(trajectory)])
        else:
            # For non-BA case, just repeat the last uncertainty
            last_uncert = uncertainties[-1:] if len(uncertainties) > 0 else torch.zeros_like(uncertainties[:1]) if len(uncertainties) > 0 else None
            if last_uncert is not None:
                while len(uncertainties) < len(trajectory):
                    uncertainties = torch.cat((uncertainties, last_uncert), dim=0)
    
    print(f"Processed video: {len(trajectory)} poses, {len(depths)} depth maps")

    # Save trajectory and projection matrix if output_path is specified
    if output_path and not export_colmap:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving results to {output_path}")
        
        # Save trajectory as numpy array
        trajectory_np = np.stack([pose.cpu().numpy() for pose in trajectory])
        np.save(output_path / "trajectory.npy", trajectory_np)
        
        # Save projection matrix
        if isinstance(proj, torch.Tensor):
            proj_np = proj.cpu().numpy()
        else:
            proj_np = proj
        np.save(output_path / "projection.npy", proj_np)
        
        # Save depths if available
        if depths is not None and len(depths) > 0:
            depths_np = np.stack([depth.cpu().numpy() for depth in depths])
            np.save(output_path / "depths.npy", depths_np)
        
        # Save uncertainties if available
        if uncertainties is not None:
            uncertainties_np = np.stack([uncert.cpu().numpy() for uncert in uncertainties])
            np.save(output_path / "uncertainties.npy", uncertainties_np)
            
        print("Saved all results successfully")

        # Visualization or export
    if visualize:
        # Get visualization parameters from vis subconfig
        vis_config = cfg.get("vis", {})
        
        print(f"Visualizing results with rerun (mode: {rerun_mode})...")
        plot_to_rerun(
            trajectory=trajectory,
            depths=depths,
            imgs=vis_frames,  # Use appropriate frames for visualization
            proj=proj,
            uncertainties=uncertainties,
            subsample_pts=vis_config.get("subsample_pts", 2),
            radii=vis_config.get("radii", 1.5),
            uncertainty_thresh=vis_config.get("uncertainty_thresh", 0.05),
            max_depth=vis_config.get("max_depth", -1),
            filter_depth_threshold=vis_config.get("filter_depth_threshold", 0.1),
            image_plane_distance=vis_config.get("image_plane_distance", 0.05),
            rerun_mode=rerun_mode,
        )
        
    print("Done")


if __name__ == "__main__":
    main()
