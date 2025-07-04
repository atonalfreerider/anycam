import sys
import os

import cv2

sys.path.append(".")
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

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
from anycam.models.depth_predictor_wrapper import NPZDepthWrapper


def load_video_batch(video_path, start_frame, end_frame):
    """Load a specific range of frames from video"""
    video = VideoFileClip(video_path)

    frames = []
    for i, frame in enumerate(video.iter_frames()):
        if i < start_frame:
            continue
        if i >= end_frame:
            break
        frames.append(frame.astype(np.float32) / 255.0)

    fps = video.fps
    video.close()  # Explicitly close to free memory

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


def process_video_batch(model, criterion, frames, ba_refinement=True, output_path=None, batch_idx=0):
    """
    Process a batch of video frames by fitting the AnyCam model.
    
    Args:
        model: The AnyCam model
        criterion: The loss criterion
        frames: List of frames as numpy arrays with shape (H,W,3) and values in [0,1]
        ba_refinement: Whether to perform bundle adjustment refinement (default: True)
        output_path: Path to output directory (depths assumed to be in output_path/depths)
        batch_idx: Index of the current batch for frame offset calculation
    
    Returns:
        trajectory: The estimated camera trajectory
        proj: The camera projection matrix
        extras_dict: Additional information from the fitting process
        ba_extras: Bundle adjustment extra information
    """

    if output_path is None:
        raise ValueError("output_path is required")
    
    # Assume depths are in output_path/depths
    depth_dir = Path(output_path) / "depths"
    if not depth_dir.exists():
        raise ValueError(f"Depths directory does not exist: {depth_dir}")

    # Default configuration with memory optimizations
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
    
    # Ensure the BA refinement setting is applied to the config
    config.do_ba_refinement = ba_refinement

    # Replace depth predictor with NPZ loader
    npz_depth_predictor = NPZDepthWrapper({
        "depth_dir": str(depth_dir),
        "scaling": 1.0,
        "frame_pattern": "depth_{:06d}.npz",
        "depth_key": "depth"
    }).cuda()

    # Replace the depth predictor in the model
    model._depth_predictor = npz_depth_predictor

    # Set frame index to start from the batch offset - CRITICAL FIX for batch processing
    start_frame_idx = batch_idx * len(frames)
    npz_depth_predictor.set_frame_index(start_frame_idx)

    # Verify frame count alignment and check for valid depth range
    print(f"Batch {batch_idx}: Processing {len(frames)} frames starting from frame {start_frame_idx}")
    print(f"Available depth frames: {len(npz_depth_predictor.available_frames)}")
    
    # Check if we have enough depth frames for this batch
    max_frame_needed = start_frame_idx + len(frames) - 1
    if max_frame_needed >= max(npz_depth_predictor.available_frames):
        print(f"Warning: Batch {batch_idx} needs frame {max_frame_needed} but max available is {max(npz_depth_predictor.available_frames)}")

    print(f"Bundle adjustment refinement: {'Enabled' if ba_refinement else 'Disabled'}")
    
    # Clear ALL GPU memory before processing - more aggressive cleanup
    if torch.cuda.is_available():
        # Clear depth predictor cache
        npz_depth_predictor.clear_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial GPU memory usage: {initial_memory:.2f} GB")

    try:
        # Run fit_video function with batch-specific frame offset
        trajectory, proj, extras_dict, ba_extras = fit_video(
            config,
            model,
            criterion,
            frames,
            return_extras=True,
            start_frame_offset=start_frame_idx,
        )
        
        # Check for NaN values in trajectory
        for i, pose in enumerate(trajectory):
            if torch.isnan(pose).any() or torch.isinf(pose).any():
                print(f"Warning: NaN or Inf detected in trajectory pose {i} for batch {batch_idx}")
                # Replace with identity matrix
                trajectory[i] = torch.eye(4, dtype=pose.dtype, device=pose.device)
        
        print(f"Finished processing batch {batch_idx}")
        return trajectory, proj, extras_dict, ba_extras
        
    except Exception as e:
        print(f"Error in process_video_batch for batch {batch_idx}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return dummy results to prevent complete failure
        dummy_trajectory = [torch.eye(4).float() for _ in range(len(frames))]
        dummy_proj = np.eye(3).astype(np.float32)
        dummy_extras = {"best_candidate": 0, "seq_depths": [], "ba_uncertainties": None}
        return dummy_trajectory, dummy_proj, dummy_extras, None


def get_video_info(video_path):
    """Get basic video information without loading frames"""
    video = VideoFileClip(video_path)
    total_frames = int(video.fps * video.duration)
    fps = video.fps
    width = video.w
    height = video.h
    video.close()
    
    return total_frames, fps, width, height


@hydra.main(version_base=None, config_name=None)
def main(cfg: DictConfig):
    """
    AnyCam demo script for processing videos and extracting 3D information.
    
    Example usage:
    - Process video in batches: python anycam_demo.py input_path=/path/to/video output_path=/path/to/output batch_size=150
    - Disable BA refinement: python anycam_demo.py input_path=/path/to/video output_path=/path/to/output ba_refinement=false
    - Process single batch: python anycam_demo.py input_path=/path/to/video output_path=/path/to/output batch_idx=0 batch_size=150
    
    Config parameters:
    - input_path: Path to video or directory of images
    - output_path: Path to output directory (depths expected in output_path/depths)
    - model_path: Path to model (optional)
    - checkpoint: Specific checkpoint to use (optional)
    - batch_size: Number of frames to process per batch (default: 150)
    - batch_idx: Specific batch index to process (optional, if not specified, processes all batches)
    - export_colmap: Whether to export to COLMAP format (boolean)
    - image_size: Target image size for processing (default: 336)
    - ba_refinement: Whether to perform bundle adjustment refinement (default: True)
    - fps: Target frames per second (default: 0, use all frames)
    """
    # Enable TensorFloat32 for better performance on modern GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print("Enabled TensorFloat32 for improved matrix multiplication performance")
    
    input_path = cfg.get("input_path", None)
    output_path = cfg.get("output_path", None)
    model_path = cfg.get("model_path", None)
    checkpoint = cfg.get("checkpoint", None)
    batch_size = cfg.get("batch_size", 300)
    batch_idx = cfg.get("batch_idx", None)  # If specified, process only this batch
    image_size = cfg.get("image_size", 336)
    ba_refinement = cfg.get("ba_refinement", True)
    
    if input_path is None:
        print("Error: input_path is required")
        return
        
    if output_path is None:
        print("Error: output_path is required")
        return
        
    if model_path is None:
        print("Using default model path")
        model_path = Path(__file__).parent.parent.parent / "outputs"
    else:
        model_path = Path(model_path)
    
    # Verify depths directory exists
    depth_dir = Path(output_path) / "depths"
    if not depth_dir.exists():
        print(f"Error: depths directory does not exist: {depth_dir}")
        print("Please ensure depth maps are pre-computed and stored in output_path/depths/")
        return
    
    # Get video information without loading frames
    print(f"Analyzing video: {input_path}")
    total_frames, fps, width, height = get_video_info(input_path)
    
    print(f"Video info: {total_frames} frames, {fps:.2f} fps, {width}x{height}")
    
    # Calculate total number of batches
    total_batches = (total_frames + batch_size - 1) // batch_size
    print(f"Will process {total_batches} batches of up to {batch_size} frames each")
    
    # Determine which batches to process
    if batch_idx is not None:
        if batch_idx >= total_batches:
            print(f"Error: batch_idx {batch_idx} is out of range (0 to {total_batches-1})")
            return
        batch_indices = [batch_idx]
        print(f"Processing only batch {batch_idx}")
    else:
        batch_indices = list(range(total_batches))
        print(f"Processing all {total_batches} batches")
    
    # Load model once
    print(f"Loading model from {model_path}")
    model, criterion = load_anycam(model_path, depth_dir, checkpoint)
    model = model.cuda().eval()
    
    # Process each batch
    for current_batch_idx in batch_indices:
        print(f"\n{'='*50}")
        print(f"Processing batch {current_batch_idx + 1}/{total_batches}")
        print(f"{'='*50}")
        
        # Calculate batch frame range
        start_idx = current_batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_frames)
        
        print(f"Batch {current_batch_idx}: Loading frames {start_idx} to {end_idx-1} ({end_idx - start_idx} frames)")
        
        # Load only the frames needed for this batch
        try:
            batch_frames, _ = load_video_batch(input_path, start_idx, end_idx)
            
            if not batch_frames:
                print(f"Error: No frames loaded for batch {current_batch_idx}")
                continue
            
            # Format frames for processing
            batch_frames = format_frames(batch_frames, target_size=image_size)
            print(f"Loaded and resized {len(batch_frames)} frames to {batch_frames[0].shape[:2]}")
            
        except Exception as e:
            print(f"Error loading frames for batch {current_batch_idx}: {str(e)}")
            continue
        
        # Aggressive GPU memory clearing before each batch
        if torch.cuda.is_available():
            # Move model to CPU temporarily to clear GPU memory
            model_device = next(model.parameters()).device
            if model_device.type == 'cuda':
                print(f"Moving model to CPU to clear GPU memory")
                model = model.cpu()
                
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear PyTorch cache multiple times
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Check memory after clearing
            memory_after_clear = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"Memory after clearing: {memory_after_clear:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            
            # Move model back to GPU
            print(f"Moving model back to GPU")
            model = model.cuda()
            
            print(f"Cleared GPU cache before batch {current_batch_idx}")
        
        try:
            # Process this batch
            trajectory, proj, extras_dict, ba_extras = process_video_batch(
                model, 
                criterion, 
                batch_frames, 
                ba_refinement=ba_refinement,
                output_path=output_path,
                batch_idx=current_batch_idx
            )

            trajectory = [se3_ensure_numerical_accuracy(pose.clone().detach()) for pose in trajectory]
            
            # Save batch results
            if output_path:
                batch_output_path = Path(output_path) / f"anycam_batch_{current_batch_idx}"
                batch_output_path.mkdir(parents=True, exist_ok=True)
                
                print(f"Saving batch {current_batch_idx} results to {batch_output_path}")
                
                # Save trajectory as numpy array
                trajectory_np = np.stack([pose.cpu().numpy() for pose in trajectory])
                np.save(batch_output_path / "trajectory.npy", trajectory_np)
                
                # Save projection matrix
                if isinstance(proj, torch.Tensor):
                    proj_np = proj.cpu().numpy()
                else:
                    proj_np = proj
                np.save(batch_output_path / "projection.npy", proj_np)
                
                # Save batch metadata
                batch_metadata = {
                    "batch_idx": current_batch_idx,
                    "start_frame": start_idx,
                    "end_frame": end_idx - 1,
                    "num_frames": len(batch_frames),
                    "total_batches": total_batches,
                    "total_frames": total_frames,
                    "batch_size": batch_size,
                    "ba_refinement": ba_refinement,
                    "image_size": image_size,
                }
                np.save(batch_output_path / "metadata.npy", batch_metadata)

                print(f"Saved batch {current_batch_idx} results successfully")
                
        except Exception as e:
            print(f"Error processing batch {current_batch_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        finally:
            # Aggressive memory cleanup after each batch
            # Delete trajectory and other large objects
            if 'trajectory' in locals():
                del trajectory
            if 'proj' in locals():
                del proj
            if 'extras_dict' in locals():
                del extras_dict
            if 'ba_extras' in locals():
                del ba_extras
            
            # Clear batch frames from memory
            del batch_frames
            
            # Clear depth predictor cache if it exists
            if hasattr(model, '_depth_predictor') and hasattr(model._depth_predictor, 'clear_cache'):
                model._depth_predictor.clear_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Force clear GPU memory multiple times
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                final_memory = torch.cuda.memory_allocated() / 1024**3
                reserved_memory = torch.cuda.memory_reserved() / 1024**3
                print(f"Final memory after batch {current_batch_idx}: {final_memory:.2f}GB allocated, {reserved_memory:.2f}GB reserved")
    
    print(f"\nCompleted processing {'all' if batch_idx is None else 'selected'} batches")
    if output_path and batch_idx is None:
        print(f"All batch results saved to: {output_path}")
        print(f"Use anycam_stitch.py to combine results and visualize")


if __name__ == "__main__":
    main()
