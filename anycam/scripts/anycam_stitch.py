import sys
import os
import cv2
import argparse
import numpy as np
import torch
from pathlib import Path
import glob
from moviepy import VideoFileClip

sys.path.append(".")
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from anycam.scripts.plot import plot_to_rerun

def load_batch_results(batch_dir):
    """Load results from a single batch directory"""
    batch_path = Path(batch_dir)
    
    if not batch_path.exists():
        raise ValueError(f"Batch directory does not exist: {batch_dir}")
    
    # Load trajectory
    trajectory_path = batch_path / "trajectory.npy"
    if not trajectory_path.exists():
        raise ValueError(f"Trajectory file not found: {trajectory_path}")
    trajectory_np = np.load(trajectory_path)
    trajectory = [torch.from_numpy(pose) for pose in trajectory_np]
    
    # Load projection
    proj_path = batch_path / "projection.npy"
    if not proj_path.exists():
        raise ValueError(f"Projection file not found: {proj_path}")
    proj = np.load(proj_path)
    
    # Load metadata
    metadata_path = batch_path / "metadata.npy"
    metadata = {}
    if metadata_path.exists():
        metadata = np.load(metadata_path, allow_pickle=True).item()

    return {
        "trajectory": trajectory,
        "proj": proj,
        "metadata": metadata,
        "batch_path": batch_path
    }


def stitch_trajectories(batch_results, method="concatenate"):
    """
    Stitch multiple batch trajectories together
    
    Args:
        batch_results: List of batch result dictionaries
        method: Stitching method ("concatenate", "smooth", "overlap")
    
    Returns:
        stitched_trajectory: Combined trajectory
        stitched_metadata: Combined metadata
    """
    
    # Sort batches by batch index
    batch_results = sorted(batch_results, key=lambda x: x["metadata"].get("batch_idx", 0))
    
    print(f"Stitching {len(batch_results)} batches using method: {method}")
    
    if method == "concatenate":
        # Simple concatenation - need to handle frame overlap properly
        stitched_trajectory = []
        for i, batch_result in enumerate(batch_results):
            trajectory = batch_result["trajectory"]
            
            if i == 0:
                # First batch - add all poses
                stitched_trajectory.extend(trajectory)
            else:
                # Subsequent batches - skip first pose to avoid duplication
                # but apply transformation to align with previous batch
                if len(stitched_trajectory) > 0 and len(trajectory) > 1:
                    last_pose = stitched_trajectory[-1]
                    first_pose = trajectory[0]
                    
                    # Compute relative transformation
                    rel_transform = torch.inverse(first_pose.float()) @ last_pose.float()
                    
                    # Transform all poses in this batch (skip first to avoid duplication)
                    for pose in trajectory[1:]:
                        transformed_pose = rel_transform @ pose.float()
                        stitched_trajectory.append(transformed_pose)
                else:
                    # Fallback: just extend if something goes wrong
                    stitched_trajectory.extend(trajectory[1:] if len(trajectory) > 1 else trajectory)
        
    elif method == "smooth":
        # TODO: Implement smooth stitching with overlap blending
        print("Smooth stitching not yet implemented, falling back to concatenate")
        return stitch_trajectories(batch_results, method="concatenate")
        
    elif method == "overlap":
        # TODO: Implement overlap-based stitching
        print("Overlap stitching not yet implemented, falling back to concatenate")
        return stitch_trajectories(batch_results, method="concatenate")
    
    else:
        raise ValueError(f"Unknown stitching method: {method}")
    
    # Combine metadata
    stitched_metadata = {
        "total_frames": len(stitched_trajectory),  # Use actual stitched length
        "num_batches": len(batch_results),
        "batch_indices": [br["metadata"].get("batch_idx", i) for i, br in enumerate(batch_results)],
        "stitching_method": method,
    }
    
    # Copy metadata from first batch
    if batch_results:
        first_metadata = batch_results[0]["metadata"]
        for key in ["ba_refinement", "image_size", "batch_size"]:
            if key in first_metadata:
                stitched_metadata[key] = first_metadata[key]
    
    print(f"Stitched trajectory contains {len(stitched_trajectory)} poses")
    
    return stitched_trajectory, stitched_metadata


def export_combined_results(output_path, trajectory, proj, metadata=None):
    """Export the stitched results to files"""
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting combined results to {output_path}")
    
    # Save stitched trajectory
    trajectory_np = np.stack([pose.cpu().numpy() for pose in trajectory])
    np.save(output_path / "stitched_trajectory.npy", trajectory_np)
    
    # Save projection matrix
    if proj is not None:
        np.save(output_path / "stitched_projection.npy", proj)

    # Save metadata
    if metadata is not None:
        np.save(output_path / "stitched_metadata.npy", metadata)
    
    print(f"Successfully exported stitched results")


def load_original_frames(video_path, frame_count, image_size=336):
    """Load frames for visualization without holding all in memory"""
    try:
        """Load specific number of frames for visualization without holding all in memory"""
        video = VideoFileClip(video_path)

        frames = []
        total_frames = int(video.fps * video.duration)

        # Sample frames evenly across the video
        if frame_count >= total_frames:
            # Use all frames
            indices = list(range(total_frames))
        else:
            # Sample evenly
            indices = [int(i * total_frames / frame_count) for i in range(frame_count)]

        print(f"Loading {len(indices)} visualization frames from {total_frames} total frames")

        for i, frame in enumerate(video.iter_frames()):
            if i in indices:
                frame_processed = frame.astype(np.float32) / 255.0
                frame_processed = format_frames([frame_processed], target_size=image_size)[0]
                frames.append(frame_processed)

            if len(frames) >= len(indices):
                break

        video.close()
        return frames
    except Exception as e:
        print(f"Warning: Could not load original frames: {e}")
        return None


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


def main():
    """
    Stitch AnyCam batch results and optionally visualize them.
    
    Example usage:
    - Stitch all batches: python anycam_stitch.py --input_path=/path/to/batch/output
    - Stitch and visualize: python anycam_stitch.py --input_path=/path/to/batch/output --visualize=true
    - Use specific video for visualization: python anycam_stitch.py --input_path=/path/to/batch/output --video_path=/path/to/video.mp4 --visualize=true
    
    Config parameters:
    - input_path: Path to directory containing batch results (anycam_batch_* folders)
    - video_path: Path to original video for visualization (optional)
    - visualize: Whether to visualize results with rerun (boolean)
    - rerun_mode: Mode to use for rerun visualization ('spawn' or 'connect', default: 'spawn')
    - stitching_method: Method for stitching trajectories ('concatenate', 'smooth', 'overlap', default: 'concatenate')
    """
    
    # Enable TensorFloat32 for better performance on modern GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print("Enabled TensorFloat32 for improved matrix multiplication performance")
    
    parser = argparse.ArgumentParser(description="Stitch AnyCam batch results and optionally visualize them.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to directory containing batch results (anycam_batch_* folders)")
    parser.add_argument("--video_path", type=str, default=None, help="Path to original video for visualization (optional)")
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--rerun_mode", type=str, default="spawn", help="Mode to use for rerun visualization ('spawn' or 'connect')")
    parser.add_argument("--stitching_method", type=str, default="concatenate", choices=["concatenate", "smooth", "overlap"], help="Method for stitching trajectories")
    args = parser.parse_args()

    input_path = args.input_path
    video_path = args.video_path
    visualize = args.visualize
    rerun_mode = args.rerun_mode
    stitching_method = args.stitching_method

    if input_path is None:
        print("Error: input_path is required")
        return
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return
    
    # Find all batch directories
    batch_dirs = sorted(glob.glob(str(input_path / "anycam_batch_*")))
    
    if not batch_dirs:
        print(f"Error: No batch directories found in {input_path}")
        print("Expected directories named 'anycam_batch_0', 'anycam_batch_1', etc.")
        return
    
    # Sort batch directories numerically instead of lexicographically
    def extract_batch_number(batch_dir):
        """Extract batch number from directory name for proper sorting"""
        try:
            batch_name = Path(batch_dir).name
            # Extract number from 'anycam_batch_X'
            batch_num = int(batch_name.split('_')[-1])
            return batch_num
        except (ValueError, IndexError):
            return 0
    
    batch_dirs = sorted(batch_dirs, key=extract_batch_number)
    
    print(f"Found {len(batch_dirs)} batch directories")
    
    # Load all batch results
    batch_results = []
    total_expected_frames = 0
    
    for batch_dir in batch_dirs:
        try:
            batch_result = load_batch_results(batch_dir)
            batch_results.append(batch_result)
            
            # Get frame count from metadata if available
            metadata = batch_result.get("metadata", {})
            batch_idx = metadata.get("batch_idx", "?")
            num_frames = len(batch_result["trajectory"])
            start_frame = metadata.get("start_frame", "?")
            end_frame = metadata.get("end_frame", "?")
            
            total_expected_frames += num_frames
            
            print(f"Loaded batch {batch_idx}: {num_frames} poses (frames {start_frame}-{end_frame})")
            
        except Exception as e:
            print(f"Warning: Could not load batch {batch_dir}: {e}")
            continue
    
    if not batch_results:
        print("Error: No valid batch results could be loaded")
        return
    
    print(f"Total expected frames from all batches: {total_expected_frames}")
    
    # Stitch trajectories
    stitched_trajectory, stitched_metadata = stitch_trajectories(batch_results, method=stitching_method)
    
    # Verify stitched frame count
    actual_stitched_frames = len(stitched_trajectory)
    print(f"Expected frames: {total_expected_frames}, Actual stitched frames: {actual_stitched_frames}")
    
    if actual_stitched_frames != total_expected_frames:
        print(f"WARNING: Frame count mismatch! Expected {total_expected_frames}, got {actual_stitched_frames}")
    
    # Get total frames from metadata if available
    if batch_results and "total_frames" in batch_results[0]["metadata"]:
        original_total_frames = batch_results[0]["metadata"]["total_frames"]
        print(f"Original video total frames: {original_total_frames}")
        
        # Update stitched metadata with correct total
        stitched_metadata["original_total_frames"] = original_total_frames
    
    # Stitch other data
    stitched_proj = batch_results[0]["proj"] if batch_results else None
    
    print(f"Stitched results:")
    print(f"  Trajectory: {len(stitched_trajectory)} poses")
    print(f"  Depths will be loaded from input_path/depths during visualization")
    
    # Assume depth_dir is input_path/depths
    depth_dir = input_path / "depths"
    if depth_dir.exists():
        stitched_metadata["depth_dir"] = str(depth_dir)
        print(f"Found depths directory: {depth_dir}")
    else:
        print(f"Warning: No depths directory found at: {depth_dir}")
        depth_dir = None
    
    # Export stitched results
    export_combined_results(
        input_path,
        stitched_trajectory,
        stitched_proj,
        stitched_metadata
    )
    
    # Visualization
    if visualize:
        print("Starting visualization...")
        
        # Load original frames if video_path is provided
        vis_frames = None
        if video_path:
            image_size = stitched_metadata.get("image_size", 336)
            frame_count = len(stitched_trajectory)
            
            # Get original video frame count for validation
            original_total_frames = stitched_metadata.get("original_total_frames", frame_count)
            
            print(f"Loading {frame_count} frames for visualization")
            print(f"Stitched trajectory: {frame_count} poses, Original video: {original_total_frames} frames")
            
            # Ensure we don't try to load more frames than exist in the video
            frame_count_to_load = min(frame_count, original_total_frames)
            
            vis_frames = load_original_frames(video_path, frame_count_to_load, image_size)
        
        if vis_frames is None:
            print("Warning: No frames available for visualization")
            vis_frames = []

        plot_to_rerun(
            trajectory=stitched_trajectory,
            depths=None,  # Will be loaded from depth_dir in plot_to_rerun
            imgs=vis_frames,
            proj=stitched_proj,
            subsample_pts=2,
            radii=1.5,
            max_depth=-1,
            filter_depth_threshold=0.1,
            image_plane_distance=0.05,
            rerun_mode=rerun_mode,
            depth_dir=str(depth_dir) if depth_dir else None,  # Pass depth_dir to plot function
        )
    
    print("Done")


if __name__ == "__main__":
    main()
    main()
    main()
