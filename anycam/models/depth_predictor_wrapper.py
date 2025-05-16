import torch
import numpy as np
import os

from torch import nn
import torch.nn.functional as F


class NPZDepthWrapper(nn.Module):
    """
    Depth predictor that loads pre-computed depth maps from NPZ files
    """

    def __init__(self, conf):
        super().__init__()
        self.depth_dir = conf.get("depth_dir", None)
        self.scaling = conf.get("scaling", 1.0)
        self.frame_pattern = conf.get("frame_pattern", "depth_{:06d}.npz")
        self.depth_key = conf.get("depth_key", "depth")
        
        if self.depth_dir is None:
            raise ValueError("depth_dir must be specified for NPZDepthWrapper")
        
        if not os.path.exists(self.depth_dir):
            raise ValueError(f"Depth directory does not exist: {self.depth_dir}")
        
        # Cache for loaded depth maps to avoid repeated I/O
        self.depth_cache = {}
        self.current_frame_idx = 0
        
        # Get available frame indices
        self.available_frames = set()
        for f in os.listdir(self.depth_dir):
            if f.endswith('.npz'):
                try:
                    idx = int(f.split('_')[1].split('.')[0])
                    self.available_frames.add(idx)
                except (ValueError, IndexError):
                    continue
        
        print(f"NPZDepthWrapper initialized with depth_dir: {self.depth_dir}")
        print(f"Found {len(self.available_frames)} depth files")
        
    def _load_depth_frame(self, frame_idx):
        """Load a single depth frame from NPZ file"""

        if frame_idx in self.depth_cache:
            return self.depth_cache[frame_idx]
            
        depth_path = os.path.join(self.depth_dir, self.frame_pattern.format(frame_idx))
        
        if not os.path.exists(depth_path):
            if not self.available_frames:
                raise FileNotFoundError(f"No NPZ depth files found in directory: {self.depth_dir}")
            
            # Find closest frame
            closest_idx = min(self.available_frames, key=lambda x: abs(x - frame_idx))
            depth_path = os.path.join(self.depth_dir, self.frame_pattern.format(closest_idx))
            print(f"Warning: Frame {frame_idx} not found, using closest frame {closest_idx}")
        
        try:
            depth_data = np.load(depth_path)
            depth = depth_data[self.depth_key].astype(np.float32)

            # Cache the depth map (limit cache size to avoid memory issues)
            if len(self.depth_cache) < 100:  # Limit cache size
                self.depth_cache[frame_idx] = depth
                
            return depth
        except Exception as e:
            raise RuntimeError(f"Failed to load depth from {depth_path}: {str(e)}")

    def forward(self, rgbs, return_features=False, frame_indices=None):
        """
        Args:
            rgbs: Input RGB images (not used for depth computation)
            return_features: Whether to return features (always returns zeros)
            frame_indices: List/tensor of frame indices to load depth for
        """
        n, c, h, w = rgbs.shape
        device = rgbs.device
        
        depths = []
        
        # If frame_indices are provided, use them directly without auto-incrementing
        use_explicit_indices = frame_indices is not None

        for i in range(n):
            if use_explicit_indices:
                if isinstance(frame_indices, (list, tuple)):
                    frame_idx = frame_indices[i] if i < len(frame_indices) else frame_indices[-1]
                elif torch.is_tensor(frame_indices):
                    frame_idx = int(frame_indices[i].item()) if i < len(frame_indices) else int(frame_indices[-1].item())
                else:
                    frame_idx = int(frame_indices)
            else:
                frame_idx = self.current_frame_idx + i
            
            # Load depth from NPZ file
            depth_np = self._load_depth_frame(frame_idx)
            
            # Convert to tensor and resize to match input dimensions
            depth_tensor = torch.from_numpy(depth_np.copy()).to(device).float()
            
            # Ensure depth has correct dimensions
            if depth_tensor.dim() == 2:
                depth_tensor = depth_tensor.unsqueeze(0)  # Add channel dimension
            elif depth_tensor.dim() == 3 and depth_tensor.shape[0] != 1:
                depth_tensor = depth_tensor.unsqueeze(0)  # Add batch dimension
            
            if depth_tensor.shape[-2:] != (h, w):
                depth_tensor = F.interpolate(
                    depth_tensor.unsqueeze(0) if depth_tensor.dim() == 3 else depth_tensor,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            # Apply scaling
            depth_tensor = depth_tensor * self.scaling

            # Create a completely new tensor to ensure no references are shared
            depth_tensor = depth_tensor.clone().detach()
            
            depths.append(depth_tensor)
        
        # Only update frame index for next call if we're not using explicit indices
        if not use_explicit_indices:
            self.current_frame_idx += n

        depth_batch = torch.stack(depths, dim=0)

        if not return_features:
            return [depth_batch]
        else:
            # Return zero features since we don't have actual features
            features = torch.zeros_like(depth_batch)
            return [depth_batch], features
    
    def set_frame_index(self, frame_idx):
        """Set the current frame index for sequential processing"""
        self.current_frame_idx = frame_idx

    def reset_frame_index(self):
        """Reset frame index to 0"""
        self.current_frame_idx = 0

    def clear_cache(self):
        """Clear the depth cache to free memory"""
        self.depth_cache.clear()

    @classmethod
    def from_conf(cls, conf):
        return cls(conf)
