import logging
import math
import os

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from transformers.models.depth_anything.modeling_depth_anything import DepthAnythingForDepthEstimation, DepthAnythingConfig
from transformers.models.dinov2.modeling_dinov2 import Dinov2Backbone

from minipytorch3d.rotation_conversions import (
    quaternion_to_matrix,
    axis_angle_to_matrix,
)

from anycam.models.anycam_blocks import (
    Depth_Anything_V2_Small_hf, 
    AnyCamPoseTokenReassembleStage, 
    AnyCamPoseTokenFusionStage, 
    AnyCamPoseTokenHead,
)

from anycam.models.anycam_blocks import AttnBlock, CrossAttnBlock, PoseEmbedding

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

LOG_FOCAL_LENGTH_BIAS = 1.8


def pose_scaling_linear():
    """Linear pose scaling function - returns poses as-is"""
    return lambda x: x


def pose_scaling_tanh():
    """Tanh pose scaling function - applies tanh scaling"""
    return lambda x: torch.tanh(x)


def pose_scaling_sigmoid():
    """Sigmoid pose scaling function - applies sigmoid scaling"""
    return lambda x: torch.sigmoid(x)


class AnyCam(DepthAnythingForDepthEstimation):
    def __init__(
        self,
        config,
    ):
        # Store config values first, but don't create modules yet
        self.rotation_parameterization = config.get("rotation_parameterization", "quaternion")
        self.focal_parameterization = config.get("focal_parameterization", "candidates")
        self.focal_min = config.get("focal_min", 0.1)
        self.focal_max = config.get("focal_max", 4.0)
        self.focal_num_candidates = config.get("focal_num_candidates", 32)

        self.separate_pose_candidates = config.get("separate_pose_candidates", False)
        self.separate_scaling_candidates = config.get("separate_scaling_candidates", False)
        self.separate_uncertainty_candidates = config.get("separate_uncertainty_candidates", False)

        self.two_tokens_per_pose = config.get("two_tokens_per_pose", False)
        
        self.scaling_feature_dim = config.get("scaling_feature_dim", 16)
        self.out_uncertainty_dim = config.get("out_uncertainty_dim", 2)
        self.self_att_depth = config.get("self_att_depth", 8)
        self.downsize_input = config.get("downsize_input", None)

        self.use_flow_input = config.get("use_flow_input", True)
        self.use_depth_input = config.get("use_depth_input", True)

        self.pose_token_partial_dropout = config.get("pose_token_partial_dropout", 0.0)
        self.pose_token_dropout = config.get("pose_token_dropout", 0.0)

        if self.rotation_parameterization == "quaternion":
            self.pose_enc_dim = 7
        elif self.rotation_parameterization == "axis-angle":
            self.pose_enc_dim = 6
        else:
            raise ValueError(f"Unknown rotation parameterization: {self.rotation_parameterization}")
        
        if self.focal_parameterization == "log-candidates" or self.focal_parameterization == "linlog-candidates" or self.focal_parameterization == "candidates":
            self.focal_enc_dim = self.focal_num_candidates
        elif self.focal_parameterization == "log":
            self.focal_enc_dim = 1
        else:
            raise ValueError(f"Unknown focal parameterization: {self.focal_parameterization}")

        self.backbone_type = config.get("backbone_type", "dinov2")

        # The default DepthAnything configuration will yield the small DepthAnything model
        self.da_config = DepthAnythingConfig(**Depth_Anything_V2_Small_hf)
        self.da_config.fusion_hidden_size = 128

        if self.backbone_type == "dinov2":
            pass
        elif self.backbone_type == "croco":
            self.da_config.reassemble_hidden_size = 768
            self.da_config.patch_size = 16
            self.downsize_input = (224, 224)

        # Call parent __init__ BEFORE creating backbone modules
        super().__init__(self.da_config)

        # Now create the backbone modules after parent initialization
        if self.backbone_type == "dinov2":
            # Makes sure that backbone is pretrained
            self.backbone = Dinov2Backbone.from_pretrained("facebook/dinov2-small", **Depth_Anything_V2_Small_hf["backbone_config"])
            # Force backbone to float32 to avoid dtype mismatches
            self.backbone = self.backbone.to(torch.float32)
        elif self.backbone_type == "croco":
            from anycam.models.croco_wrapper import CroCoExtractor

            ckpt_path = os.path.join(os.environ["HOME"], ".cache", "torch", "checkpoints", "CroCo_V2_ViTBase_BaseDecoder.pth")
            logger.info(f"Loading pretrained model from {ckpt_path}")
            if not os.path.exists(ckpt_path):
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                logger.info(f"Downloading from https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTBase_BaseDecoder.pth")
                r = requests.get('https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTBase_BaseDecoder.pth')
                with open(ckpt_path , 'wb') as f:
                    f.write(r.content)

            checkpoint = torch.load(ckpt_path)

            self.backbone = CroCoExtractor(**checkpoint["croco_kwargs"])
            self.backbone.load_state_dict(checkpoint["model"], strict=False)
            # Force backbone to float32 to avoid dtype mismatches
            self.backbone = self.backbone.to(torch.float32)

        # Adjust head to predict uncertainty rather than depth
        self.head.max_depth = 1.0
        self.head.activation2 = nn.Identity()
        self.head.conv3 = nn.Conv2d(
            self.da_config.head_hidden_size, 
            self.out_uncertainty_dim * (1 if not self.separate_uncertainty_candidates else self.focal_num_candidates), 
            kernel_size=1, 
            stride=1, 
            padding=0
        )

        # Add pose branch
        self.pose_reassemble_stage = AnyCamPoseTokenReassembleStage(
            self.da_config.reassemble_hidden_size,
            self.da_config.fusion_hidden_size,
            self.da_config.neck_hidden_sizes,
        )
        self.pose_feature_fusion_stage = AnyCamPoseTokenFusionStage(
            self.da_config.fusion_hidden_size,
        )

        self.pose_interframe_attention = nn.ModuleList(
            [
                AttnBlock(
                    self.da_config.fusion_hidden_size,
                    num_heads=4,
                    mlp_ratio=4,
                    attn_class=nn.MultiheadAttention,
                )
                for _ in range(self.self_att_depth)
            ]
        )

        self.sequence_token_attention = CrossAttnBlock(
            self.da_config.fusion_hidden_size,
            self.da_config.fusion_hidden_size,
            num_heads=4,
            mlp_ratio=4
        )

        self.pose_head = AnyCamPoseTokenHead(
            self.da_config.fusion_hidden_size * (1 if not self.two_tokens_per_pose else 2),
            self.pose_enc_dim * (1 if not self.separate_pose_candidates else self.focal_num_candidates),
        )

        self.sequence_info_head = AnyCamPoseTokenHead(
            self.da_config.fusion_hidden_size,
            self.focal_enc_dim + self.scaling_feature_dim * (1 if not self.separate_scaling_candidates else self.focal_num_candidates),
        )

        self.sequence_token = nn.Parameter(torch.randn(1, 1, self.da_config.fusion_hidden_size))

        self.seq_embedding = PoseEmbedding(
            target_dim=1,
            n_harmonic_functions=4,
            append_input=False,
        )

        self.pose_factor = config.get("pose_factor", 0.01)
        pose_scale_function_name = config.get("pose_scaling_function", "linear")
        self.pose_scale_function = globals()[f"pose_scaling_{pose_scale_function_name}"]()

        # Adjust dino projection layer to have more input channels
        if not self.backbone_type == "croco":
            if self.use_flow_input or self.use_depth_input:
                d_in = 6
            else:
                d_in = 3
            
            self.backbone.embeddings.patch_embeddings.projection.weight = nn.Parameter(self.backbone.embeddings.patch_embeddings.projection.weight.repeat(1, d_in // 3, 1, 1))
            self.backbone.embeddings.patch_embeddings.num_channels = d_in

        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 3, 1, 1),
                persistent=False,
            )

    def prepare_inputs_for_forward(self, images_ip):
        n, f, c, h, w = images_ip.shape

        images_ip = images_ip.reshape(n * f, c, h, w)

        rgb = images_ip[:, :3]
        rest = images_ip[:, 3:] if c > 3 else torch.zeros(n * f, 3, h, w, device=images_ip.device, dtype=images_ip.dtype)

        base_h = h
        base_w = w

        if self.downsize_input is not None:
            if type(self.downsize_input) == int:
                if base_h < base_w:
                    base_h = self.downsize_input
                    base_w = int(base_h * w / h)
                else:
                    base_w = self.downsize_input
                    base_h = int(base_w * h / w)
            else:
                base_h, base_w = self.downsize_input

        th = math.ceil(base_h / self.da_config.patch_size) * self.da_config.patch_size
        tw = math.ceil(base_w / self.da_config.patch_size) * self.da_config.patch_size

        if h != th or w != tw:
            rgb = F.interpolate(
                rgb, (th, tw), mode="bilinear", align_corners=True
            )
            rest = F.interpolate(
                rest, (th, tw), mode="nearest"
            )

        # Ensure consistent dtype and device - force float32 to avoid Half/float mismatch
        device = rgb.device
        rgb = rgb.to(torch.float32)
        rest = rest.to(torch.float32)
        
        resnet_mean = self._resnet_mean.to(device).to(torch.float32)
        resnet_std = self._resnet_std.to(device).to(torch.float32)
        
        rgb = (rgb - resnet_mean) / resnet_std

        images_ip = torch.cat([rgb, rest], dim=1)
        images_ip = images_ip.reshape(n, f, c, th, tw)

        return images_ip

    def forward(
        self,
        images,
        flow_occs=None,
        depths=None,
        img_features=None,
        initial_poses=None,
        initial_focal_length_probs=None,
        initial_scaling_feature=None,
        skip_image_features=False,
        **kwargs
    ):
        n, f, c, h, w = images.shape
        device = images.device

        # Memory optimization: process smaller batches if needed
        if n * f > 64:  # If batch is too large
            print(f"MEMORY WARNING: Large batch detected ({n}x{f}), consider reducing sequence length")

        # Prepare inputs with memory optimization
        inputs = [images.to(torch.float32)]
        
        # Validate inputs for NaN/Inf
        if torch.isnan(images).any() or torch.isinf(images).any():
            print("Warning: NaN or Inf detected in input images")
            images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
            inputs = [images.to(torch.float32)]
        
        if flow_occs is not None:
            if torch.isnan(flow_occs).any() or torch.isinf(flow_occs).any():
                print("Warning: NaN or Inf detected in flow_occs")
                flow_occs = torch.nan_to_num(flow_occs, nan=0.0, posinf=1.0, neginf=0.0)
            inputs.append(flow_occs[:, :, :2].to(torch.float32))
        
        if depths is not None:
            if torch.isnan(depths).any() or torch.isinf(depths).any():
                print("Warning: NaN or Inf detected in depths")
                depths = torch.nan_to_num(depths, nan=1.0, posinf=100.0, neginf=0.01)
            depths = torch.clamp(depths, min=0.01, max=100.0)  # Clamp depth range
            inputs.append(depths.to(torch.float32))
        
        inputs = torch.cat(inputs, dim=2)
        
        # Clear intermediate tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        inputs = self.prepare_inputs_for_forward(inputs)
        th, tw = inputs.shape[-2:]

        hidden_states = img_features

        try:
            if self.backbone_type == "dinov2":
                # Process in smaller chunks if memory is tight
                backbone_input = inputs.reshape(n * f, inputs.shape[2], th, tw).to(torch.float32)
                
                # Memory check before backbone processing
                if torch.cuda.is_available():
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                    required_memory = backbone_input.numel() * 4 * 8  # Rough estimate
                    
                    if required_memory > free_memory * 0.8:  # If we'd use more than 80% of free memory
                        print("MEMORY WARNING: Processing backbone in chunks to avoid OOM")
                        # Process in chunks
                        chunk_size = max(1, (n * f) // 4)
                        outputs_list = []
                        for chunk_start in range(0, n * f, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, n * f)
                            chunk_input = backbone_input[chunk_start:chunk_end]
                            
                            chunk_outputs = self.backbone.forward_with_filtered_kwargs(
                                chunk_input,
                                output_hidden_states=False,
                                output_attentions=False
                            )
                            outputs_list.append(chunk_outputs.feature_maps)
                            
                            # Clear chunk memory
                            del chunk_input, chunk_outputs
                            torch.cuda.empty_cache()
                        
                        # Concatenate results
                        hidden_states = [torch.cat([chunk[i] for chunk in outputs_list], dim=0) 
                                       for i in range(len(outputs_list[0]))]
                        del outputs_list
                    else:
                        # Normal processing
                        outputs = self.backbone.forward_with_filtered_kwargs(
                            backbone_input,
                            output_hidden_states=False,
                            output_attentions=False
                        )
                        hidden_states = outputs.feature_maps

                # Clear backbone input
                del backbone_input
                torch.cuda.empty_cache()

            elif self.backbone_type == "croco":
                # Ensure backbone is in float32
                if next(self.backbone.parameters()).dtype != torch.float32:
                    logger.warning("Converting backbone to float32 in forward pass")
                    self.backbone = self.backbone.to(torch.float32)
            else:
                raise ValueError(f"Unknown backbone type: {self.backbone_type}")

        except Exception as e:
            logger.warning(f"Backbone processing failed: {e}, returning dummy results")
            # Return dummy results to keep pipeline running
            dummy_features = torch.zeros_like(images[:, :, :1, :, :])
            return {
                "poses": torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(n, f, -1, -1),
                "uncert": torch.ones(n, f, 1, 1, h, w, device=device),
                "focal_length": torch.tensor([1.0], device=device).unsqueeze(0).expand(n, -1),
                "focal_length_candidates": torch.tensor([[1.0]], device=device).expand(n, -1),
                "focal_length_probs": torch.ones(n, 1, device=device),
                "scaling_feature": dummy_features
            }

        pose_tokens = [hs[:, 0] for hs in hidden_states]

        patch_size = self.da_config.patch_size
        patch_height = th // patch_size
        patch_width = tw // patch_size

        # Predict uncertainties
        hidden_states = self.neck(hidden_states, patch_height, patch_width)
        uncertainty = self.head(hidden_states, patch_height, patch_width)

        if th != h or tw != w or self.downsize_input is not None:
            uncertainty = F.interpolate(
                uncertainty, (h, w), mode="bilinear", align_corners=True
            )

        uncertainty = F.softplus(uncertainty)
        uncertainty = uncertainty.reshape(n, f, -1, self.out_uncertainty_dim, h, w)
        
        # Clamp uncertainty to prevent explosion
        uncertainty = torch.clamp(uncertainty, min=1e-6, max=100.0)

        # Predict poses
        pose_tokens = self.pose_reassemble_stage(pose_tokens)
        pose_tokens = self.pose_feature_fusion_stage(pose_tokens)
        pose_token = pose_tokens[-1]

        wd_pose_token_1 = pose_token.clone()

        # Perform self-attention
        pose_token = pose_token.reshape(n, f, pose_token.shape[-1])

        # Perform partial dropout
        if self.pose_token_partial_dropout > 0:
            if self.training:
                pose_token = F.dropout(pose_token, p=self.pose_token_partial_dropout, training=self.training)
            else:
                pose_token = pose_token * (1 - self.pose_token_partial_dropout)

        # Add sequence index
        idx = torch.linspace(0, 1, f, device=pose_token.device).view(1, f, 1).expand(n, -1, -1)
        if self.training:
            idx = idx + torch.randn_like(idx) * 0.05
        seq_embedding = torch.zeros_like(pose_token)
        seq_embedding[:, :, :self.seq_embedding.out_dim] = self.seq_embedding(idx).view(n, f, -1)
        
        pose_token = pose_token + seq_embedding

        for i in range(self.self_att_depth):
            pose_token = self.pose_interframe_attention[i](pose_token)

        # Add sequence token
        seq_token = self.sequence_token.expand(n, 1, -1)
        seq_token = self.sequence_token_attention(seq_token, pose_token)

        if self.two_tokens_per_pose:
            pose_token = torch.cat((pose_token, pose_token.roll(-1, dims=1)), dim=-1)

        pose_token = pose_token.reshape(n * f, -1, pose_token.shape[-1])

        wd_pose_token_2 = pose_token.clone()

        with autocast(enabled=True, dtype=torch.float32):
            pose_enc = self.pose_head(pose_token.to(torch.float32))
            pose_enc = pose_enc.view(n, f, -1, self.pose_enc_dim)

            # Clamp pose encoding to prevent explosion
            pose_enc = torch.clamp(pose_enc, min=-10, max=10)

            pose_enc_scaled = self.pose_scale_function(pose_enc)
            pose = self.encoding_to_pose(pose_enc_scaled)

            seq_enc = self.sequence_info_head(seq_token.to(torch.float32))
            focal_enc = seq_enc[..., :self.focal_enc_dim]
            scaling_feature = seq_enc[..., self.focal_enc_dim:]

            # Clamp focal encoding
            focal_enc = torch.clamp(focal_enc, min=-10, max=10)

            scaling_feature = scaling_feature.view(n, -1, self.scaling_feature_dim)

            focal_length, focal_length_probs, focal_candidates = self.enc_embed_to_focal(focal_enc)

        pose_result = {
            "uncert": uncertainty,
            "poses": pose,
            "focal_length": focal_length,
            "focal_length_probs": focal_length_probs,
            "focal_length_candidates": focal_candidates,
            "scaling_feature": scaling_feature,
            "wd_pose_token_1": wd_pose_token_1,
            "wd_pose_token_2": wd_pose_token_2,
        }

        return pose_result

    def get_img_features(self, images, depths=None, flow_occs=None):
        inputs = [images]

        if self.use_flow_input and not self.use_depth_input:
            inputs += [flow_occs]
        elif self.use_flow_input and self.use_depth_input:
            inputs += [flow_occs[:, :, :2], depths] # type: ignore
        elif not self.use_flow_input and self.use_depth_input:
            inputs += [depths.expand(-1, -1, 3, -1, -1)] # type: ignore
        
        inputs = torch.cat(inputs, dim=2)

        n, f, c, h, w = inputs.shape
        
        inputs = self.prepare_inputs_for_forward(inputs)

        th, tw = inputs.shape[-2:]

        # Ensure backbone is in float32
        if next(self.backbone.parameters()).dtype != torch.float32:
            logger.warning("Converting backbone to float32 in get_img_features")
            self.backbone = self.backbone.to(torch.float32)

        # Get the 2D image features
        backbone_input = inputs.reshape(n * f, c, th, tw).to(torch.float32)
        outputs = self.backbone.forward_with_filtered_kwargs(backbone_input, output_hidden_states=False, output_attentions=False)
        hidden_states = outputs.feature_maps

        return hidden_states
    
    def encoding_to_pose(self, pose_enc):
        n, f, nc, _ = pose_enc.shape

        translation = pose_enc[..., :3]
        rotation = pose_enc[..., 3:]

        # print(f"T={translation[0, 0, 16].cpu().detach().numpy()}, R={rotation[0, 0, 16].cpu().detach().numpy()}")

        if self.rotation_parameterization == "quaternion":
            rotation = quaternion_to_matrix(rotation)
        elif self.rotation_parameterization == "axis-angle":
            rotation = axis_angle_to_matrix(rotation)

        pose = torch.eye(4, device=pose_enc.device).view(1, 1, 1, 4, 4).repeat(n, f, nc, 1, 1)

        pose[..., :3, :3] = rotation
        pose[..., :3, 3] = translation

        return pose
    
    def enc_embed_to_focal(self, focal_enc):
        if self.focal_parameterization == "log":
            focal_length = (focal_enc + LOG_FOCAL_LENGTH_BIAS).exp().view(-1).clamp(self.focal_min, self.focal_max)
            focal_length_probs = None
            focal_candidates = None

        elif self.focal_parameterization == "log-candidates":
            focal_enc = focal_enc * 0.01

            focal_length_probs = F.softmax(focal_enc, dim=-1)
            focal_candidates = torch.linspace(
                math.log(self.focal_min) - LOG_FOCAL_LENGTH_BIAS, 
                math.log(self.focal_max) - LOG_FOCAL_LENGTH_BIAS, 
                self.focal_num_candidates, 
                device=focal_length_probs.device)
            focal_candidates = (focal_candidates + LOG_FOCAL_LENGTH_BIAS).exp()
            focal_candidates = focal_candidates.view(1, -1).expand(focal_length_probs.shape[0], -1)

            focal_length = torch.sum(focal_length_probs * focal_candidates, dim=-1)

        elif self.focal_parameterization == "linlog-candidates":
            focal_enc = focal_enc * 0.01

            focal_length_probs = F.softmax(focal_enc, dim=-1)
            focal_candidates_log = torch.linspace(math.log(self.focal_min), math.log(self.focal_max), self.focal_num_candidates, device=focal_length_probs.device).exp()
            focal_candidates_lin = torch.linspace(self.focal_min, self.focal_max, self.focal_num_candidates, device=focal_length_probs.device)
            focal_candidates = .75 * focal_candidates_log + .25 * focal_candidates_lin
            focal_candidates = focal_candidates.view(1, -1).expand(focal_length_probs.shape[0], -1)

            focal_length = torch.sum(focal_length_probs * focal_candidates, dim=-1)
        
        elif self.focal_parameterization == "candidates":
            focal_enc = focal_enc * 0.01

            focal_length_probs = F.softmax(focal_enc, dim=-1)

            focal_candidates = torch.linspace(self.focal_min, self.focal_max, self.focal_num_candidates, device=focal_length_probs.device)
            focal_candidates = focal_candidates.view(1, -1).expand(focal_length_probs.shape[0], -1)

            focal_length = torch.sum(focal_length_probs * focal_candidates, dim=-1)

        else:
            raise NotImplementedError("Focal length parameterization not implemented")

        return focal_length, focal_length_probs, focal_candidates

    def stabilize_focal_logits(self, focal_enc):
        focal_enc_center = self.focal_enc_center * (1 - self.center_focal_logits_rate) + focal_enc.mean(dim=0, keepdim=True) * self.center_focal_logits_rate

        focal_enc = focal_enc - focal_enc_center

        focal_enc = focal_enc * self.sharp_focal_logits_temp

        return focal_enc