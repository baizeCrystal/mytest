import os
import sys

import torch
import torch.nn as nn


def add_action_slot_repo(action_slot_repo: str):
    repo = os.path.abspath(action_slot_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    models_dir = os.path.join(repo, "models")
    scripts_dir = os.path.join(repo, "scripts")
    for path in [models_dir, scripts_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)


class X3DTemporalBackbone(nn.Module):
    """X3D feature extractor that returns temporal features [B, T, C]."""

    def __init__(self, out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=pretrained)
        self.blocks = self.model.blocks[:-1]
        self.in_channels = 192
        self.proj = nn.Sequential(
            nn.Conv3d(self.in_channels, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = int(out_dim)

    def forward(self, frames, return_spatial: bool = False):
        if isinstance(frames, list):
            x = torch.stack(frames, dim=0).permute(1, 2, 0, 3, 4)
        else:
            x = frames
        for block in self.blocks:
            x = block(x)
        x = self.proj(x)
        temporal = x.mean(dim=(-1, -2)).transpose(1, 2).contiguous()
        if not return_spatial:
            return temporal
        spatial = x.permute(0, 2, 3, 4, 1).contiguous()
        return {
            "temporal_features": temporal,
            "spatial_features": spatial,
        }


class SimpleCNNTemporalBackbone(nn.Module):
    """Small dependency-free temporal backbone for smoke tests and debugging."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = int(out_dim)

    def forward(self, frames, return_spatial: bool = False):
        if not isinstance(frames, list):
            frames = [frames[:, :, i] for i in range(frames.shape[2])]
        temporal_features = []
        spatial_features = []
        for frame in frames:
            fmap = self.encoder(frame)
            temporal_features.append(self.pool(fmap).flatten(1))
            if return_spatial:
                spatial_features.append(fmap.permute(0, 2, 3, 1).contiguous())
        temporal = torch.stack(temporal_features, dim=1)
        if not return_spatial:
            return temporal
        spatial = torch.stack(spatial_features, dim=1)
        return {
            "temporal_features": temporal,
            "spatial_features": spatial,
        }


class ActionSlotTemporalBackbone(nn.Module):
    """Reuse original ACTION_SLOT backbone and expose pre-head temporal features."""

    def __init__(self, args, action_slot_repo: str, num_actor_class: int):
        super().__init__()
        add_action_slot_repo(action_slot_repo)
        import action_slot

        self.model = action_slot.ACTION_SLOT(
            args,
            num_ego_class=0,
            num_actor_class=num_actor_class,
            num_slots=args.num_slots,
            box=False,
        )
        self.out_dim = int(args.channel)

    def forward(self, frames, return_spatial: bool = False):
        x = self._extract_3d_features(frames)
        x = self.model.conv3d(x)
        temporal = x.mean(dim=(-1, -2)).transpose(1, 2).contiguous()
        if not return_spatial:
            return temporal
        spatial = x.permute(0, 2, 3, 4, 1).contiguous()
        return {
            "temporal_features": temporal,
            "spatial_features": spatial,
        }

    def _extract_3d_features(self, frames):
        seq_len = len(frames)
        batch_size = frames[0].shape[0]
        height, width = frames[0].shape[2], frames[0].shape[3]

        if self.model.args.backbone == "r50":
            x = torch.stack(frames, dim=0)
            x = torch.reshape(x, (seq_len * batch_size, 3, height, width))
            x = self.model.resnet(x)
            _, channels, h, w = x.shape
            x = torch.reshape(x, (self.model.args.seq_len, batch_size, channels, h, w))
            return x.permute(1, 2, 0, 3, 4)

        if self.model.args.backbone == "slowfast":
            slow_x = [frames[i] for i in range(0, seq_len, 4)]
            fast = torch.stack(frames, dim=0).permute(1, 2, 0, 3, 4)
            slow = torch.stack(slow_x, dim=0).permute(1, 2, 0, 3, 4)
            x = [slow, fast]
            for block in self.model.resnet:
                x = block(x)
            x[1] = self.model.path_pool(x[1])
            return torch.cat((x[0], x[1]), dim=1)

        x = torch.stack(frames, dim=0).permute(1, 2, 0, 3, 4)
        for block in self.model.resnet:
            x = block(x)
        return x


def freeze_module(module: nn.Module, train_last_blocks: int = 0):
    for param in module.parameters():
        param.requires_grad = False

    blocks = getattr(module, "blocks", None)
    if blocks is not None and train_last_blocks > 0:
        for block in blocks[-train_last_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
