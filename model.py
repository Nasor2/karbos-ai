"""Arquitectura DA-VIT (Dilation-based Attention Vision Transformer).

Variante Tiny para segmentación semántica de macerales de carbón.
4.95M parámetros, 4 clases de salida.
Basado en: https://github.com/Nasor2/coal-maceral-segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class DCSA(nn.Module):
    """Dilation-based Convolutional Self-Attention.

    Convoluciones dilatadas multi-escala (k=3,5,7) con atención de canales.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_sizes: list[int] | None = None,
        dilations: list[int] | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        if dilations is None:
            dilations = [1, 2, 3]

        self.si_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
            dilation=dilations[0],
            groups=in_channels, bias=False,
        )
        self.md_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_sizes[1],
            padding=(kernel_sizes[1] // 2) * dilations[1],
            dilation=dilations[1],
            groups=in_channels, bias=False,
        )
        self.ld_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_sizes[2],
            padding=(kernel_sizes[2] // 2) * dilations[2],
            dilation=dilations[2],
            groups=in_channels, bias=False,
        )
        self.channel_fuse = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(in_channels)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        fused = self.si_conv(x) + self.md_conv(x) + self.ld_conv(x)
        attn = self.act(self.norm(self.channel_fuse(fused)))
        attn_flat = attn.view(B, C, -1)
        attn_flat = self.attn_drop(attn_flat)
        attn_map = F.softmax(attn_flat, dim=-1).view(B, C, H, W)
        return self.proj_drop(attn_map * x)


class DCSABlock(nn.Module):
    """Bloque DCSA con residual connection y MLP (estilo Transformer)."""

    def __init__(
        self,
        in_channels: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        kernel_sizes: list[int] | None = None,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.dcsa = DCSA(in_channels, kernel_sizes, dilations, attn_drop, drop)
        hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, in_channels),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        x_flat = x_flat + self.dcsa(
            x_flat.permute(0, 2, 1).view(B, C, H, W)
        ).view(B, C, H * W).permute(0, 2, 1)
        x_flat = x_flat + self.mlp(self.norm1(x_flat))
        return x_flat.permute(0, 2, 1).view(B, C, H, W)


class PatchEmbed(nn.Module):
    """Patch embedding: Conv2d + LayerNorm."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 96, patch_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        B, C, H, W = x.shape
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class DAViTStage(nn.Module):
    """Etapa del encoder: N bloques DCSABlock + downsample opcional."""

    def __init__(
        self,
        in_channels: int,
        num_blocks: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        downsample: bool = True,
        kernel_sizes: list[int] | None = None,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            DCSABlock(in_channels, mlp_ratio, drop, attn_drop, kernel_sizes, dilations)
            for _ in range(num_blocks)
        ])
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 2, kernel_size=2, stride=2),
                nn.LayerNorm(in_channels * 2),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x_spatial = x.permute(0, 2, 1).view(B, C, H, W)
        for block in self.blocks:
            x_spatial = block(x_spatial)
        if self.downsample is not None:
            x_spatial = self.downsample[0](x_spatial)
            H, W = H // 2, W // 2
            x_spatial = x_spatial.flatten(2).permute(0, 2, 1)
            x_spatial = self.downsample[1](x_spatial)
            return x_spatial, H, W
        return x_spatial.flatten(2).permute(0, 2, 1), H, W


# Configuraciones por variante
DAVIT_CONFIGS = {
    "tiny": {"embed_dims": [32, 64, 128, 256], "num_blocks": [3, 3, 5, 2]},
    "small": {"embed_dims": [64, 128, 256, 512], "num_blocks": [2, 2, 4, 2]},
    "base": {"embed_dims": [64, 128, 256, 512], "num_blocks": [3, 3, 12, 3]},
}


class DAViT(nn.Module):
    """Backbone DA-VIT: PatchEmbed + 4 stages."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: list[int] | None = None,
        num_blocks: list[int] | None = None,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        patch_size: int = 4,
        kernel_sizes: list[int] | None = None,
        dilations: list[int] | None = None,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        if embed_dims is None:
            embed_dims = [32, 64, 128, 256]
        if num_blocks is None:
            num_blocks = [3, 3, 5, 2]
        if out_indices is None:
            out_indices = [0, 1, 2, 3]

        self.embed_dims = embed_dims
        self.out_indices = out_indices
        self.patch_embed = PatchEmbed(in_channels, embed_dims[0], patch_size)
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(DAViTStage(
                in_channels=embed_dims[i],
                num_blocks=num_blocks[i],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=(i < 3),
                kernel_sizes=kernel_sizes,
                dilations=dilations,
            ))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        B, C, H, W = x.shape
        x, H, W = self.patch_embed(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x, H, W = stage(x, H, W)
            if i in self.out_indices:
                out = x.permute(0, 2, 1).view(B, -1, H, W)
                outs.append(out)
        return outs


class DAViTHead(nn.Module):
    """FPN decoder + segmentation head."""

    def __init__(
        self,
        in_channels: list[int] | None = None,
        num_classes: int = 4,
        channels: int = 256,
        dropout_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels is None:
            in_channels = [64, 128, 256, 512]

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ic, channels, kernel_size=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for ic in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels
        ])
        self.cls_seg = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_ratio),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode="bilinear", align_corners=False,
            )
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return self.cls_seg(outs[0])


class DAViTModel(nn.Module):
    """DA-VIT completo: Backbone + FPN Head.

    Args:
        variant: Variante del modelo ('tiny', 'small', 'base').
        num_classes: Número de clases de salida (4 para macerales).
    """

    def __init__(
        self,
        variant: str = "tiny",
        num_classes: int = 4,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if variant not in DAVIT_CONFIGS:
            raise ValueError(f"Variante '{variant}' no soportada. Opciones: {list(DAVIT_CONFIGS)}")

        cfg = DAVIT_CONFIGS[variant]
        self.backbone = DAViT(
            embed_dims=cfg["embed_dims"],
            num_blocks=cfg["num_blocks"],
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            kernel_sizes=[3, 5, 7],
            dilations=[1, 2, 3],
        )
        out_channels = [
            cfg["embed_dims"][0] * 2,
            cfg["embed_dims"][1] * 2,
            cfg["embed_dims"][2] * 2,
            cfg["embed_dims"][3],
        ]
        self.head = DAViTHead(in_channels=out_channels, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
