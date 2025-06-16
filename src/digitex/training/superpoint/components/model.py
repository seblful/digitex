import math

import torch
import torch.nn as nn
import torchvision.models as models


class HeatmapKeypointModel(nn.Module):
    def __init__(
        self,
        max_keypoints: int,
        backbone_out_channels: int,
        backbone_stride: int,
        deconv_channels: int,
        output_stride: int,
        freeze_backbone_params: bool,
    ) -> None:
        """
        Heatmap-based keypoint prediction model with visibility mask prediction.

        Args:
            backbone: Feature extraction backbone (e.g., ResNet50 without classification head)
            max_keypoints: Maximum number of keypoints to predict (defines output channels)
            backbone_out_channels: Number of output channels from the backbone
            deconv_channels: Number of channels in deconvolution layers
            output_stride: Target output stride relative to input (must be power of 2)
            freeze_backbone_params: Whether to freeze the backbone parameters
        """
        super().__init__()
        self.max_keypoints = max_keypoints

        # Verify output_stride is power of 2
        if not (output_stride & (output_stride - 1) == 0) and output_stride != 0:
            raise ValueError("output_stride must be a power of 2")

        # Create backbone
        self.backbone = self.create_backbone(
            freeze_backbone_params=freeze_backbone_params
        )

        # Create deconvolution layers
        self.deconv_layers = self.create_deconv_layers(
            backbone_out_channels, backbone_stride, deconv_channels, output_stride
        )

        # Create final prediction layer for heatmaps
        self.final_conv = nn.Conv2d(deconv_channels, max_keypoints, kernel_size=1)

        # Create visibility prediction head
        # Global average pooling followed by fully connected layers
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(
                1
            ),  # Global average pooling: (B, deconv_channels, H, W) -> (B, deconv_channels, 1, 1)
            nn.Flatten(),  # (B, deconv_channels, 1, 1) -> (B, deconv_channels)
            nn.Linear(deconv_channels, deconv_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(
                deconv_channels // 2, max_keypoints
            ),  # Output: (B, max_keypoints)
        )

        # Initialize weights
        self._initialize_weights()

    def create_deconv_layers(
        self,
        backbone_out_channels: int,
        backbone_stride: int,
        deconv_channels: int,
        output_stride: int,
    ) -> nn.Sequential:
        deconv_layers = nn.Sequential()
        in_channels = backbone_out_channels

        # Add deconvolution blocks
        num_deconv_layers = int(math.log2(backbone_stride // output_stride))
        for _ in range(num_deconv_layers):
            deconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    deconv_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            )
            deconv_layers.append(nn.BatchNorm2d(deconv_channels))
            deconv_layers.append(nn.ReLU(inplace=True))
            deconv_layers.append(nn.Dropout(0.1))
            in_channels = deconv_channels

        return deconv_layers

    def create_backbone(
        self,
        freeze_backbone_params: bool,
    ) -> nn.Module:
        # Load pretrained model
        resnet = getattr(models, "resnet50")(weights="ResNet50_Weights.DEFAULT")

        # Remove avgpool and fc layers
        backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze initial layers if specified
        if freeze_backbone_params:
            for param in backbone.parameters():
                param.requires_grad = False

        return backbone

    def _initialize_weights(self) -> None:
        # Initialize deconv layers
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize final heatmap layer
        nn.init.normal_(self.final_conv.weight, std=0.001)
        nn.init.constant_(self.final_conv.bias, 0)

        # Initialize visibility head
        for m in self.visibility_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tuple containing:
            - Heatmaps tensor of shape (B, max_keypoints, H//output_stride, W//output_stride)
              with values bounded between 0 and 1 (sigmoid activation applied)
            - Visibility masks tensor of shape (B, max_keypoints)
              with values bounded between 0 and 1 (sigmoid activation applied)
        """
        features = self.backbone(x)
        upsampled = self.deconv_layers(features)

        # Predict heatmaps
        heatmaps = self.final_conv(upsampled)
        heatmaps = torch.sigmoid(heatmaps)

        # Predict visibility masks
        visibility = self.visibility_head(upsampled)
        visibility = torch.sigmoid(visibility)

        return heatmaps, visibility
